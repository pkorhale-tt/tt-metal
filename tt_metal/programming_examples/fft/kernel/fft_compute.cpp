// ============================================================
//  fft_compute.cpp  –  TRISC (compute engine)
//
//  Executes Cooley-Tukey butterfly for both local and NOC stages.
//
//  Butterfly operation (DIT, radix-2):
//    Given complex pair (a, b) and twiddle W = e^{-j*2*pi*k/N}:
//      out_a = a + W*b
//      out_b = a - W*b
//    Where W*b = (b_r*W_r - b_i*W_i) + j*(b_r*W_i + b_i*W_r)
//
//  We use the SFPU (vector unit) throughout since FFT is NOT
//  a matrix multiply — the FPU (matrix unit) doesn't help here.
//  This matches Davies paper observation: "vector vs matrix units
//  comparable" — vector is the right choice for 1D FFT.
//
//  For IFFT: same code, but twiddle factors are conjugated
//  (W_i negated) and output is divided by N (scale after last stage).
// ============================================================

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_include.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary_api.h"
#include "fft_common.h"

// ── Complex multiply via SFPU ────────────────────────────────
// Computes (a_r + j*a_i) * (b_r + j*b_i)
//   real part = a_r*b_r - a_i*b_i
//   imag part = a_r*b_i + a_i*b_r
//
// Parameters are CB indices.  Result lands in cb_out_r / cb_out_i.
// Caller must have called cb_reserve_back on output CBs.
ALWI void complex_mul_sfpu(
    uint32_t cb_a_r, uint32_t cb_a_i,   // inputs (will be popped)
    uint32_t cb_b_r, uint32_t cb_b_i,   // inputs (will be popped)
    uint32_t cb_tmp_r, uint32_t cb_tmp_i,
    uint32_t cb_out_r, uint32_t cb_out_i)
{
    // ── real = a_r*b_r - a_i*b_i ────────────────────────────

    // Step 1: a_r * b_r → cb_out_r (partial)
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_a_r);
    copy_tile(cb_a_r, 0, 0);   // dst[0] = a_r

    mul_tiles_init(cb_a_r, cb_b_r);
    mul_tiles(cb_a_r, cb_b_r, 0, 0, 0);  // dst[0] = a_r * b_r
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out_r);
    tile_regs_release();

    // Step 2: a_i * b_i → cb_tmp_r
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_a_i);
    copy_tile(cb_a_i, 0, 0);

    mul_tiles_init(cb_a_i, cb_b_i);
    mul_tiles(cb_a_i, cb_b_i, 0, 0, 0);  // dst[0] = a_i * b_i
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_tmp_r);
    tile_regs_release();

    // Step 3: real = cb_out_r - cb_tmp_r
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_out_r);
    copy_tile(cb_out_r, 0, 0);

    sub_tiles_init(cb_out_r, cb_tmp_r);
    sub_tiles(cb_out_r, cb_tmp_r, 0, 0, 0);  // dst[0] = a_r*b_r - a_i*b_i
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out_r);
    tile_regs_release();

    // ── imag = a_r*b_i + a_i*b_r ────────────────────────────

    // Step 4: a_r * b_i → cb_out_i (partial)
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_a_r);
    copy_tile(cb_a_r, 0, 0);

    mul_tiles_init(cb_a_r, cb_b_i);
    mul_tiles(cb_a_r, cb_b_i, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out_i);
    tile_regs_release();

    // Step 5: a_i * b_r → cb_tmp_i
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_a_i);
    copy_tile(cb_a_i, 0, 0);

    mul_tiles_init(cb_a_i, cb_b_r);
    mul_tiles(cb_a_i, cb_b_r, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_tmp_i);
    tile_regs_release();

    // Step 6: imag = cb_out_i + cb_tmp_i
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_out_i);
    copy_tile(cb_out_i, 0, 0);

    add_tiles_init(cb_out_i, cb_tmp_i);
    add_tiles(cb_out_i, cb_tmp_i, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out_i);
    tile_regs_release();
}

// ── Full butterfly: out_a = lhs + W*rhs,  out_b = lhs - W*rhs ──
// lhs = (cb_lhs_r, cb_lhs_i)
// rhs = (cb_rhs_r, cb_rhs_i)
// twiddle = (cb_tw_r, cb_tw_i)
// outputs land in CB_OUT_R / CB_OUT_I
//   first half  = out_a (add results)
//   second half = out_b (sub results)
// Caller must cb_reserve_back(CB_OUT_R/I, 1) before calling.
ALWI void butterfly(
    uint32_t cb_lhs_r, uint32_t cb_lhs_i,
    uint32_t cb_rhs_r, uint32_t cb_rhs_i,
    uint32_t cb_tw_r,  uint32_t cb_tw_i,
    uint32_t cb_tmp_r, uint32_t cb_tmp_i,  // scratch
    uint32_t cb_wr_r,  uint32_t cb_wr_i,   // W*rhs intermediate
    uint32_t cb_out_r, uint32_t cb_out_i)
{
    // ── Step A: W_r_i = W * rhs ─────────────────────────────
    cb_reserve_back(cb_wr_r, 1);
    cb_reserve_back(cb_wr_i, 1);

    complex_mul_sfpu(
        cb_tw_r,  cb_tw_i,
        cb_rhs_r, cb_rhs_i,
        cb_tmp_r, cb_tmp_i,
        cb_wr_r,  cb_wr_i);

    cb_push_back(cb_wr_r, 1);
    cb_push_back(cb_wr_i, 1);

    cb_wait_front(cb_wr_r, 1);
    cb_wait_front(cb_wr_i, 1);
    cb_wait_front(cb_lhs_r, 1);
    cb_wait_front(cb_lhs_i, 1);

    // ── Step B: out_a = lhs + W*rhs ─────────────────────────
    // Real part
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_lhs_r);
    copy_tile(cb_lhs_r, 0, 0);
    add_tiles_init(cb_lhs_r, cb_wr_r);
    add_tiles(cb_lhs_r, cb_wr_r, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out_r);       // out_a real → first half of output tile
    tile_regs_release();

    // Imag part
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_lhs_i);
    copy_tile(cb_lhs_i, 0, 0);
    add_tiles_init(cb_lhs_i, cb_wr_i);
    add_tiles(cb_lhs_i, cb_wr_i, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out_i);
    tile_regs_release();

    // ── Step C: out_b = lhs - W*rhs ─────────────────────────
    // Real part
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_lhs_r);
    copy_tile(cb_lhs_r, 0, 0);
    sub_tiles_init(cb_lhs_r, cb_wr_r);
    sub_tiles(cb_lhs_r, cb_wr_r, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    // Pack into second half of output tile (offset by local_N/2 elements)
    // In practice: use a second output CB or pack with dst offset.
    // Here we pack into the same CB at tile index 1 (requires 2-tile CB).
    pack_tile(0, cb_out_r, 1);
    tile_regs_release();

    // Imag part
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_lhs_i);
    copy_tile(cb_lhs_i, 0, 0);
    sub_tiles_init(cb_lhs_i, cb_wr_i);
    sub_tiles(cb_lhs_i, cb_wr_i, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out_i, 1);
    tile_regs_release();

    // Pop all inputs
    cb_pop_front(cb_lhs_r, 1);
    cb_pop_front(cb_lhs_i, 1);
    cb_pop_front(cb_rhs_r, 1);
    cb_pop_front(cb_rhs_i, 1);
    cb_pop_front(cb_tw_r,  1);
    cb_pop_front(cb_tw_i,  1);
    cb_pop_front(cb_wr_r,  1);
    cb_pop_front(cb_wr_i,  1);
}

// ── IFFT scale: multiply by 1/N after last stage ─────────────
ALWI void scale_by_inv_N(
    uint32_t cb_r, uint32_t cb_i, float inv_N)
{
    cb_wait_front(cb_r, 1);
    cb_wait_front(cb_i, 1);
    cb_reserve_back(CB_OUT_R, 1);
    cb_reserve_back(CB_OUT_I, 1);

    // Load 1/N as a scalar into dst and multiply
    // Using SFPU mul_unary (multiplies every element by scalar)
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_r);
    copy_tile(cb_r, 0, 0);
    mul_unary_tile_init();
    mul_unary_tile(0, inv_N);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, CB_OUT_R);
    tile_regs_release();

    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_i);
    copy_tile(cb_i, 0, 0);
    mul_unary_tile_init();
    mul_unary_tile(0, inv_N);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, CB_OUT_I);
    tile_regs_release();

    cb_pop_front(cb_r, 1);
    cb_pop_front(cb_i, 1);
    cb_push_back(CB_OUT_R, 1);
    cb_push_back(CB_OUT_I, 1);
}

// ── Temporary CBs used inside butterfly ─────────────────────
// Defined as compile-time constants; must be registered in host.
constexpr uint32_t CB_TMP_R  = 11;
constexpr uint32_t CB_TMP_I  = 12;
constexpr uint32_t CB_WR_R   = 13;  // W*rhs intermediate real
constexpr uint32_t CB_WR_I   = 14;  // W*rhs intermediate imag

void MAIN {
    uint32_t num_local_stages = get_compile_time_arg_val(0);
    uint32_t num_noc_stages   = get_compile_time_arg_val(1);
    uint32_t is_ifft          = get_compile_time_arg_val(2); // 1=IFFT
    uint32_t total_N          = get_compile_time_arg_val(3);
    uint32_t num_total_stages = num_local_stages + num_noc_stages;

    mm_init();  // required once before any math ops

    // ── LOCAL STAGES ─────────────────────────────────────────
    // Data is already in CB_LHS_R/I (loaded by reader in bit-reversed order).
    // Reader also pushed CB_RHS_R/I and CB_TWIDDLE_R/I for each stage.
    for (uint32_t s = 0; s < num_local_stages; s++) {
        cb_wait_front(CB_LHS_R,     1);
        cb_wait_front(CB_LHS_I,     1);
        cb_wait_front(CB_RHS_R,     1);
        cb_wait_front(CB_RHS_I,     1);
        cb_wait_front(CB_TWIDDLE_R, 1);
        cb_wait_front(CB_TWIDDLE_I, 1);

        cb_reserve_back(CB_OUT_R, 1);
        cb_reserve_back(CB_OUT_I, 1);

        butterfly(
            CB_LHS_R,     CB_LHS_I,
            CB_RHS_R,     CB_RHS_I,
            CB_TWIDDLE_R, CB_TWIDDLE_I,
            CB_TMP_R,     CB_TMP_I,
            CB_WR_R,      CB_WR_I,
            CB_OUT_R,     CB_OUT_I);

        cb_push_back(CB_OUT_R, 1);
        cb_push_back(CB_OUT_I, 1);

        // Rotate: OUT → LHS for next stage
        // Writer kernel pops CB_OUT and the next iteration's
        // cb_wait_front(CB_LHS, 1) blocks until reader re-fills.
        // (In the single-copy optimization, OUT is directly the
        //  next stage's LHS — this is the Davies paper optimization.)
    }

    // ── NOC STAGES ───────────────────────────────────────────
    // After local stages, CB_OUT has our locally-computed results.
    // Writer sends those out to peers and fills CB_SCRATCH with
    // incoming data. Writer then pushes CB_SYNC to unblock us.
    for (uint32_t s = 0; s < num_noc_stages; s++) {
        // Wait for writer to confirm all peer data is in CB_SCRATCH
        cb_wait_front(CB_SYNC, 1);
        cb_pop_front(CB_SYNC, 1);

        // At this point:
        //   CB_OUT_R/I  = our local results from previous stage
        //   CB_SCRATCH_R/I = data received from our butterfly partner core
        //
        // We treat CB_OUT as LHS and CB_SCRATCH as RHS (or vice versa
        // depending on whether our global index is the 'lo' or 'hi' element).
        // For power-of-2 partitioning: lower-id core always holds LHS.
        cb_wait_front(CB_OUT_R,     1);
        cb_wait_front(CB_OUT_I,     1);
        cb_wait_front(CB_SCRATCH_R, 1);
        cb_wait_front(CB_SCRATCH_I, 1);
        cb_wait_front(CB_TWIDDLE_R, 1);
        cb_wait_front(CB_TWIDDLE_I, 1);

        cb_reserve_back(CB_OUT_R, 1);
        cb_reserve_back(CB_OUT_I, 1);

        butterfly(
            CB_OUT_R,     CB_OUT_I,      // LHS = our data
            CB_SCRATCH_R, CB_SCRATCH_I,  // RHS = received from partner
            CB_TWIDDLE_R, CB_TWIDDLE_I,
            CB_TMP_R,     CB_TMP_I,
            CB_WR_R,      CB_WR_I,
            CB_OUT_R,     CB_OUT_I);

        cb_push_back(CB_OUT_R, 1);
        cb_push_back(CB_OUT_I, 1);
    }

    // ── IFFT: scale by 1/N on last stage ─────────────────────
    if (is_ifft) {
        float inv_N = 1.0f / static_cast<float>(total_N);
        scale_by_inv_N(CB_OUT_R, CB_OUT_I, inv_N);
    }
}
