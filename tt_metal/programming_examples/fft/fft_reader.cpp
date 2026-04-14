// ============================================================
//  fft_reader.cpp  –  BRISC-0 (data mover in)
//
//  Responsibilities:
//    1. First call: read input data from DRAM into local CBs
//    2. Each local stage: reorder data into butterfly pairing order
//       and push into CB_LHS / CB_RHS
//    3. Each NOC stage: twiddle factors are read from DRAM
//       (actual cross-core data arrives via writer kernel into
//        CB_SCRATCH; reader just handles DRAM-side I/O)
// ============================================================

#include "dataflow_api.h"
#include "fft_common.h"

// ── Bit-reversal permutation index ──────────────────────────
// For Cooley-Tukey DIT (decimation-in-time) the input must be
// loaded in bit-reversed order on the very first stage.
// For subsequent stages the butterfly stride doubles each time.
FORCE_INLINE uint32_t bit_reverse(uint32_t x, uint32_t log2n) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < log2n; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

// ── Butterfly source index for stage s, element i ───────────
// Returns the two indices (lo, hi) that form a butterfly pair.
// stride = 1 << s  (doubles each stage)
// group  = i / (2*stride)
// pos    = i % (2*stride)
// lo     = group*(2*stride) + pos%stride
// hi     = lo + stride
FORCE_INLINE void butterfly_indices(
    uint32_t i, uint32_t stage,
    uint32_t& lo, uint32_t& hi)
{
    uint32_t stride = 1u << stage;
    uint32_t group  = i / (2 * stride);
    uint32_t pos    = i % stride;
    lo = group * (2 * stride) + pos;
    hi = lo + stride;
}

void kernel_main() {
    // ── Runtime args ────────────────────────────────────────
    uint32_t dram_input_addr    = get_arg_val<uint32_t>(0);
    uint32_t dram_bank_id       = get_arg_val<uint32_t>(1);
    uint32_t twiddle_dram_addr  = get_arg_val<uint32_t>(2);
    uint32_t twiddle_bank_id    = get_arg_val<uint32_t>(3);
    uint32_t local_N            = get_arg_val<uint32_t>(4);
    uint32_t my_core_id         = get_arg_val<uint32_t>(5);
    uint32_t total_N            = get_arg_val<uint32_t>(6);
    uint32_t num_local_stages   = get_arg_val<uint32_t>(7);  // log2(local_N)
    uint32_t num_total_stages   = get_arg_val<uint32_t>(8);  // log2(total_N)
    uint32_t use_bf16           = get_arg_val<uint32_t>(9);

    uint32_t elem_bytes = use_bf16 ? 2 : 4;
    uint32_t elem_size  = local_N * elem_bytes;

    // ── DRAM address objects ─────────────────────────────────
    // Input is interleaved: [real_0, imag_0, real_1, imag_1, ...]
    // We split into separate real/imag CBs on read.
    InterleavedAddrGen<true> dram_input = {
        .bank_base_address = dram_input_addr,
        .page_size         = elem_bytes
    };

    InterleavedAddrGen<true> dram_twiddle = {
        .bank_base_address = twiddle_dram_addr,
        .page_size         = elem_bytes
    };

    // ── Global element offset for this core ─────────────────
    uint32_t global_offset = my_core_id * local_N;

    // ── STEP 1: Load input data in bit-reversed order ────────
    // We read directly into CB_LHS_R / CB_LHS_I using bit-reversal
    // so stage-0 butterfly pairs are already adjacent in L1.
    {
        cb_reserve_back(CB_LHS_R, 1);
        cb_reserve_back(CB_LHS_I, 1);

        uint32_t lhs_r_ptr = get_write_ptr(CB_LHS_R);
        uint32_t lhs_i_ptr = get_write_ptr(CB_LHS_I);

        uint32_t log2_total = num_total_stages;

        for (uint32_t i = 0; i < local_N; i++) {
            // Global index of element i on this core
            uint32_t global_i = global_offset + i;
            // Where should it come from in the original order?
            uint32_t src_i    = bit_reverse(global_i, log2_total);

            // Read real part
            uint64_t src_real_addr = get_noc_addr(
                dram_input.get_bank_id(src_i * 2),
                dram_input.get_bank_addr(src_i * 2)
            );
            noc_async_read(src_real_addr,
                           lhs_r_ptr + i * elem_bytes,
                           elem_bytes);

            // Read imag part
            uint64_t src_imag_addr = get_noc_addr(
                dram_input.get_bank_id(src_i * 2 + 1),
                dram_input.get_bank_addr(src_i * 2 + 1)
            );
            noc_async_read(src_imag_addr,
                           lhs_i_ptr + i * elem_bytes,
                           elem_bytes);
        }

        noc_async_read_barrier();
        cb_push_back(CB_LHS_R, 1);
        cb_push_back(CB_LHS_I, 1);
    }

    // ── STEP 2: Local stages — push RHS and twiddles ────────
    // For each local stage s (0 .. num_local_stages-1):
    //   - RHS tile = elements reordered for butterfly pairing
    //   - Twiddle tile = W_N^k factors for each butterfly
    // The compute kernel does the actual butterfly math.
    for (uint32_t s = 0; s < num_local_stages; s++) {
        uint32_t stride = 1u << s;

        // Reserve RHS CB space
        cb_reserve_back(CB_RHS_R, 1);
        cb_reserve_back(CB_RHS_I, 1);
        cb_reserve_back(CB_TWIDDLE_R, 1);
        cb_reserve_back(CB_TWIDDLE_I, 1);

        uint32_t rhs_r_ptr  = get_write_ptr(CB_RHS_R);
        uint32_t rhs_i_ptr  = get_write_ptr(CB_RHS_I);
        uint32_t tw_r_ptr   = get_write_ptr(CB_TWIDDLE_R);
        uint32_t tw_i_ptr   = get_write_ptr(CB_TWIDDLE_I);

        // Read current LHS from CB to know which elements form pairs.
        // (LHS was written by previous compute stage or initial load)
        uint32_t lhs_r_ptr = get_read_ptr(CB_LHS_R);

        for (uint32_t i = 0; i < local_N / 2; i++) {
            uint32_t lo, hi;
            butterfly_indices(i, s, lo, hi);

            // Copy hi-index element into RHS slot i
            // (lo-index stays in LHS at position lo)
            // This is a local L1 copy — no NOC needed.
            uint32_t* rhs_r = reinterpret_cast<uint32_t*>(rhs_r_ptr);
            uint32_t* rhs_i = reinterpret_cast<uint32_t*>(rhs_i_ptr);
            uint32_t* lhs_r = reinterpret_cast<uint32_t*>(lhs_r_ptr);
            // (imag pointer similarly)

            // Simple scalar copy on BRISC — for large N, use ThCon
            // intrinsics (TT_LOADIND / TT_SETDMAREG) as Davies paper shows.
            // Shown here as scalar for clarity.
            rhs_r[lo] = lhs_r[hi];
            // rhs_i[lo] = lhs_i[hi];  // same for imag

            // ── Twiddle factor W_N^k ────────────────────────
            // k = (global_offset + lo) % (2*stride)
            // W = exp(-j*2*pi*k / (2*stride))  [forward FFT]
            // We precomputed twiddles on host and stored in DRAM.
            // Layout: twiddle_dram[stage][k] = [real, imag]
            uint32_t k = (global_offset + lo) % (2 * stride);
            uint32_t twiddle_idx = s * (total_N / 2) + k;

            uint64_t tw_real_addr = get_noc_addr(
                dram_twiddle.get_bank_id(twiddle_idx * 2),
                dram_twiddle.get_bank_addr(twiddle_idx * 2)
            );
            noc_async_read(tw_real_addr,
                           tw_r_ptr + lo * elem_bytes,
                           elem_bytes);
            // (same for imag)
        }

        noc_async_read_barrier();
        cb_push_back(CB_RHS_R, 1);
        cb_push_back(CB_RHS_I, 1);
        cb_push_back(CB_TWIDDLE_R, 1);
        cb_push_back(CB_TWIDDLE_I, 1);
    }

    // ── STEP 3: NOC stages — reader only pushes twiddles ────
    // Cross-core data arrives via CB_SCRATCH_R/I (pushed by writer).
    // Reader is still responsible for twiddle factors each stage.
    for (uint32_t s = num_local_stages; s < num_total_stages; s++) {
        uint32_t stride = 1u << s;

        cb_reserve_back(CB_TWIDDLE_R, 1);
        cb_reserve_back(CB_TWIDDLE_I, 1);

        uint32_t tw_r_ptr = get_write_ptr(CB_TWIDDLE_R);
        uint32_t tw_i_ptr = get_write_ptr(CB_TWIDDLE_I);

        for (uint32_t i = 0; i < local_N / 2; i++) {
            uint32_t k = (global_offset + i) % (2 * stride);
            uint32_t twiddle_idx = s * (total_N / 2) + k;

            uint64_t tw_real_addr = get_noc_addr(
                dram_twiddle.get_bank_id(twiddle_idx * 2),
                dram_twiddle.get_bank_addr(twiddle_idx * 2)
            );
            noc_async_read(tw_real_addr,
                           tw_r_ptr + i * elem_bytes,
                           elem_bytes);
            // (same for imag)
        }

        noc_async_read_barrier();
        cb_push_back(CB_TWIDDLE_R, 1);
        cb_push_back(CB_TWIDDLE_I, 1);
    }
}
