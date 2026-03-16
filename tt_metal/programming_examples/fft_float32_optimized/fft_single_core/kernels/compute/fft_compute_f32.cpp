// fft_compute_f32.cpp  — OPTIMAL v3: FPU butterfly + PACK/MATH/UNPACK shuffle
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// V3 improvement over V2:
//   V2 shuffled inter-stage data using RISCV_1 scalar writes (4 B/write, 1 thread).
//   V3 moves the shuffle into the COMPUTE kernel and parallelises it across
//   the three Tensix sub-engines using TT_LOADIND / TT_STOREIND:
//
//     PACK   thread → scatter into cb_even_r
//     MATH   thread → scatter into cb_even_i    } all three run simultaneously
//     UNPACK thread → scatter into cb_odd_r
//     PACK   thread → scatter into cb_odd_i     (sequential on PACK after above)
//
//   Effective copy bandwidth: ~3× vs RISCV_1 scalar.
//   Writer kernel is now trivial — waits for final output, writes to DRAM once.
//
// SHUFFLE FORMULA (unchanged from V2, just executed on sub-engines now):
//   After stage s: m=1<<(s+1), half_m=m>>1, log2m=s+1, m_mask=m-1
//   m2=m<<1, half_m2=m2>>1, G2=N/m2
//   For new_even[dst] and new_odd[dst] (dst=0..half_N-1):
//     for g2 in [0,G2), for j2 in [0,half_m2):
//       f_e = g2*m2 + j2
//       f_o = f_e + half_m2
//       g_old_e = f_e >> log2m,  offset_e = f_e & m_mask
//       g_old_o = f_o >> log2m,  offset_o = f_o & m_mask
//       new_even[dst] = (offset_e < half_m) ? out0[g_old_e*half_m+offset_e]
//                                            : out1[g_old_e*half_m+offset_e-half_m]
//       new_odd[dst]  = (offset_o < half_m) ? out0[g_old_o*half_m+offset_o]
//                                            : out1[g_old_o*half_m+offset_o-half_m]
//       dst++
//
// CB map:
//   0  cb_even_r     stage input even real
//   1  cb_even_i     stage input even imag
//   2  cb_odd_r      stage input odd  real
//   3  cb_odd_i      stage input odd  imag
//   4  cb_tw_r       expanded twiddle real (reader fills per stage)
//   5  cb_tw_i       expanded twiddle imag
//  16  cb_out0_r     butterfly sum  real  → writer (last stage) or self-shuffle
//  17  cb_out0_i     butterfly sum  imag
//  18  cb_out1_r     butterfly diff real
//  19  cb_out1_i     butterfly diff imag
//  20  cb_tmp0       scratch
//  21  cb_tmp1       scratch
//  22  cb_tw_odd_r   scratch
//  23  cb_tw_odd_i   scratch

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/copy_dest_values.h"

// Hardware DMA register instructions for PACK/MATH/UNPACK parallel copy
#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"

// ── Low-level 32-bit load: reads one float (4 bytes) from an L1 address
// into hardware data register DATA_REG using DMA register ADDR_REG/BASE_REG.
template<int ADDR_OFF_REG, int BASE_REG, int DATA_REG>
inline void ld32(uint32_t addr) {
    uint32_t base  = addr / 16;
    uint32_t off   = addr - base * 16;
    TT_SETDMAREG(0, LOWER_HALFWORD(off),  0, LO_16(ADDR_OFF_REG));
    TT_SETDMAREG(0, UPPER_HALFWORD(off),  0, HI_16(ADDR_OFF_REG));
    TT_SETDMAREG(0, LOWER_HALFWORD(base), 0, LO_16(BASE_REG));
    TT_SETDMAREG(0, UPPER_HALFWORD(base), 0, HI_16(BASE_REG));
    TT_LOADIND(p_ind::LD_32bit, LO_16(ADDR_OFF_REG), p_ind::INC_NONE, DATA_REG, BASE_REG);
}

// ── Low-level 32-bit store: writes DATA_REG (one float) to an L1 address.
template<int ADDR_OFF_REG, int BASE_REG, int DATA_REG>
inline void st32(uint32_t addr) {
    uint32_t base  = addr / 16;
    uint32_t off   = addr - base * 16;
    TT_SETDMAREG(0, LOWER_HALFWORD(off),  0, LO_16(ADDR_OFF_REG));
    TT_SETDMAREG(0, UPPER_HALFWORD(off),  0, HI_16(ADDR_OFF_REG));
    TT_SETDMAREG(0, LOWER_HALFWORD(base), 0, LO_16(BASE_REG));
    TT_SETDMAREG(0, UPPER_HALFWORD(base), 0, HI_16(BASE_REG));
    TT_STOREIND(1, 0, p_ind::LD_32bit, LO_16(ADDR_OFF_REG),
                p_ind::INC_NONE, DATA_REG, BASE_REG);
}

// ── Parallel shuffle: copies one element to all 4 destination CBs simultaneously.
// Each sub-engine handles one CB:
//   PACK   → dst_er (cb_even_r)
//   MATH   → dst_ei (cb_even_i)
//   UNPACK → dst_or (cb_odd_r)
//   PACK   → dst_oi (cb_odd_i)   [runs after PACK finishes dst_er]
//
// DMA register allocation (no overlap between sub-engines):
//   PACK:   regs 16-19 for even_r, 28-31 for odd_i
//   MATH:   regs 20-23 for even_i
//   UNPACK: regs 24-27 for odd_r
inline void shuffle_element(
    uint32_t src_e_r, uint32_t src_e_i,   // source addresses for new_even
    uint32_t src_o_r, uint32_t src_o_i,   // source addresses for new_odd
    uint32_t dst_er,  uint32_t dst_ei,    // destination addresses
    uint32_t dst_or,  uint32_t dst_oi)
{
    // All three sub-engines load their source and store to destination in parallel.
    // TTI_STALLWAIT after each pair ensures the store completes before moving on.
    PACK((
        ld32<16,17,0>(src_e_r);
        st32<16,17,0>(dst_er);
        TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::PACK);
    ));
    MATH((
        ld32<20,21,4>(src_e_i);
        st32<20,21,4>(dst_ei);
        TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::MATH);
    ));
    UNPACK((
        ld32<24,25,8>(src_o_r);
        st32<24,25,8>(dst_or);
        TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::UNPACK);
    ));
    // 4th CB (odd_i) on PACK — runs after even_r PACK completes
    PACK((
        ld32<28,29,12>(src_o_i);
        st32<28,29,12>(dst_oi);
        TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::PACK);
    ));
}

void kernel_main() {
    const uint32_t num_stages      = get_arg_val<uint32_t>(0);
    const uint32_t tiles_per_stage = get_arg_val<uint32_t>(1);
    const uint32_t half_N          = get_arg_val<uint32_t>(2);  // N/2 elements

    constexpr uint32_t cb_even_r   = 0;
    constexpr uint32_t cb_even_i   = 1;
    constexpr uint32_t cb_odd_r    = 2;
    constexpr uint32_t cb_odd_i    = 3;
    constexpr uint32_t cb_tw_r     = 4;
    constexpr uint32_t cb_tw_i     = 5;
    constexpr uint32_t cb_out0_r   = 16;
    constexpr uint32_t cb_out0_i   = 17;
    constexpr uint32_t cb_out1_r   = 18;
    constexpr uint32_t cb_out1_i   = 19;
    constexpr uint32_t cb_tmp0     = 20;
    constexpr uint32_t cb_tmp1     = 21;
    constexpr uint32_t cb_tw_odd_r = 22;
    constexpr uint32_t cb_tw_odd_i = 23;

    constexpr uint32_t ELEM = sizeof(float);  // 4 bytes

    binary_op_init_common(cb_even_r, cb_odd_r, cb_tmp0);

    for (uint32_t stage = 0; stage < num_stages; stage++) {
        const bool is_last = (stage == num_stages - 1);

        for (uint32_t t = 0; t < tiles_per_stage; t++) {

            // ── Wait for inputs ───────────────────────────────────
            cb_wait_front(cb_tw_r,   1);
            cb_wait_front(cb_tw_i,   1);
            cb_wait_front(cb_odd_r,  1);
            cb_wait_front(cb_odd_i,  1);
            cb_wait_front(cb_even_r, 1);
            cb_wait_front(cb_even_i, 1);

            // ── t_r = tw_r*odd_r − tw_i*odd_i ────────────────────
            cb_reserve_back(cb_tmp0, 1);
            tile_regs_acquire();
            mul_tiles_init(cb_tw_r, cb_odd_r, cb_tmp0);
            mul_tiles(cb_tw_r, cb_odd_r, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            pack_tile(0, cb_tmp0); tile_regs_release();
            cb_push_back(cb_tmp0, 1);

            cb_reserve_back(cb_tmp1, 1);
            tile_regs_acquire();
            mul_tiles_init(cb_tw_i, cb_odd_i, cb_tmp1);
            mul_tiles(cb_tw_i, cb_odd_i, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            pack_tile(0, cb_tmp1); tile_regs_release();
            cb_push_back(cb_tmp1, 1);

            cb_wait_front(cb_tmp0, 1); cb_wait_front(cb_tmp1, 1);
            cb_reserve_back(cb_tw_odd_r, 1);
            tile_regs_acquire();
            sub_tiles_init(cb_tmp0, cb_tmp1, cb_tw_odd_r);
            sub_tiles(cb_tmp0, cb_tmp1, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            pack_tile(0, cb_tw_odd_r); tile_regs_release();
            cb_push_back(cb_tw_odd_r, 1);
            cb_pop_front(cb_tmp0, 1); cb_pop_front(cb_tmp1, 1);

            // ── t_i = tw_r*odd_i + tw_i*odd_r ────────────────────
            cb_reserve_back(cb_tmp0, 1);
            tile_regs_acquire();
            mul_tiles_init(cb_tw_r, cb_odd_i, cb_tmp0);
            mul_tiles(cb_tw_r, cb_odd_i, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            pack_tile(0, cb_tmp0); tile_regs_release();
            cb_push_back(cb_tmp0, 1);

            cb_reserve_back(cb_tmp1, 1);
            tile_regs_acquire();
            mul_tiles_init(cb_tw_i, cb_odd_r, cb_tmp1);
            mul_tiles(cb_tw_i, cb_odd_r, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            pack_tile(0, cb_tmp1); tile_regs_release();
            cb_push_back(cb_tmp1, 1);

            cb_wait_front(cb_tmp0, 1); cb_wait_front(cb_tmp1, 1);
            cb_reserve_back(cb_tw_odd_i, 1);
            tile_regs_acquire();
            add_tiles_init(cb_tmp0, cb_tmp1, cb_tw_odd_i);
            add_tiles(cb_tmp0, cb_tmp1, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            pack_tile(0, cb_tw_odd_i); tile_regs_release();
            cb_push_back(cb_tw_odd_i, 1);
            cb_pop_front(cb_tmp0, 1); cb_pop_front(cb_tmp1, 1);

            cb_wait_front(cb_tw_odd_r, 1); cb_wait_front(cb_tw_odd_i, 1);

            // ── out0 = even + t ───────────────────────────────────
            cb_reserve_back(cb_out0_r, 1);
            tile_regs_acquire();
            add_tiles_init(cb_even_r, cb_tw_odd_r, cb_out0_r);
            add_tiles(cb_even_r, cb_tw_odd_r, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            pack_tile(0, cb_out0_r); tile_regs_release();
            cb_push_back(cb_out0_r, 1);

            cb_reserve_back(cb_out0_i, 1);
            tile_regs_acquire();
            add_tiles_init(cb_even_i, cb_tw_odd_i, cb_out0_i);
            add_tiles(cb_even_i, cb_tw_odd_i, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            pack_tile(0, cb_out0_i); tile_regs_release();
            cb_push_back(cb_out0_i, 1);

            // ── out1 = even − t ───────────────────────────────────
            cb_reserve_back(cb_out1_r, 1);
            tile_regs_acquire();
            sub_tiles_init(cb_even_r, cb_tw_odd_r, cb_out1_r);
            sub_tiles(cb_even_r, cb_tw_odd_r, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            pack_tile(0, cb_out1_r); tile_regs_release();
            cb_push_back(cb_out1_r, 1);

            cb_reserve_back(cb_out1_i, 1);
            tile_regs_acquire();
            sub_tiles_init(cb_even_i, cb_tw_odd_i, cb_out1_i);
            sub_tiles(cb_even_i, cb_tw_odd_i, 0, 0, 0);
            tile_regs_commit(); tile_regs_wait();
            pack_tile(0, cb_out1_i); tile_regs_release();
            cb_push_back(cb_out1_i, 1);

            // ── Pop consumed inputs ───────────────────────────────
            cb_pop_front(cb_tw_r,     1);
            cb_pop_front(cb_tw_i,     1);
            cb_pop_front(cb_odd_r,    1);
            cb_pop_front(cb_odd_i,    1);
            cb_pop_front(cb_even_r,   1);
            cb_pop_front(cb_even_i,   1);
            cb_pop_front(cb_tw_odd_r, 1);
            cb_pop_front(cb_tw_odd_i, 1);

            // ── Wait for out0/out1 to be ready for shuffle/output ─
            cb_wait_front(cb_out0_r, 1); cb_wait_front(cb_out0_i, 1);
            cb_wait_front(cb_out1_r, 1); cb_wait_front(cb_out1_i, 1);

            if (!is_last) {
                // ── PACK/MATH/UNPACK parallel L1 shuffle ─────────
                //
                // Source CB base addresses
                volatile uint *p0r_ptr, *p0i_ptr, *p1r_ptr, *p1i_ptr;
                cb_get_tile(cb_out0_r, 0, &p0r_ptr);
                cb_get_tile(cb_out0_i, 0, &p0i_ptr);
                cb_get_tile(cb_out1_r, 0, &p1r_ptr);
                cb_get_tile(cb_out1_i, 0, &p1i_ptr);
                const uint32_t src0r = reinterpret_cast<uint32_t>(p0r_ptr);
                const uint32_t src0i = reinterpret_cast<uint32_t>(p0i_ptr);
                const uint32_t src1r = reinterpret_cast<uint32_t>(p1r_ptr);
                const uint32_t src1i = reinterpret_cast<uint32_t>(p1i_ptr);

                // Destination CB base addresses
                cb_reserve_back(cb_even_r, tiles_per_stage);
                cb_reserve_back(cb_even_i, tiles_per_stage);
                cb_reserve_back(cb_odd_r,  tiles_per_stage);
                cb_reserve_back(cb_odd_i,  tiles_per_stage);

                volatile uint *der_ptr, *dei_ptr, *dor_ptr, *doi_ptr;
                cb_get_tile(cb_even_r, 0, &der_ptr);
                cb_get_tile(cb_even_i, 0, &dei_ptr);
                cb_get_tile(cb_odd_r,  0, &dor_ptr);
                cb_get_tile(cb_odd_i,  0, &doi_ptr);
                const uint32_t dst_er = reinterpret_cast<uint32_t>(der_ptr);
                const uint32_t dst_ei = reinterpret_cast<uint32_t>(dei_ptr);
                const uint32_t dst_or = reinterpret_cast<uint32_t>(dor_ptr);
                const uint32_t dst_oi = reinterpret_cast<uint32_t>(doi_ptr);

                // Shuffle index parameters
                const uint32_t log2m   = stage + 1;
                const uint32_t m       = 1u << log2m;
                const uint32_t half_m  = m >> 1;
                const uint32_t m_mask  = m - 1u;
                const uint32_t m2      = m << 1;
                const uint32_t half_m2 = m2 >> 1;
                const uint32_t G2      = half_N / half_m2;

                uint32_t dst = 0;
                for (uint32_t g2 = 0; g2 < G2; g2++) {
                    const uint32_t base_e = g2 * m2;
                    const uint32_t base_o = base_e + half_m2;
                    for (uint32_t j2 = 0; j2 < half_m2; j2++) {

                        // Compute source index for new_even
                        const uint32_t fe     = base_e + j2;
                        const uint32_t ge     = fe >> log2m;
                        const uint32_t offe   = fe & m_mask;
                        const uint32_t idx_e  = ge * half_m + (offe < half_m ? offe : offe - half_m);
                        const uint32_t src_er = (offe < half_m ? src0r : src1r) + idx_e * ELEM;
                        const uint32_t src_ei = (offe < half_m ? src0i : src1i) + idx_e * ELEM;

                        // Compute source index for new_odd
                        const uint32_t fo     = base_o + j2;
                        const uint32_t go_    = fo >> log2m;
                        const uint32_t offo   = fo & m_mask;
                        const uint32_t idx_o  = go_ * half_m + (offo < half_m ? offo : offo - half_m);
                        const uint32_t src_or = (offo < half_m ? src0r : src1r) + idx_o * ELEM;
                        const uint32_t src_oi = (offo < half_m ? src0i : src1i) + idx_o * ELEM;

                        // Issue parallel copy across PACK / MATH / UNPACK
                        shuffle_element(
                            src_er, src_ei,
                            src_or, src_oi,
                            dst_er + dst * ELEM,
                            dst_ei + dst * ELEM,
                            dst_or + dst * ELEM,
                            dst_oi + dst * ELEM);

                        dst++;
                    }
                }

                // Free out0/out1 slots
                cb_pop_front(cb_out0_r, 1); cb_pop_front(cb_out0_i, 1);
                cb_pop_front(cb_out1_r, 1); cb_pop_front(cb_out1_i, 1);

                // Signal that next stage's inputs are ready (compute reads these itself)
                cb_push_back(cb_even_r, tiles_per_stage);
                cb_push_back(cb_even_i, tiles_per_stage);
                cb_push_back(cb_odd_r,  tiles_per_stage);
                cb_push_back(cb_odd_i,  tiles_per_stage);

            }
            // If is_last: out0/out1 stay pushed — writer drains them to DRAM.
        }
    }
}