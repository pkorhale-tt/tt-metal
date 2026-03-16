// fft_compute_f32.cpp  — OPTIMAL v3 (compatible)
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Architecture change from V2:
//   V2: writer (RISCV_1) performed the inter-stage L1 shuffle after each butterfly.
//       Compute waited for RISCV_1 to finish before starting the next stage.
//   V3: the shuffle is moved INTO this compute kernel, eliminating the
//       RISCV_1 round-trip latency between every stage.
//       Writer is now trivial — one DRAM write at the very end.
//
// NOTE on PACK/MATH/UNPACK: in this TT-Metal API version these macros
// are identity functions (#define PACK(x) x) and do NOT route code to
// separate sub-engine threads. True 3-way parallel copy is not available
// through the standard high-level API. The shuffle is performed with
// direct RISC-V pointer reads/writes — proven correct in V2.
//
// The key V3 benefit is structural: removing the writer↔compute
// synchronisation round-trip (cb_push_back / cb_wait_front pair)
// that existed between every stage in V2. Compute now self-feeds
// its own even/odd CBs without involving RISCV_1.
//
// CB map:
//   0  cb_even_r   stage input even real
//   1  cb_even_i   stage input even imag
//   2  cb_odd_r    stage input odd  real
//   3  cb_odd_i    stage input odd  imag
//   4  cb_tw_r     expanded twiddle real  (reader fills per stage)
//   5  cb_tw_i     expanded twiddle imag
//  16  cb_out0_r   butterfly sum  real  → kept for last stage → writer
//  17  cb_out0_i   butterfly sum  imag
//  18  cb_out1_r   butterfly diff real
//  19  cb_out1_i   butterfly diff imag
//  20  cb_tmp0     scratch
//  21  cb_tmp1     scratch
//  22  cb_tw_odd_r scratch
//  23  cb_tw_odd_i scratch

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/copy_dest_values.h"

void kernel_main() {
    const uint32_t num_stages      = get_arg_val<uint32_t>(0);
    const uint32_t tiles_per_stage = get_arg_val<uint32_t>(1);
    const uint32_t half_N          = get_arg_val<uint32_t>(2);

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

    constexpr uint32_t ELEM = sizeof(float);

    binary_op_init_common(cb_even_r, cb_odd_r, cb_tmp0);

    // Inline helpers: safe float copy via uint32 (avoids strict-aliasing)
    auto rd = [](uint32_t addr) -> float {
        uint32_t raw = *reinterpret_cast<volatile uint32_t*>(addr);
        float v = 0.0f;
        __builtin_memcpy(&v, &raw, sizeof(float));
        return v;
    };
    auto wr = [](uint32_t addr, float v) {
        uint32_t raw = 0u;
        __builtin_memcpy(&raw, &v, sizeof(float));
        *reinterpret_cast<volatile uint32_t*>(addr) = raw;
    };

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

            // ── Wait for outputs to be available ──────────────────
            cb_wait_front(cb_out0_r, 1); cb_wait_front(cb_out0_i, 1);
            cb_wait_front(cb_out1_r, 1); cb_wait_front(cb_out1_i, 1);

            if (!is_last) {
                // ── L1-to-L1 shuffle: self-feed next stage ────────
                //
                // Source: out0/out1 CBs (in L1, just written by FPU above)
                // Dest:   even/odd CBs  (in L1, read by this kernel next stage)
                //
                // Uses get_read_ptr/get_write_ptr — available in all
                // sub-engine compilation units (unlike cb_get_tile).
                //
                // Shuffle formula (verified N=4..1024):
                //   log2m = stage+1,  m_mask = (1<<log2m)-1
                //   m2 = 1<<(stage+2),  half_m2 = m2>>1,  G2 = half_N/half_m2
                //   For g2 in [0,G2), j2 in [0,half_m2):
                //     fe = g2*m2+j2,       ge = fe>>log2m, offe = fe&m_mask
                //     fo = fe+half_m2,     go = fo>>log2m, offo = fo&m_mask
                //     new_even[dst] = (offe<half_m) ? out0[ge*half_m+offe]
                //                                   : out1[ge*half_m+offe-half_m]
                //     new_odd[dst]  = (offo<half_m) ? out0[go*half_m+offo]
                //                                   : out1[go*half_m+offo-half_m]

                const uint32_t src0r = get_read_ptr(cb_out0_r);
                const uint32_t src0i = get_read_ptr(cb_out0_i);
                const uint32_t src1r = get_read_ptr(cb_out1_r);
                const uint32_t src1i = get_read_ptr(cb_out1_i);

                cb_reserve_back(cb_even_r, tiles_per_stage);
                cb_reserve_back(cb_even_i, tiles_per_stage);
                cb_reserve_back(cb_odd_r,  tiles_per_stage);
                cb_reserve_back(cb_odd_i,  tiles_per_stage);

                const uint32_t dst_er = get_write_ptr(cb_even_r);
                const uint32_t dst_ei = get_write_ptr(cb_even_i);
                const uint32_t dst_or = get_write_ptr(cb_odd_r);
                const uint32_t dst_oi = get_write_ptr(cb_odd_i);

                const uint32_t log2m   = stage + 1u;
                const uint32_t half_m  = 1u << stage;
                const uint32_t m_mask  = (1u << log2m) - 1u;
                const uint32_t m2      = 1u << (stage + 2u);
                const uint32_t half_m2 = m2 >> 1u;
                const uint32_t G2      = half_N / half_m2;

                uint32_t dst = 0;
                for (uint32_t g2 = 0; g2 < G2; g2++) {
                    const uint32_t base_e = g2 * m2;
                    const uint32_t base_o = base_e + half_m2;
                    for (uint32_t j2 = 0; j2 < half_m2; j2++) {

                        // new_even source
                        const uint32_t fe    = base_e + j2;
                        const uint32_t ge    = fe >> log2m;
                        const uint32_t offe  = fe & m_mask;
                        const bool     e_from_out0 = (offe < half_m);
                        const uint32_t idx_e = ge * half_m + (e_from_out0 ? offe : offe - half_m);
                        const uint32_t se_r  = (e_from_out0 ? src0r : src1r) + idx_e * ELEM;
                        const uint32_t se_i  = (e_from_out0 ? src0i : src1i) + idx_e * ELEM;

                        // new_odd source
                        const uint32_t fo    = base_o + j2;
                        const uint32_t go_   = fo >> log2m;
                        const uint32_t offo  = fo & m_mask;
                        const bool     o_from_out0 = (offo < half_m);
                        const uint32_t idx_o = go_ * half_m + (o_from_out0 ? offo : offo - half_m);
                        const uint32_t so_r  = (o_from_out0 ? src0r : src1r) + idx_o * ELEM;
                        const uint32_t so_i  = (o_from_out0 ? src0i : src1i) + idx_o * ELEM;

                        wr(dst_er + dst * ELEM, rd(se_r));
                        wr(dst_ei + dst * ELEM, rd(se_i));
                        wr(dst_or + dst * ELEM, rd(so_r));
                        wr(dst_oi + dst * ELEM, rd(so_i));

                        dst++;
                    }
                }

                cb_pop_front(cb_out0_r, 1); cb_pop_front(cb_out0_i, 1);
                cb_pop_front(cb_out1_r, 1); cb_pop_front(cb_out1_i, 1);

                cb_push_back(cb_even_r, tiles_per_stage);
                cb_push_back(cb_even_i, tiles_per_stage);
                cb_push_back(cb_odd_r,  tiles_per_stage);
                cb_push_back(cb_odd_i,  tiles_per_stage);

            }
            // is_last: out0/out1 remain pushed — writer drains them to DRAM.
        }
    }
}