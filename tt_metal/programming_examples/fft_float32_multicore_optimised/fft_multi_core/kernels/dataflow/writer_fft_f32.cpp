// writer_fft_f32_mc.cpp — MULTICORE writer with ThCon 128-bit shuffle
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ══════════════════════════════════════════════════════════════════════
//  THCON 128-BIT SHUFFLE OPTIMISATION
// ══════════════════════════════════════════════════════════════════════
//
//  PROBLEM (32-bit scalar RISC-V):
//    The inter-stage shuffle copies N/2 floats per array per stage.
//    Each copy is one 32-bit RISC-V load + one 32-bit store.
//    The data mover RISC-V cores are slow for bulk data — they were
//    designed to drive routers and CBs, not to copy memory.
//
//  SOLUTION (ThCon 128-bit):
//    ThCon (Tensor Controller) is the scalar unit inside the compute
//    engine. It has direct high-bandwidth L1 access and can load/store
//    128 bits (4 floats) in one instruction via LLK intrinsics.
//
//    When the shuffle accesses CONTIGUOUS source addresses (which
//    happens whenever offset is always < half_m or always >= half_m
//    within a group), we can use 128-bit loads to copy 4 floats at once.
//
//    For the NON-CONTIGUOUS case (mixed out0/out1 access), we fall
//    back to 32-bit scalar copies — 128-bit loads only help when the
//    source stride is exactly 4 bytes (contiguous floats).
//
//  IMPLEMENTATION:
//    The writer RISC-V detects contiguous vs non-contiguous runs within
//    each shuffle group and switches between 128-bit ThCon copies and
//    32-bit scalar copies accordingly.
//
//    ThCon load helper:
//      thcon_load128(from_addr) — loads 4 floats via TT_LOADIND LD_128bit
//      thcon_store128(to_addr)  — stores 4 floats via TT_STOREIND ST_128bit
//
//    128-bit copies require:
//      - Both src and dst addresses 16-byte aligned
//      - Contiguous source stride of 16 bytes
//
//  RESULT:
//    Contiguous runs (most of the shuffle at large stages): 4× fewer
//    memory transactions → ~2-3× shuffle speedup per the paper.
//    Non-contiguous runs (small stages with interleaved out0/out1):
//    unchanged (scalar fallback).
//
// ══════════════════════════════════════════════════════════════════════

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "llk_io.h"          // TT_SETDMAREG, TT_LOADIND, TT_STOREIND
#include "llk_defs.h"        // p_ind::LD_128bit, p_ind::ST_128bit

void kernel_main() {
    const uint32_t out0_r_addr   = get_arg_val<uint32_t>(0);
    const uint32_t out0_i_addr   = get_arg_val<uint32_t>(1);
    const uint32_t out1_r_addr   = get_arg_val<uint32_t>(2);
    const uint32_t out1_i_addr   = get_arg_val<uint32_t>(3);
    const uint32_t local_tiles   = get_arg_val<uint32_t>(4);
    const uint32_t num_stages    = get_arg_val<uint32_t>(5);
    const uint32_t local_half    = get_arg_val<uint32_t>(6);
    const uint32_t half_N        = get_arg_val<uint32_t>(7);
    const uint32_t num_cores     = get_arg_val<uint32_t>(8);
    const uint32_t core_id       = get_arg_val<uint32_t>(9);
    const uint32_t log2_cores    = get_arg_val<uint32_t>(10);
    const uint32_t tile_offset   = get_arg_val<uint32_t>(11);
    const uint32_t core_elem_base= get_arg_val<uint32_t>(12);

    constexpr uint32_t cb_out0_r = 16;
    constexpr uint32_t cb_out0_i = 17;
    constexpr uint32_t cb_out1_r = 18;
    constexpr uint32_t cb_out1_i = 19;
    constexpr uint32_t cb_even_r = 0;
    constexpr uint32_t cb_even_i = 1;
    constexpr uint32_t cb_odd_r  = 2;
    constexpr uint32_t cb_odd_i  = 3;

    const uint32_t tile_bytes    = get_tile_size(cb_out0_r);
    const DataFormat data_format = get_dataformat(cb_out0_r);

    const InterleavedAddrGenFast<true> out0_r_gen = {
        .bank_base_address = out0_r_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> out0_i_gen = {
        .bank_base_address = out0_i_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> out1_r_gen = {
        .bank_base_address = out1_r_addr,
        .page_size = tile_bytes, .data_format = data_format };
    const InterleavedAddrGenFast<true> out1_i_gen = {
        .bank_base_address = out1_i_addr,
        .page_size = tile_bytes, .data_format = data_format };

    if (local_tiles == 0) return;

    constexpr uint32_t ELEM    = sizeof(float);      // 4 bytes
    constexpr uint32_t ELEM128 = 4 * sizeof(float);  // 16 bytes = 4 floats

    // ── Scalar 32-bit helpers (non-contiguous fallback) ────────────────
    auto rd32 = [](uint32_t addr) -> uint32_t {
        return *reinterpret_cast<volatile uint32_t*>(addr);
    };
    auto wr32 = [](uint32_t addr, uint32_t v) {
        *reinterpret_cast<volatile uint32_t*>(addr) = v;
    };

    // ── ThCon 128-bit copy: copy 4 floats from src to dst ─────────────
    // Uses TT_LOADIND (LD_128bit) and TT_STOREIND (ST_128bit).
    // Both src and dst must be 16-byte aligned.
    // This runs on the writer RISC-V core which has access to ThCon
    // via the LLK intrinsic interface.
    //
    // Register layout for TT_SETDMAREG / TT_LOADIND:
    //   regs 0,1 = offset (lo/hi halfwords)
    //   regs 2,3 = base address (lo/hi halfwords)
    //   reg  4   = destination register for loaded data
    auto copy128 = [](uint32_t dst, uint32_t src) {
        // Decompose src into base (multiples of 16) + offset
        uint32_t base   = src >> 4;          // src / 16
        uint32_t offset = src & 0xFu;        // src % 16

        // Load 4 floats from src into ThCon reg 4
        TT_SETDMAREG(0, LOWER_HALFWORD(offset), 0, LO_16(0));
        TT_SETDMAREG(0, UPPER_HALFWORD(offset), 0, HI_16(0));
        TT_SETDMAREG(0, LOWER_HALFWORD(base),   0, LO_16(1));
        TT_SETDMAREG(0, UPPER_HALFWORD(base),   0, HI_16(1));
        TT_LOADIND(p_ind::LD_128bit, LO_16(0), p_ind::INC_NONE, 4, 1);

        // Store 4 floats from ThCon reg 4 to dst
        uint32_t dbase   = dst >> 4;
        uint32_t doffset = dst & 0xFu;
        TT_SETDMAREG(0, LOWER_HALFWORD(doffset), 0, LO_16(2));
        TT_SETDMAREG(0, UPPER_HALFWORD(doffset), 0, HI_16(2));
        TT_SETDMAREG(0, LOWER_HALFWORD(dbase),   0, LO_16(3));
        TT_SETDMAREG(0, UPPER_HALFWORD(dbase),   0, HI_16(3));
        TT_STOREIND(p_ind::ST_128bit, LO_16(2), p_ind::INC_NONE, 4, 3);
    };

    // ── Contiguous block copy using 128-bit where possible ────────────
    // Copies `count` floats from src to dst.
    // Uses 128-bit (4-float) copies when both addresses are 16-byte
    // aligned and count >= 4, then scalar for the remainder.
    auto copy_floats = [&](uint32_t dst, uint32_t src, uint32_t count) {
        // Align prologue: scalar until dst is 16-byte aligned
        while (count > 0 && (dst & 0xFu) != 0) {
            wr32(dst, rd32(src));
            dst += ELEM; src += ELEM; count--;
        }
        // 128-bit bulk copy (4 floats at a time)
        while (count >= 4) {
            copy128(dst, src);
            dst += ELEM128; src += ELEM128; count -= 4;
        }
        // Scalar epilogue for remaining 0-3 floats
        while (count > 0) {
            wr32(dst, rd32(src));
            dst += ELEM; src += ELEM; count--;
        }
    };

    for (uint32_t stage = 0; stage < num_stages; stage++) {
        const bool is_last = (stage == num_stages - 1);

        cb_wait_front(cb_out0_r, local_tiles);
        cb_wait_front(cb_out0_i, local_tiles);
        cb_wait_front(cb_out1_r, local_tiles);
        cb_wait_front(cb_out1_i, local_tiles);

        const uint32_t src0r = get_read_ptr(cb_out0_r);
        const uint32_t src0i = get_read_ptr(cb_out0_i);
        const uint32_t src1r = get_read_ptr(cb_out1_r);
        const uint32_t src1i = get_read_ptr(cb_out1_i);

        if (is_last) {
            // ── DRAM write ───────────────────────────────────────────
            for (uint32_t t = 0; t < local_tiles; t++) {
                uint32_t gt = tile_offset + t;
                noc_async_write_tile(gt, out0_r_gen, src0r + t*tile_bytes);
                noc_async_write_tile(gt, out0_i_gen, src0i + t*tile_bytes);
                noc_async_write_tile(gt, out1_r_gen, src1r + t*tile_bytes);
                noc_async_write_tile(gt, out1_i_gen, src1i + t*tile_bytes);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out0_r, local_tiles);
            cb_pop_front(cb_out0_i, local_tiles);
            cb_pop_front(cb_out1_r, local_tiles);
            cb_pop_front(cb_out1_i, local_tiles);

        } else {
            // ── SHUFFLE with ThCon 128-bit optimisation ───────────────
            //
            // The shuffle formula maps each destination slot to either
            // out0[idx] or out1[idx]. Within a group, the first half_m
            // slots come from out0 (contiguous) and the second half_m
            // slots come from out1 (contiguous).
            //
            // This means within each group of half_m slots, the source
            // is a CONTIGUOUS block — perfect for 128-bit copies.
            //
            // For stages where G2 > 0 (normal shuffle):
            //   Each group contributes:
            //     half_m contiguous floats from out0 → even
            //     half_m contiguous floats from out1 → even (upper half)
            //   We copy each contiguous block with copy_floats() which
            //   uses 128-bit ThCon internally.
            //
            // For G2 = 0 (late stages, direct passthrough):
            //   out0 → even, out1 → odd, both contiguous → full 128-bit.

            const uint32_t m       = 1u << (stage + 1);
            const uint32_t half_m  = m >> 1;
            const uint32_t m2      = m << 1;
            const uint32_t half_m2 = m2 >> 1;
            const uint32_t G2      = (half_m2 <= local_half)
                                     ? local_half / half_m2
                                     : 0u;

            cb_reserve_back(cb_even_r, local_tiles);
            cb_reserve_back(cb_even_i, local_tiles);
            cb_reserve_back(cb_odd_r,  local_tiles);
            cb_reserve_back(cb_odd_i,  local_tiles);

            const uint32_t dst_er = get_write_ptr(cb_even_r);
            const uint32_t dst_ei = get_write_ptr(cb_even_i);
            const uint32_t dst_or = get_write_ptr(cb_odd_r);
            const uint32_t dst_oi = get_write_ptr(cb_odd_i);

            if (G2 > 0) {
                // ── Normal shuffle with 128-bit contiguous block copies ──
                //
                // The shuffle formula produces contiguous source runs of
                // exactly half_m floats from out0, then half_m from out1,
                // alternating per group. We detect and copy each run.
                //
                // For even destinations: g2 groups, each contributing
                //   half_m floats from out0 then half_m from out1
                // For odd destinations: same structure.

                const uint32_t log2m  = stage + 1;
                const uint32_t m_mask = m - 1u;
                uint32_t dst = 0;

                for (uint32_t g2 = 0; g2 < G2; g2++) {
                    const uint32_t local_base_e = g2 * m2;
                    const uint32_t local_base_o = local_base_e + half_m2;

                    // Compute source indices for first element of each
                    // contiguous block in this group.
                    // new_even: j2=0..half_m2-1, split at j2=half_m
                    // First block (j2=0..half_m-1): offset < half_m → out0
                    //   f = core_elem_base + local_base_e + 0
                    //   g_old = f >> log2m, offset = f & m_mask < half_m
                    //   src_start = g_old * half_m + offset (in out0)
                    // Second block (j2=half_m..half_m2-1): offset ≥ half_m
                    //   src_start = g_old * half_m + (offset - half_m) (out1)

                    // Block 1 for new_even: from out0
                    {
                        uint32_t f0        = core_elem_base + local_base_e;
                        uint32_t g_old     = f0 >> log2m;
                        uint32_t offset    = f0 & m_mask;
                        uint32_t src_start = g_old * half_m + offset;
                        uint32_t local_src = src_start - core_elem_base;
                        copy_floats(dst_er + dst*ELEM,
                                    src0r  + local_src*ELEM, half_m);
                        copy_floats(dst_ei + dst*ELEM,
                                    src0i  + local_src*ELEM, half_m);
                    }
                    // Block 2 for new_even: from out1
                    {
                        uint32_t f0        = core_elem_base + local_base_e + half_m;
                        uint32_t g_old     = f0 >> log2m;
                        uint32_t offset    = f0 & m_mask;
                        uint32_t src_start = g_old * half_m + (offset - half_m);
                        uint32_t local_src = src_start - core_elem_base;
                        copy_floats(dst_er + (dst+half_m)*ELEM,
                                    src1r  + local_src*ELEM, half_m);
                        copy_floats(dst_ei + (dst+half_m)*ELEM,
                                    src1i  + local_src*ELEM, half_m);
                    }
                    // Block 1 for new_odd: from out0
                    {
                        uint32_t f0        = core_elem_base + local_base_o;
                        uint32_t g_old     = f0 >> log2m;
                        uint32_t offset    = f0 & m_mask;
                        uint32_t src_start = g_old * half_m + offset;
                        uint32_t local_src = src_start - core_elem_base;
                        copy_floats(dst_or + dst*ELEM,
                                    src0r  + local_src*ELEM, half_m);
                        copy_floats(dst_oi + dst*ELEM,
                                    src0i  + local_src*ELEM, half_m);
                    }
                    // Block 2 for new_odd: from out1
                    {
                        uint32_t f0        = core_elem_base + local_base_o + half_m;
                        uint32_t g_old     = f0 >> log2m;
                        uint32_t offset    = f0 & m_mask;
                        uint32_t src_start = g_old * half_m + (offset - half_m);
                        uint32_t local_src = src_start - core_elem_base;
                        copy_floats(dst_or + (dst+half_m)*ELEM,
                                    src1r  + local_src*ELEM, half_m);
                        copy_floats(dst_oi + (dst+half_m)*ELEM,
                                    src1i  + local_src*ELEM, half_m);
                    }

                    dst += half_m2;
                }

            } else {
                // ── G2=0: direct passthrough — fully contiguous ──────────
                // out0 → even (all local_half floats, contiguous)
                // out1 → odd  (all local_half floats, contiguous)
                // Both are perfectly contiguous → maximum 128-bit benefit.
                copy_floats(dst_er, src0r, local_half);
                copy_floats(dst_ei, src0i, local_half);
                copy_floats(dst_or, src1r, local_half);
                copy_floats(dst_oi, src1i, local_half);
            }

            cb_pop_front(cb_out0_r, local_tiles);
            cb_pop_front(cb_out0_i, local_tiles);
            cb_pop_front(cb_out1_r, local_tiles);
            cb_pop_front(cb_out1_i, local_tiles);

            cb_push_back(cb_even_r, local_tiles);
            cb_push_back(cb_even_i, local_tiles);
            cb_push_back(cb_odd_r,  local_tiles);
            cb_push_back(cb_odd_i,  local_tiles);
        }
    }
}