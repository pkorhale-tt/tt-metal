#pragma once

// ============================================================
//  fft_common.h  –  shared constants & CB index definitions
//  for the multi-core 1D Cooley-Tukey FFT on Wormhole
// ============================================================

// ── Circular Buffer indices ──────────────────────────────────
// Four CBs for split complex representation (Davies paper style)
// Plus scratch CBs for incoming NOC data, twiddles, and output

constexpr uint32_t CB_LHS_R      = 0;   // local LHS real
constexpr uint32_t CB_LHS_I      = 1;   // local LHS imag
constexpr uint32_t CB_RHS_R      = 2;   // local RHS real  (or scratch incoming)
constexpr uint32_t CB_RHS_I      = 3;   // local RHS imag  (or scratch incoming)
constexpr uint32_t CB_TWIDDLE_R  = 4;   // twiddle real
constexpr uint32_t CB_TWIDDLE_I  = 5;   // twiddle imag
constexpr uint32_t CB_OUT_R      = 6;   // output real
constexpr uint32_t CB_OUT_I      = 7;   // output imag
constexpr uint32_t CB_SCRATCH_R  = 8;   // scratch: NOC-received real
constexpr uint32_t CB_SCRATCH_I  = 9;   // scratch: NOC-received imag
constexpr uint32_t CB_SYNC       = 10;  // 1-tile sync signal writer→compute

// ── Compile-time args (set at program creation) ──────────────
constexpr uint32_t CT_LOCAL_N    = 0;   // elements per core
constexpr uint32_t CT_NUM_CORES  = 1;   // total cores in use
constexpr uint32_t CT_NUM_STAGES = 2;   // log2(N) total stages
constexpr uint32_t CT_USE_BF16   = 3;   // 1=bfloat16, 0=fp32

// ── Runtime arg layout for writer kernel ────────────────────
// Index  Meaning
//   0    local_cb_r base addr (L1 physical)
//   1    local_cb_i base addr
//   2    scratch_cb_r base addr
//   3    scratch_cb_i base addr
//   4    twiddle_dram_addr (base)
//   5    twiddle_dram_bank_id
//   6    num_cores
//   7    my_core_linear_id
//   8    first_noc_stage  (= log2(local_N) + 1)
//   9    semaphore_id     (index into semaphore table)
//  10+   peer table: [noc_x, noc_y, peer_scratch_r, peer_scratch_i, peer_sem_addr]
//        5 uint32s × (num_cores - 1) peers

constexpr uint32_t RT_CB_R          = 0;
constexpr uint32_t RT_CB_I          = 1;
constexpr uint32_t RT_SCRATCH_R     = 2;
constexpr uint32_t RT_SCRATCH_I     = 3;
constexpr uint32_t RT_TWIDDLE_DRAM  = 4;
constexpr uint32_t RT_TWIDDLE_BANK  = 5;
constexpr uint32_t RT_NUM_CORES     = 6;
constexpr uint32_t RT_MY_CORE_ID    = 7;
constexpr uint32_t RT_FIRST_NOC_STG = 8;
constexpr uint32_t RT_SEM_ID        = 9;
constexpr uint32_t RT_PEER_BASE     = 10;
constexpr uint32_t RT_PEER_STRIDE   = 5;  // fields per peer entry

// Peer entry offsets (relative to RT_PEER_BASE + peer_idx * RT_PEER_STRIDE)
constexpr uint32_t PEER_NOC_X       = 0;
constexpr uint32_t PEER_NOC_Y       = 1;
constexpr uint32_t PEER_SCRATCH_R   = 2;
constexpr uint32_t PEER_SCRATCH_I   = 3;
constexpr uint32_t PEER_SEM_ADDR    = 4;

// ── Tile geometry ────────────────────────────────────────────
// Wormhole tile = 32×32 elements.  For 1D FFT we use 1×N tiles
// (a single row of N elements packed into a tile's row-major layout).
// For N ≤ 1024 and fp32 this fits comfortably in L1.
constexpr uint32_t TILE_HW     = 32;
constexpr uint32_t TILE_SIZE_FP32  = TILE_HW * TILE_HW * 4;  // bytes
constexpr uint32_t TILE_SIZE_BF16  = TILE_HW * TILE_HW * 2;
