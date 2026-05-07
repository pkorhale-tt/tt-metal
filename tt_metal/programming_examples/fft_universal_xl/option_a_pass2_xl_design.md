# Option A — `pass2_xl` multi-tile twiddle kernel (design + skeleton)

## Goal

Replace the current `fft_stockham::pass2_twiddle_transpose` constraint
`N2 <= 1024` with a streaming kernel that supports **N2 up to 1M** —
i.e. multiple tiles per row.

Once this lands, the XL dispatcher can drop its host-side outer
twiddle (currently the only host arithmetic in the K=3 path) and
become **truly zero-host-arithmetic** for any pow2 N up to 1G.

## Existing kernel (recap)

`tt_metal/programming_examples/fft_stockham/kernel/pass2_compute.cpp`
processes ONE tile per row:

```
for (n1 = c*tiles_per_core; n1 < (c+1)*tiles_per_core; ++n1):
    cb_wait_front(A_R, 1)
    cb_wait_front(A_I, 1)
    cb_wait_front(T_R, 1)         # twiddle row n1, length N2 (1 tile)
    cb_wait_front(T_I, 1)
    # compute B = A * T (complex multiply, SFPU)
    pack_tile(B_R) ; pack_tile(B_I)
    cb_pop_front(A_R) ; ...
```

The DRAM buffers are sized `N1 * kTileSizeFp32` (4 KB per row).

## New `pass2_xl` design

Each row is now `tiles_per_row = ceil(N2 / 1024)` tiles wide.
Iterate the inner tile dimension explicitly:

```
for n1 in [start, end):
    for tile_j in [0, tiles_per_row):
        cb_wait_front(A_R, 1)
        cb_wait_front(A_I, 1)
        cb_wait_front(T_R, 1)     # twiddle tile (n1, tile_j)
        cb_wait_front(T_I, 1)
        # B = A * T per tile (same SFPU sequence as today's pass2)
        pack_tile(B_R) ; pack_tile(B_I)
        cb_pop_front(...)
```

The compute math is **identical** per tile — just looped twice
(over rows AND tiles within each row) instead of once.

The twiddle table grows from `N1 * 1024 floats` to `N1 * N2 floats`
(real and imag), uploaded once at plan build and re-used across calls.

## Files to create

```
fft_stockham/kernel/pass2_xl_compute.cpp     # NEW (skeleton in this PR)
fft_stockham/kernel/pass2_xl_reader.cpp      # NEW (skeleton in this PR)
fft_stockham/kernel/pass2_xl_writer.cpp      # NEW (skeleton in this PR)
fft_stockham/kernel/pass2_xl_common.h        # NEW (CB IDs + constants)
```

And in the host file:

```cpp
// fft_stockham_host.cpp additions:
struct Pass2XLPlan { ... };                  // mirrors Pass2Plan but per-tile
inline std::shared_ptr<Pass2XLPlan> make_pass2_xl_plan(md, N1, N2);
inline std::vector<Complex> pass2_xl_twiddle_transpose(md, A, plan);
```

The dispatcher in `fft_universal_xl/fft_universal_xl_host.cpp` swaps
the host twiddle loop for a single call to `pass2_xl_twiddle_transpose`
once this is wired up.

## Plan struct (host side)

```cpp
struct Pass2XLPlan {
    uint32_t N1, N2, tiles_per_row;
    uint32_t num_cores, rows_per_core;
    uint32_t grid_cols, grid_rows;

    std::shared_ptr<MeshDevice>  md;
    std::shared_ptr<MeshBuffer>  in_r_buf,  in_i_buf;     // sized N1 * N2 floats
    std::shared_ptr<MeshBuffer>  out_r_buf, out_i_buf;
    std::shared_ptr<MeshBuffer>  tw_r_buf,  tw_i_buf;     // sized N1 * N2 floats
    MeshWorkload                 workload;

    std::vector<float> in_r_host, in_i_host, out_r_host, out_i_host;

    bool initialized = false;
};
```

Compared to the existing `Pass2Plan`:

* DRAM buffer size `N1 * 1024 * sizeof(float)` -> `N1 * N2 * sizeof(float)`.
  At N1=1024, N2=1024 (1M total): 4 MB per buffer x 4 = 16 MB. Fits.
* Twiddle DRAM size: same growth, 16 MB. Still fits.
* Each core handles `rows_per_core = N1 / num_cores` rows; each row is
  `tiles_per_row` tiles streamed through the existing two-tile CB
  pipeline.

## Compute kernel ABI

```c++
// Compile-time args (KERNEL_COMPILE_TIME_ARGS):
//   0: tiles_per_row     // = N2 / 1024
//   1: rows_per_core     // = N1 / num_cores
//
// Runtime args (per core):
//   0: row_start         // first n1 this core handles
//   1: row_end           // exclusive

// CB IDs (match pass2_xl_common.h):
//   A_R, A_I    : input tiles  (2-tile pipelined)
//   T_R, T_I    : twiddle tiles (2-tile pipelined)
//   B_R, B_I    : output tiles  (2-tile pipelined)
//   TMP_R, TMP_I: scratch       (1 tile)
```

## Skeleton: `pass2_xl_compute.cpp`

```c++
// SPDX: 2026 Tenstorrent
//
// pass2_xl_compute — multi-tile twiddle multiply (B = A * T) per row.
// Identical inner math to pass2_compute, looped over tiles_per_row.

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "pass2_xl_common.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t TPR  = get_compile_time_arg_val(0);  // tiles_per_row
    constexpr uint32_t RPC  = get_compile_time_arg_val(1);  // rows_per_core

    // Init binary SFPU (complex multiply)
    binary_op_init_common(CB_A_R, CB_T_R, CB_B_R);

    for (uint32_t r = 0; r < RPC; ++r) {
        for (uint32_t t = 0; t < TPR; ++t) {
            cb_wait_front(CB_A_R, 1);
            cb_wait_front(CB_A_I, 1);
            cb_wait_front(CB_T_R, 1);
            cb_wait_front(CB_T_I, 1);

            cb_reserve_back(CB_B_R, 1);
            cb_reserve_back(CB_B_I, 1);

            // ---- TO IMPLEMENT: SFPU complex multiply ----
            // (real, imag) = (a_r*t_r - a_i*t_i, a_r*t_i + a_i*t_r)
            //
            // The existing pass2_compute.cpp already does exactly this
            // for one tile. Lift its inner block here.
            // ---------------------------------------------

            cb_push_back(CB_B_R, 1);
            cb_push_back(CB_B_I, 1);

            cb_pop_front(CB_A_R, 1);
            cb_pop_front(CB_A_I, 1);
            cb_pop_front(CB_T_R, 1);
            cb_pop_front(CB_T_I, 1);
        }
    }
}
}
```

## Skeleton: `pass2_xl_reader.cpp`

```c++
// SPDX: 2026 Tenstorrent
//
// pass2_xl_reader — streams rows of A (N2 elements each, multiple tiles)
// and the matching twiddle row from DRAM, in tile-major order per row.

#include "dataflow_api.h"
#include "pass2_xl_common.h"

void kernel_main() {
    const uint32_t row_start    = get_arg_val<uint32_t>(0);
    const uint32_t row_end      = get_arg_val<uint32_t>(1);
    const uint32_t a_r_addr     = get_arg_val<uint32_t>(2);
    const uint32_t a_i_addr     = get_arg_val<uint32_t>(3);
    const uint32_t t_r_addr     = get_arg_val<uint32_t>(4);
    const uint32_t t_i_addr     = get_arg_val<uint32_t>(5);

    constexpr uint32_t TPR = get_compile_time_arg_val(0);
    constexpr uint32_t TILE_BYTES = 4096;        // fp32 tile

    InterleavedAddrGenFast<true> a_r_gen{
        .bank_base_address = a_r_addr,
        .page_size = TILE_BYTES, .data_format = DataFormat::Float32};
    // ... same for a_i, t_r, t_i

    for (uint32_t r = row_start; r < row_end; ++r) {
        for (uint32_t t = 0; t < TPR; ++t) {
            const uint32_t tile_id = r * TPR + t;

            cb_reserve_back(CB_A_R, 1);
            uint32_t l1_w = get_write_ptr(CB_A_R);
            noc_async_read_tile(tile_id, a_r_gen, l1_w);
            // ... same pattern for A_I, T_R, T_I

            noc_async_read_barrier();
            cb_push_back(CB_A_R, 1);
            cb_push_back(CB_A_I, 1);
            cb_push_back(CB_T_R, 1);
            cb_push_back(CB_T_I, 1);
        }
    }
}
```

## Skeleton: `pass2_xl_writer.cpp`

Mirror of the reader, popping `CB_B_R`/`CB_B_I` and writing tiles back
out via `noc_async_write_tile`.

## Test plan

1. Unit test: build `pass2_xl` against random inputs of N1=4, N2=2048,
   compare each output element to the host-computed `A * twiddle`.
   Tolerance: 1e-5 relative (single fp32 multiply, no accumulation).
2. Integration test: wire into `fft_universal_xl/fft_universal_xl_host.cpp`,
   re-run the existing `fft_universal_xl_test` — should match the
   host-twiddle results bit-for-bit (modulo fp32 reordering).
3. Bench: at N=8M, measure that the on-device twiddle is at least as
   fast as the host loop (it should be ~10x faster).

## Estimated effort

| Piece | Time |
|---|---|
| `pass2_xl_common.h` + plan struct | 0.5 day |
| Reader + writer skeletons → real | 1 day |
| Compute kernel (SFPU complex mul over loop) | 1-2 days |
| Host plan + dispatcher wiring | 1 day |
| Tests + bench | 1 day |
| **Total** | **~1 week** |
