// fft_multicore_8x8.cpp — 2D FFT on an 8×8 = 64-core grid  [OPTIMISED]
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ══════════════════════════════════════════════════════════════════════
//  OPTIMISATIONS / FIXES vs. previous version
// ══════════════════════════════════════════════════════════════════════
//
//  FIX-A  Compute arg 1 (tiles_per_stage) must be tiles_this, not
//         tiles_per_row.
//  ─────────────────────────────────────────────────────────────────────
//  For a core with rows_this > 1, the compute kernel processes
//  rows_this × tiles_per_row tiles per butterfly stage.  Passing only
//  tiles_per_row caused the compute kernel to stall after processing
//  one row's worth of tiles per stage, leaving the remaining rows in
//  the CBs unprocessed → deadlock / data corruption.
//
//  FIX-B  Reader arg 10 (local_half) must be rows_this × half_row.
//  ─────────────────────────────────────────────────────────────────────
//  The reader uses local_half to size the twiddle expansion loop.
//  Passing half_row instead of rows_this × half_row caused the ASSERT
//  tw_tiles_needed <= local_tiles to fire for any multi-row core
//  where half_row < TILE_SIZE (e.g. N_row=1024, rows_this=2).
//
//  FIX-C  Writer args 13 (rows_this) and 14 (tiles_per_row) added.
//  ─────────────────────────────────────────────────────────────────────
//  The writer now loops over rows; it needs both values to compute
//  per-row CB byte offsets correctly.
//
//  OPT-9  CB double-buffering for twiddle and output CBs.
//  ─────────────────────────────────────────────────────────────────────
//  BRISC (reader, twiddle expansion) and TRISC (compute, butterfly) run
//  on separate physical RISC-V cores and can pipeline stages:
//
//    reader: expand twiddles for stage S+1 ─────────────────────────┐
//    compute: butterfly for stage S ────────────────────────────────┤→ overlap
//
//  This requires the twiddle CBs (4, 5) to hold at least 2 × tiles_this
//  tiles so reader can write stage S+1 before compute finishes stage S.
//
//  Similarly, compute can produce stage S output while the writer
//  shuffles stage S-1 output, requiring output CBs (16-19) to hold
//  2 × tiles_this tiles.
//
//  Scratch and tw_odd CBs (20-23) hold only one stage worth — they are
//  strictly intra-stage temporaries so double-buffering does not help.
//
//  Memory budget for N_row=1024, rows_this=2, tiles_per_row=1,
//  tiles_this=2, TILE_BYTES=4096:
//    Input   CBs  (0-3):   1× 2 tiles  =  8 KB each →  32 KB
//    Twiddle CBs  (4-5):   2× 2 tiles  = 16 KB each →  32 KB
//    Output  CBs  (16-19): 2× 2 tiles  = 16 KB each →  64 KB
//    Scratch CBs  (20-23): 1× 2 tiles  =  8 KB each →  32 KB
//    Compact CBs  (10-11): 1 tile       =  4 KB each →   8 KB
//    Total                                           = 168 KB  (< 1.5 MB)
//
// ══════════════════════════════════════════════════════════════════════
//  CORE SELECTION STRATEGY  (unchanged — mirrors matmul_multicore)
// ══════════════════════════════════════════════════════════════════════
//  split_work_to_cores(grid, num_rows) returns two groups:
//    group_1: ceil(num_rows/num_cores) rows each
//    group_2: floor(num_rows/num_cores) rows each
//  Column-major linear index → physical coord: {i/grid_y, i%grid_y}.

#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <fstream>
#include <string>
#include <set>

#include "tt_metal/api/tt-metalium/host_api.hpp"
#include "tt_metal/api/tt-metalium/constants.hpp"
#include "tt_metal/api/tt-metalium/distributed.hpp"
#include "tt_metal/api/tt-metalium/base_types.hpp"
#include "tt_metal/api/tt-metalium/mesh_workload.hpp"
#include "tt_metal/api/tt-metalium/allocator.hpp"
#include "tt_metal/api/tt-metalium/hal.hpp"
#include "tt_metal/programming_examples/matmul/matmul_common/bmm_op.hpp"

using namespace tt;
using namespace tt::tt_metal;

// ── Constants ─────────────────────────────────────────────────────────
constexpr float    PI         = 3.14159265358979323846f;
constexpr uint32_t TILE_H     = tt::constants::TILE_HEIGHT;   // 32
constexpr uint32_t TILE_W     = tt::constants::TILE_WIDTH;    // 32
constexpr uint32_t TILE_SIZE  = TILE_H * TILE_W;              // 1024 elements
constexpr uint32_t TILE_BYTES = TILE_SIZE * sizeof(float);    // 4096 bytes

inline uint32_t f2u(float f)    { uint32_t u; std::memcpy(&u,&f,4); return u; }
inline float    u2f(uint32_t u) { float    f; std::memcpy(&f,&u,4); return f; }

std::vector<uint32_t> pack_tiles(const std::vector<float>& d, uint32_t ntiles) {
    std::vector<uint32_t> o(ntiles * TILE_SIZE, 0u);
    for (uint32_t i = 0; i < d.size() && i < o.size(); i++) o[i] = f2u(d[i]);
    return o;
}

uint32_t bit_reverse(uint32_t x, uint32_t log2n) {
    uint32_t r = 0;
    for (uint32_t i = 0; i < log2n; i++) { r = (r << 1) | (x & 1); x >>= 1; }
    return r;
}

void cpu_fft(std::vector<float>& re, std::vector<float>& im, bool inv) {
    uint32_t N = re.size(), log2N = 0;
    while ((1u << log2N) < N) log2N++;
    for (uint32_t i = 0; i < N; i++) {
        uint32_t j = bit_reverse(i, log2N);
        if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
    }
    for (uint32_t s = 0; s < log2N; s++) {
        uint32_t m = 1u << (s + 1);
        float ab = (inv ? 2.f : -2.f) * PI / m;
        for (uint32_t k = 0; k < N; k += m)
            for (uint32_t j = 0; j < m / 2; j++) {
                float wr = std::cos(ab * j), wi = std::sin(ab * j);
                uint32_t e = k + j, o = k + j + m / 2;
                float tr = wr*re[o] - wi*im[o], ti = wr*im[o] + wi*re[o];
                float er = re[e], ei = im[e];
                re[e] = er+tr; im[e] = ei+ti; re[o] = er-tr; im[o] = ei-ti;
            }
    }
    if (inv) for (uint32_t i = 0; i < N; i++) { re[i] /= N; im[i] /= N; }
}

void prepare_stage0_row(
    const std::vector<float>& sr, const std::vector<float>& si,
    uint32_t row_offset, uint32_t N_row, uint32_t log2_row,
    uint32_t tiles_per_row,
    std::vector<uint32_t>& er, std::vector<uint32_t>& ei,
    std::vector<uint32_t>& or_, std::vector<uint32_t>& oi)
{
    uint32_t half = N_row / 2;
    std::vector<float> _er(half), _ei(half), _or(half), _oi(half);
    for (uint32_t i = 0; i < half; i++) {
        uint32_t e = bit_reverse(2*i,   log2_row);
        uint32_t o = bit_reverse(2*i+1, log2_row);
        _er[i] = sr[row_offset+e]; _ei[i] = si[row_offset+e];
        _or[i] = sr[row_offset+o]; _oi[i] = si[row_offset+o];
    }
    auto per = pack_tiles(_er, tiles_per_row); er.insert(er.end(), per.begin(), per.end());
    auto pei = pack_tiles(_ei, tiles_per_row); ei.insert(ei.end(), pei.begin(), pei.end());
    auto por = pack_tiles(_or, tiles_per_row); or_.insert(or_.end(), por.begin(), por.end());
    auto poi = pack_tiles(_oi, tiles_per_row); oi.insert(oi.end(), poi.begin(), poi.end());
}

std::pair<std::vector<uint32_t>, std::vector<uint32_t>>
precompute_compact_twiddles(uint32_t N_row, uint32_t direction) {
    uint32_t half = N_row / 2;
    float sign = (direction == 1) ? 1.f : -1.f;
    std::vector<uint32_t> tw_r(half, 0u), tw_i(half, 0u);
    for (uint32_t k = 0; k < half; k++) {
        float angle = sign * 2.f * PI * (float)k / (float)N_row;
        tw_r[k] = f2u(std::cos(angle)); tw_i[k] = f2u(std::sin(angle));
    }
    return {tw_r, tw_i};
}

// ── create_cb: typed circular buffer ────────────────────────────────────
CBHandle create_cb(Program& p, CoreCoord c, uint32_t id,
                   uint32_t ntiles, uint32_t bytes_per_tile) {
    CircularBufferConfig cfg =
        CircularBufferConfig(ntiles * bytes_per_tile,
                             {{id, tt::DataFormat::Float32}})
            .set_page_size(id, bytes_per_tile);
    return CreateCircularBuffer(p, c, cfg);
}

bool read_input_file(const std::string& path, uint32_t N_row,
                     std::vector<float>& ir, std::vector<float>& ii) {
    std::ifstream f(path);
    if (!f.is_open()) { std::cerr << "Cannot open: " << path << "\n"; return false; }
    std::vector<float> vals; std::string tok;
    while (f >> tok) {
        if (!tok.empty() && tok.back() == ',') tok.pop_back();
        try { vals.push_back(std::stof(tok)); }
        catch (...) { std::cerr << "Bad token '" << tok << "'\n"; return false; }
    }
    ir.assign(N_row, 0.f); ii.assign(N_row, 0.f);
    if (vals.size() == 2 * N_row) {
        for (uint32_t i = 0; i < N_row; i++) { ir[i] = vals[2*i]; ii[i] = vals[2*i+1]; }
    } else {
        for (uint32_t i = 0; i < N_row && i < vals.size(); i++) ir[i] = vals[i];
    }
    return true;
}

// ══════════════════════════════════════════════════════════════════════
//  CoreWork — output of split_work_to_cores
// ══════════════════════════════════════════════════════════════════════
struct CoreWork {
    uint32_t     num_cores;
    CoreRangeSet all_cores;
    CoreRangeSet group_1;
    CoreRangeSet group_2;
    uint32_t     rows_g1;
    uint32_t     rows_g2;
    uint32_t     grid_x;
    uint32_t     grid_y;
};

CoreWork select_cores(IDevice* device, uint32_t total_work,
                      uint32_t req_x = 8, uint32_t req_y = 8) {
    CoreCoord phys = device->compute_with_storage_grid_size();
    uint32_t gx = std::min((uint32_t)phys.x, req_x);
    uint32_t gy = std::min((uint32_t)phys.y, req_y);
    CoreCoord grid{gx, gy};

    std::cout << "  Physical grid  : " << phys.x << " × " << phys.y << "\n";
    std::cout << "  Requested grid : " << req_x << " × " << req_y << "\n";
    std::cout << "  Effective grid : " << gx << " × " << gy
              << " = " << gx*gy << " cores\n";

    auto [num_cores, all_cores, cg1, cg2, wg1, wg2] =
        split_work_to_cores(grid, total_work);

    std::cout << "  Work units     : " << total_work << " (one unit = one FFT row)\n";
    std::cout << "  Cores used     : " << num_cores << "\n";
    std::cout << "  Group-1 cores  : " << cg1.num_cores() << " × " << wg1 << " rows each\n";
    if (cg2.num_cores() > 0)
        std::cout << "  Group-2 cores  : " << cg2.num_cores()
                  << " × " << wg2 << " rows each\n";

    return { num_cores, all_cores, cg1, cg2, wg1, wg2, gx, gy };
}

// ══════════════════════════════════════════════════════════════════════
//  MAIN
// ══════════════════════════════════════════════════════════════════════
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <direction:0=fwd|1=inv> [N_row] [N_col] [input_file]\n"
                  << "  Defaults: N_row=1024, N_col=64\n";
        return 1;
    }

    uint32_t direction = (uint32_t)std::atoi(argv[1]);
    uint32_t N_row     = 1024;
    uint32_t N_col     = 64;
    std::string in_file;

    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        bool is_num = !a.empty();
        for (char c : a) if (!std::isdigit(c)) { is_num = false; break; }
        if (!is_num) { in_file = a; continue; }
        uint32_t v = (uint32_t)std::stoul(a);
        if ((v & (v-1)) == 0) {
            if (v > 64) N_row = v;
            else        N_col = v;
        }
    }
    if (N_row < 2 || (N_row & (N_row-1))) { std::cerr << "N_row must be power-of-2\n"; return 1; }
    if (N_col < 2 || (N_col & (N_col-1))) { std::cerr << "N_col must be power-of-2\n"; return 1; }

    uint32_t num_rows      = N_col;
    uint32_t log2_row      = 0; while ((1u << log2_row) < N_row) log2_row++;
    uint32_t half_row      = N_row / 2;
    uint32_t tiles_per_row = (half_row + TILE_SIZE - 1) / TILE_SIZE;
    uint32_t total_N       = N_row * num_rows;

    // ── Device init ─────────────────────────────────────────────────────
    auto mesh   = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
    auto& cq    = mesh->mesh_command_queue();
    IDevice* device = mesh->get_devices().at(0);

    // ── Core selection (matmul-style split_work_to_cores) ───────────────
    std::cout << "═══════════════════════════════════════════════════\n";
    std::cout << " 8×8 GRID FFT  —  Core Selection.....\n";
    std::cout << "═══════════════════════════════════════════════════\n";
    CoreWork cw = select_cores(device, num_rows, 8, 8);

    uint32_t compact_bytes = half_row * sizeof(float);
    uint32_t compact_alloc = ((compact_bytes + TILE_BYTES - 1) / TILE_BYTES) * TILE_BYTES;
    uint32_t cmp_ntiles    = compact_alloc / TILE_BYTES;

    std::cout << "\n═══════════════════════════════════════════════════\n";
    std::cout << " 8×8 GRID FFT  —  Configuration\n";
    std::cout << "═══════════════════════════════════════════════════\n";
    std::cout << "  N_row        : " << N_row << "\n";
    std::cout << "  N_col (rows) : " << num_rows << "\n";
    std::cout << "  log2(N_row)  : " << log2_row << "\n";
    std::cout << "  tiles/row    : " << tiles_per_row << "\n";
    std::cout << "  Direction    : " << (direction ? "Inverse" : "Forward") << "\n";
    std::cout << "═══════════════════════════════════════════════════\n";

    // ── Input data ───────────────────────────────────────────────────────
    std::vector<float> ir(total_N, 0.f), ii(total_N, 0.f);
    if (!in_file.empty()) {
        std::vector<float> row_r, row_i;
        if (!read_input_file(in_file, N_row, row_r, row_i)) { mesh->close(); return 1; }
        for (uint32_t r = 0; r < num_rows; r++)
            for (uint32_t i = 0; i < N_row; i++) {
                ir[r*N_row+i] = row_r[i]; ii[r*N_row+i] = row_i[i];
            }
    } else {
        for (uint32_t r = 0; r < num_rows; r++)
            for (uint32_t i = 0; i < N_row; i++)
                ir[r*N_row+i] = std::sin(2.f*PI*4.f*i/N_row)
                               + 0.5f*std::sin(2.f*PI*8.f*i/N_row);
    }

    // CPU reference
    std::vector<float> ref_r(ir), ref_i(ii);
    for (uint32_t r = 0; r < num_rows; r++) {
        std::vector<float> rr(ir.begin()+r*N_row, ir.begin()+(r+1)*N_row);
        std::vector<float> ri(ii.begin()+r*N_row, ii.begin()+(r+1)*N_row);
        cpu_fft(rr, ri, direction == 1);
        for (uint32_t i = 0; i < N_row; i++) { ref_r[r*N_row+i] = rr[i]; ref_i[r*N_row+i] = ri[i]; }
    }

    // Stage-0 bit-reversed packing
    std::vector<uint32_t> all_er, all_ei, all_or, all_oi;
    for (uint32_t r = 0; r < num_rows; r++)
        prepare_stage0_row(ir, ii, r*N_row, N_row, log2_row, tiles_per_row,
                           all_er, all_ei, all_or, all_oi);

    auto [cmp_r_t, cmp_i_t] = precompute_compact_twiddles(N_row, direction);
    {
        uint32_t pad = TILE_SIZE - (half_row % TILE_SIZE);
        if (pad != TILE_SIZE) {
            cmp_r_t.resize(half_row + pad, 0u);
            cmp_i_t.resize(half_row + pad, 0u);
        }
    }

    // ── DRAM buffers ─────────────────────────────────────────────────────
    using namespace tt::tt_metal::distributed;
    DeviceLocalBufferConfig dram_tile{.page_size = TILE_BYTES,
                                      .buffer_type = BufferType::DRAM};
    uint32_t total_tiles = num_rows * tiles_per_row;
    uint32_t total_bytes = total_tiles * TILE_BYTES;

    auto mk = [&](uint32_t bytes) -> std::shared_ptr<MeshBuffer> {
        ReplicatedBufferConfig rc{.size = bytes};
        return MeshBuffer::create(rc, dram_tile, mesh.get());
    };
    auto b_er  = mk(total_bytes); auto b_ei  = mk(total_bytes);
    auto b_or  = mk(total_bytes); auto b_oi  = mk(total_bytes);
    auto b_o0r = mk(total_bytes); auto b_o0i = mk(total_bytes);
    auto b_o1r = mk(total_bytes); auto b_o1i = mk(total_bytes);

    DeviceLocalBufferConfig dram_cmp{.page_size = TILE_BYTES,
                                     .buffer_type = BufferType::DRAM};
    ReplicatedBufferConfig rc_cmp{.size = compact_alloc};
    auto b_cmp_r = MeshBuffer::create(rc_cmp, dram_cmp, mesh.get());
    auto b_cmp_i = MeshBuffer::create(rc_cmp, dram_cmp, mesh.get());

    // ── Program & CBs ────────────────────────────────────────────────────
    Program prog = CreateProgram();

    // OPT-9: double-buffer twiddle (4,5) and output (16-19) CBs.
    //
    // CB sizing summary for a core with rows_this rows:
    //   tiles_this = rows_this × tiles_per_row
    //
    //   Input    (0-3):   1 × tiles_this   (loaded once, no pipeline benefit)
    //   Twiddle  (4-5):   2 × tiles_this   (reader ahead by 1 stage)
    //   Compact (10-11):  cmp_ntiles        (one upload, stays resident)
    //   Output  (16-19):  2 × tiles_this   (compute ahead of writer by 1 stage)
    //   Scratch (20-23):  1 × tiles_this   (strictly intra-stage temp)

    for (uint32_t i = 0; i < cw.num_cores; i++) {
        CoreCoord cc = {i / cw.grid_y, i % cw.grid_y};

        uint32_t rows_this = cw.group_1.contains(cc) ? cw.rows_g1 : cw.rows_g2;
        uint32_t tiles_this = rows_this * tiles_per_row;

        // Input CBs — 1× (no pipelining benefit; loaded once per kernel)
        for (uint32_t id : {0u, 1u, 2u, 3u})
            create_cb(prog, cc, id, tiles_this, TILE_BYTES);

        // Twiddle CBs — 2× (OPT-9: reader/compute pipeline overlap)
        for (uint32_t id : {4u, 5u})
            create_cb(prog, cc, id, 2 * tiles_this, TILE_BYTES);

        // Compact twiddle CBs — same size for all cores (full table once)
        create_cb(prog, cc, 10, cmp_ntiles, TILE_BYTES);
        create_cb(prog, cc, 11, cmp_ntiles, TILE_BYTES);

        // Output CBs — 2× (OPT-9: compute/writer pipeline overlap)
        for (uint32_t id : {16u, 17u, 18u, 19u})
            create_cb(prog, cc, id, 2 * tiles_this, TILE_BYTES);

        // Scratch + tw_odd — 1× (intra-stage temporaries only)
        for (uint32_t id : {20u, 21u, 22u, 23u})
            create_cb(prog, cc, id, tiles_this, TILE_BYTES);
    }

    // ── Kernels ──────────────────────────────────────────────────────────
    KernelHandle reader_k = CreateKernel(prog,
        "tt_metal/programming_examples/fft_float32_multicore/fft_multi_core/"
        "kernels/dataflow/reader_fft_f32.cpp",
        cw.all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                           .noc = NOC::RISCV_0_default});

    KernelHandle writer_k = CreateKernel(prog,
        "tt_metal/programming_examples/fft_float32_multicore/fft_multi_core/"
        "kernels/dataflow/writer_fft_f32.cpp",
        cw.all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                           .noc = NOC::RISCV_1_default});

    KernelHandle compute_k = CreateKernel(prog,
        "tt_metal/programming_examples/fft_float32_multicore/fft_multi_core/"
        "kernels/compute/fft_compute_f32.cpp",
        cw.all_cores,
        ComputeConfig{.math_fidelity    = MathFidelity::HiFi4,
                      .fp32_dest_acc_en = true,
                      .math_approx_mode = false});

    // ── Runtime args — column-major loop (matmul_multicore pattern) ──────
    uint32_t global_tile_off = 0;

    for (uint32_t i = 0; i < cw.num_cores; i++) {
        CoreCoord cc = {i / cw.grid_y, i % cw.grid_y};

        uint32_t rows_this;
        if      (cw.group_1.contains(cc)) rows_this = cw.rows_g1;
        else if (cw.group_2.contains(cc)) rows_this = cw.rows_g2;
        else {
            TT_ASSERT(false, "Core not in any group");
            rows_this = 0;
        }

        uint32_t tiles_this       = rows_this * tiles_per_row;
        // FIX-B: local_half must cover all rows this core owns.
        uint32_t local_half_total = rows_this * half_row;

        // ── Reader args ────────────────────────────────────────────────
        // arg  0  even_r_addr
        // arg  1  even_i_addr
        // arg  2  odd_r_addr
        // arg  3  odd_i_addr
        // arg  4  compact_r_addr
        // arg  5  compact_i_addr
        // arg  6  local_tiles    = tiles_this
        // arg  7  tile_offset
        // arg  8  num_stages
        // arg  9  half_N         = half_row (one row's half)
        // arg 10  local_half     = rows_this × half_row  ← FIX-B
        // arg 11  core_elem_base = 0  (row-local addressing)
        // arg 12  rows_this
        // arg 13  tiles_per_row
        std::vector<uint32_t> rdr = {
            b_er->address(), b_ei->address(),
            b_or->address(), b_oi->address(),
            b_cmp_r->address(), b_cmp_i->address(),
            tiles_this,
            global_tile_off,
            log2_row,
            half_row,
            local_half_total,   // FIX-B
            0u,                 // core_elem_base
            rows_this,
            tiles_per_row
        };

        // ── Compute args ───────────────────────────────────────────────
        // arg 0  num_stages
        // arg 1  tiles_per_stage = tiles_this  ← FIX-A (was tiles_per_row)
        std::vector<uint32_t> cmp = { log2_row, tiles_this };  // FIX-A

        // ── Writer args ────────────────────────────────────────────────
        // arg  0  out0_r_addr
        // arg  1  out0_i_addr
        // arg  2  out1_r_addr
        // arg  3  out1_i_addr
        // arg  4  local_tiles      = tiles_this
        // arg  5  num_stages
        // arg  6  local_half       = half_row  (per-row half, used in shuffle)
        // arg  7  half_N           = half_row
        // arg  8  num_cores        = 1  (self-contained per row)
        // arg  9  core_id          = 0
        // arg 10  log2_cores       = 0
        // arg 11  tile_offset
        // arg 12  core_elem_base   = 0
        // arg 13  rows_this                     ← FIX-C (new)
        // arg 14  tiles_per_row                 ← FIX-C (new)
        std::vector<uint32_t> wtr = {
            b_o0r->address(), b_o0i->address(),
            b_o1r->address(), b_o1i->address(),
            tiles_this,
            log2_row,
            half_row,       // per-row half (shuffle formula uses this)
            half_row,
            1u,             // num_cores = 1 (row-decomp)
            0u,             // core_id
            0u,             // log2_cores
            global_tile_off,
            0u,             // core_elem_base
            rows_this,      // FIX-C
            tiles_per_row   // FIX-C
        };

        SetRuntimeArgs(prog, reader_k,  cc, rdr);
        SetRuntimeArgs(prog, compute_k, cc, cmp);
        SetRuntimeArgs(prog, writer_k,  cc, wtr);

        global_tile_off += tiles_this;
    }

    // ── MeshWorkload ─────────────────────────────────────────────────────
    distributed::MeshWorkload wl;
    distributed::MeshCoordinateRange rng =
        distributed::MeshCoordinateRange(mesh->shape());
    wl.add_program(rng, std::move(prog));

    // ── Upload inputs ────────────────────────────────────────────────────
    std::cout << "Writing inputs to DRAM...\n";
    EnqueueWriteMeshBuffer(cq, b_er,    all_er,   false);
    EnqueueWriteMeshBuffer(cq, b_ei,    all_ei,   false);
    EnqueueWriteMeshBuffer(cq, b_or,    all_or,   false);
    EnqueueWriteMeshBuffer(cq, b_oi,    all_oi,   false);
    EnqueueWriteMeshBuffer(cq, b_cmp_r, cmp_r_t,  false);
    EnqueueWriteMeshBuffer(cq, b_cmp_i, cmp_i_t,  false);
    Finish(cq);

    // ── Execute ──────────────────────────────────────────────────────────
    std::cout << "Launching 8×8 FFT (" << cw.num_cores << " cores, "
              << num_rows << " rows × " << N_row << " points)...\n";
    EnqueueMeshWorkload(cq, wl, true);
    std::cout << "Kernel complete.\n";

    // ── Read results ─────────────────────────────────────────────────────
    std::vector<uint32_t> o0r(total_bytes/4), o0i(total_bytes/4);
    std::vector<uint32_t> o1r(total_bytes/4), o1i(total_bytes/4);
    EnqueueReadMeshBuffer(cq, o0r, b_o0r, true);
    EnqueueReadMeshBuffer(cq, o0i, b_o0i, true);
    EnqueueReadMeshBuffer(cq, o1r, b_o1r, true);
    EnqueueReadMeshBuffer(cq, o1i, b_o1i, true);

    std::vector<float> result_r(total_N), result_i(total_N);
    for (uint32_t r = 0; r < num_rows; r++) {
        uint32_t tb = r * tiles_per_row * TILE_SIZE;
        for (uint32_t i = 0; i < half_row; i++) {
            result_r[r*N_row + i]            = u2f(o0r[tb+i]);
            result_i[r*N_row + i]            = u2f(o0i[tb+i]);
            result_r[r*N_row + i + half_row] = u2f(o1r[tb+i]);
            result_i[r*N_row + i + half_row] = u2f(o1i[tb+i]);
        }
    }
    if (direction == 1)
        for (uint32_t i = 0; i < total_N; i++) { result_r[i] /= N_row; result_i[i] /= N_row; }

    // ── Validate ─────────────────────────────────────────────────────────
    float threshold = std::max(0.5f, 0.005f * (float)N_row);
    bool all_pass   = true;

    std::cout << "\n═══════════════════════════════════════════════════\n";
    std::cout << " VALIDATION (" << num_rows << " rows)\n";
    std::cout << "═══════════════════════════════════════════════════\n";

    for (uint32_t r = 0; r < num_rows; r++) {
        float mer = 0.f, mei = 0.f, me = 0.f;
        for (uint32_t i = 0; i < N_row; i++) {
            float er = std::abs(result_r[r*N_row+i] - ref_r[r*N_row+i]);
            float ei = std::abs(result_i[r*N_row+i] - ref_i[r*N_row+i]);
            mer = std::max(mer, er); mei = std::max(mei, ei); me += er + ei;
        }
        me /= 2 * N_row;
        bool row_ok = (mer < threshold) && (mei < threshold);
        if (!row_ok) {
            std::cout << " Row " << r << ": FAIL  max_r=" << mer
                      << " max_i=" << mei << "\n";
            all_pass = false;
        }
        if (r == 0) {
            std::cout << " Row  0: max_r=" << std::setw(10) << mer
                      << "  max_i=" << std::setw(10) << mei
                      << "  mean=" << std::setw(10) << me
                      << "  " << (row_ok ? "✓" : "✗") << "\n";
        }
    }
    std::cout << " Threshold : " << threshold << "\n";
    std::cout << " Overall   : " << (all_pass ? "✓ ALL PASSED" : "✗ SOME FAILED") << "\n";

    // ── First 16 bins of row 0 ───────────────────────────────────────────
    std::cout << "\n═══════════════════════════════════════════════════\n";
    std::cout << " FIRST 16 BINS — row 0\n";
    std::cout << "═══════════════════════════════════════════════════\n";
    std::cout << std::fixed << std::setprecision(5);
    for (uint32_t i = 0; i < 16 && i < N_row; i++) {
        std::cout << " X[" << std::setw(3) << i << "] = "
                  << std::setw(12) << result_r[i]
                  << (result_i[i] >= 0 ? " + " : " - ")
                  << std::setw(12) << std::abs(result_i[i]) << "j"
                  << "   ref: " << std::setw(12) << ref_r[i]
                  << (ref_i[i] >= 0 ? " + " : " - ")
                  << std::setw(12) << std::abs(ref_i[i]) << "j\n";
    }

    // ── Core utilisation map ─────────────────────────────────────────────
    std::cout << "\n═══════════════════════════════════════════════════\n";
    std::cout << " 8×8 CORE UTILISATION MAP  (rows/core)\n";
    std::cout << " col →  ";
    for (uint32_t x = 0; x < cw.grid_x; x++) std::cout << std::setw(4) << x;
    std::cout << "\n row ↓\n";
    {
        std::vector<std::vector<int32_t>> mp(cw.grid_y,
            std::vector<int32_t>(cw.grid_x, -1));
        for (uint32_t i = 0; i < cw.num_cores; i++) {
            uint32_t cx = i / cw.grid_y, cy = i % cw.grid_y;
            CoreCoord cc{cx, cy};
            mp[cy][cx] = cw.group_1.contains(cc)
                         ? (int32_t)cw.rows_g1 : (int32_t)cw.rows_g2;
        }
        for (uint32_t y = 0; y < cw.grid_y; y++) {
            std::cout << "    " << std::setw(2) << y << "  ";
            for (uint32_t x = 0; x < cw.grid_x; x++) {
                if (mp[y][x] < 0) std::cout << "   .";
                else               std::cout << std::setw(4) << mp[y][x];
            }
            std::cout << "\n";
        }
    }

    mesh->close();
    std::cout << "\n═══════════════════════════════════════════════════\n";
    std::cout << " Done\n";
    std::cout << "═══════════════════════════════════════════════════\n";
    return (all_pass ? 0 : 1);
}