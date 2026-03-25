// fft1d_wormhole_fixed.cpp
// Wormhole 1-D radix-2 DIT FFT
// Row-parallel multicore version for Wormhole.
//
// Key fixes:
//   1. Uses as many cores as there are rows (up to the user's request and device limit).
//   2. Distributes rows per core with remainder handling.
//   3. Builds STAGE-SPECIFIC twiddle tiles in DRAM instead of reusing tile 0 for every stage.
//   4. Keeps the same compute/writer protocol as your original code so it is easy to drop in.
//
// Important:
//   This is still a ROW-PARALLEL design.
//   If num_rows == 1, only 1 core can do useful work.
//   To split a single long row across many cores, you need a different decomposition.

#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <algorithm>

#include "tt_metal/api/tt-metalium/host_api.hpp"
#include "tt_metal/api/tt-metalium/constants.hpp"
#include "tt_metal/api/tt-metalium/distributed.hpp"
#include "tt_metal/api/tt-metalium/base_types.hpp"
#include "tt_metal/api/tt-metalium/mesh_workload.hpp"

using namespace tt;
using namespace tt::tt_metal;

constexpr float    PI         = 3.14159265358979323846f;
constexpr uint32_t TILE_H     = tt::constants::TILE_HEIGHT;
constexpr uint32_t TILE_W     = tt::constants::TILE_WIDTH;
constexpr uint32_t TILE_SIZE  = TILE_H * TILE_W;
constexpr uint32_t TILE_BYTES = TILE_SIZE * sizeof(float);

static inline uint32_t f2u(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(uint32_t));
    return u;
}

static inline float u2f(uint32_t u) {
    float f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

static uint32_t bit_reverse(uint32_t x, uint32_t bits) {
    uint32_t r = 0;
    for (uint32_t i = 0; i < bits; ++i) {
        r = (r << 1) | (x & 1u);
        x >>= 1;
    }
    return r;
}

static uint32_t log2_exact(uint32_t n) {
    uint32_t l = 0;
    while ((1u << l) < n) {
        ++l;
    }
    return l;
}

static void cpu_fft(std::vector<float>& re, std::vector<float>& im, bool inv) {
    const uint32_t N = static_cast<uint32_t>(re.size());
    const uint32_t logN = log2_exact(N);

    for (uint32_t i = 0; i < N; ++i) {
        const uint32_t j = bit_reverse(i, logN);
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }

    for (uint32_t s = 0; s < logN; ++s) {
        const uint32_t m = 1u << (s + 1);
        const float angle_base = (inv ? 2.0f : -2.0f) * PI / static_cast<float>(m);
        for (uint32_t k = 0; k < N; k += m) {
            for (uint32_t j = 0; j < (m >> 1); ++j) {
                const float wr = std::cos(angle_base * static_cast<float>(j));
                const float wi = std::sin(angle_base * static_cast<float>(j));
                const uint32_t e = k + j;
                const uint32_t o = k + j + (m >> 1);
                const float tr = wr * re[o] - wi * im[o];
                const float ti = wr * im[o] + wi * re[o];
                re[o] = re[e] - tr;
                im[o] = im[e] - ti;
                re[e] = re[e] + tr;
                im[e] = im[e] + ti;
            }
        }
    }

    if (inv) {
        const float scale = 1.0f / static_cast<float>(N);
        for (auto& v : re) v *= scale;
        for (auto& v : im) v *= scale;
    }
}

static bool read_input(const std::string& path, uint32_t N,
                       std::vector<float>& out_r,
                       std::vector<float>& out_i) {
    std::ifstream f(path);
    if (!f) {
        std::cerr << "Cannot open: " << path << "\n";
        return false;
    }

    std::vector<float> vals;
    std::string tok;
    while (f >> tok) {
        if (!tok.empty() && tok.back() == ',') tok.pop_back();
        if (tok.empty()) continue;
        try {
            vals.push_back(std::stof(tok));
        } catch (...) {
            std::cerr << "Bad token: '" << tok << "'\n";
            return false;
        }
    }

    if (vals.empty()) {
        std::cerr << "Empty input file\n";
        return false;
    }

    out_r.assign(N, 0.0f);
    out_i.assign(N, 0.0f);

    if (vals.size() >= 2 * N) {
        std::cout << " File mode: interleaved complex (" << vals.size()
                  << " values -> " << N << " complex)\n";
        for (uint32_t i = 0; i < N; ++i) {
            out_r[i] = vals[2 * i];
            out_i[i] = vals[2 * i + 1];
        }
    } else {
        std::cout << " File mode: real-only (" << vals.size()
                  << " values -> " << N << " points)\n";
        for (uint32_t i = 0; i < N && i < vals.size(); ++i) {
            out_r[i] = vals[i];
        }
    }

    return true;
}

struct StagedInput {
    std::vector<uint32_t> er;
    std::vector<uint32_t> ei;
    std::vector<uint32_t> or_;
    std::vector<uint32_t> oi;
};

static StagedInput prepare_stage0(const std::vector<float>& sr,
                                  const std::vector<float>& si,
                                  uint32_t row_offset,
                                  uint32_t N,
                                  uint32_t log2N,
                                  uint32_t tiles_per_row) {
    const uint32_t half = N >> 1;
    std::vector<float> er(half), ei(half), or_(half), oi(half);

    for (uint32_t i = 0; i < half; ++i) {
        const uint32_t e_idx = bit_reverse(2 * i, log2N);
        const uint32_t o_idx = bit_reverse(2 * i + 1, log2N);
        er[i]  = sr[row_offset + e_idx];
        ei[i]  = si[row_offset + e_idx];
        or_[i] = sr[row_offset + o_idx];
        oi[i]  = si[row_offset + o_idx];
    }

    const uint32_t n_elems = tiles_per_row * TILE_SIZE;
    StagedInput out;
    out.er.assign(n_elems, f2u(0.0f));
    out.ei.assign(n_elems, f2u(0.0f));
    out.or_.assign(n_elems, f2u(0.0f));
    out.oi.assign(n_elems, f2u(0.0f));

    for (uint32_t i = 0; i < half; ++i) {
        out.er[i]  = f2u(er[i]);
        out.ei[i]  = f2u(ei[i]);
        out.or_[i] = f2u(or_[i]);
        out.oi[i]  = f2u(oi[i]);
    }

    return out;
}

static void build_stage_twiddles(uint32_t N,
                                 bool inv,
                                 uint32_t num_stages,
                                 uint32_t tiles_per_row,
                                 std::vector<uint32_t>& tw_r,
                                 std::vector<uint32_t>& tw_i) {
    const uint32_t half_N = N >> 1;
    const uint32_t total_tw_tiles = num_stages * tiles_per_row;
    const float sign = inv ? 1.0f : -1.0f;

    tw_r.assign(total_tw_tiles * TILE_SIZE, f2u(0.0f));
    tw_i.assign(total_tw_tiles * TILE_SIZE, f2u(0.0f));

    for (uint32_t stage = 0; stage < num_stages; ++stage) {
        const uint32_t m = 1u << (stage + 1);
        const uint32_t half_m = m >> 1;
        const uint32_t stride = N / m;

        for (uint32_t p = 0; p < half_N; ++p) {
            const uint32_t j = p % half_m;
            const uint32_t k = j * stride;
            const float angle = sign * 2.0f * PI * static_cast<float>(k) / static_cast<float>(N);
            const uint32_t tile = p / TILE_SIZE;
            const uint32_t lane = p % TILE_SIZE;
            const uint32_t base = (stage * tiles_per_row + tile) * TILE_SIZE + lane;
            tw_r[base] = f2u(std::cos(angle));
            tw_i[base] = f2u(std::sin(angle));
        }
    }
}

static CBHandle make_cb(Program& prog, CoreCoord cc, uint32_t id, uint32_t depth) {
    CircularBufferConfig cfg = CircularBufferConfig(depth * TILE_BYTES,
                                                    {{id, tt::DataFormat::Float32}})
                                   .set_page_size(id, TILE_BYTES);
    return CreateCircularBuffer(prog, cc, cfg);
}

static uint32_t detect_row0_cores(IDevice* dev) {
    const CoreCoord grid = dev->compute_with_storage_grid_size();
    std::cout << " Device grid: " << grid.x << " x " << grid.y << " Tensix\n";

    uint32_t avail = 0;
    for (uint32_t x = 0; x < grid.x; ++x) {
        try {
            (void)dev->worker_core_from_logical_core({x, 0});
            ++avail;
        } catch (...) {
            break;
        }
    }
    return avail;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <0|1> [N] [rows] [cores] [input.txt]\n"
                  << "  0=forward  1=inverse\n"
                  << "  N must be a power of 2 (default 64)\n";
        return 1;
    }

    const bool inv = (std::atoi(argv[1]) == 1);
    uint32_t N = 64;
    uint32_t num_rows = 0;
    uint32_t user_cores = 0;
    std::string in_file;

    if (argc >= 3) N = static_cast<uint32_t>(std::stoul(argv[2]));
    if (argc >= 4) num_rows = static_cast<uint32_t>(std::stoul(argv[3]));
    if (argc >= 5) user_cores = static_cast<uint32_t>(std::stoul(argv[4]));
    if (argc >= 6) in_file = argv[5];

    if (N < 2 || (N & (N - 1u))) {
        std::cerr << "N must be a power of 2\n";
        return 1;
    }
    if (argc >= 4 && num_rows == 0) {
        std::cerr << "num_rows must be >= 1\n";
        return 1;
    }
    if (argc >= 5 && user_cores == 0) {
        std::cerr << "num_cores must be >= 1\n";
        return 1;
    }

    auto mesh = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = mesh->mesh_command_queue();
    IDevice* dev = mesh->get_devices().at(0);

    const uint32_t avail_cores = detect_row0_cores(dev);
    const uint32_t requested_cores = (user_cores > 0) ? user_cores : avail_cores;
    const uint32_t core_cap = std::min(avail_cores, requested_cores);

    if (num_rows == 0) {
        num_rows = std::max(1u, core_cap * 4u);
    }

    const uint32_t num_cores = std::max(1u, std::min(core_cap, num_rows));
    if (requested_cores > num_rows) {
        std::cout << " Requested " << requested_cores << " cores but only " << num_rows
                  << " row(s) exist in this row-parallel design, so only " << num_cores
                  << " core(s) can be active.\n";
    }

    const uint32_t log2N = log2_exact(N);
    const uint32_t half_N = N >> 1;
    const uint32_t tiles_per_row = (half_N + TILE_SIZE - 1) / TILE_SIZE;
    const uint32_t total_tiles = num_rows * tiles_per_row;
    const uint32_t total_elems = total_tiles * TILE_SIZE;
    const uint32_t total_N = N * num_rows;

    std::vector<uint32_t> rows_on_core(num_cores, num_rows / num_cores);
    for (uint32_t c = 0; c < (num_rows % num_cores); ++c) {
        rows_on_core[c] += 1;
    }

    std::vector<uint32_t> tile_offset_on_core(num_cores, 0);
    uint32_t row_prefix = 0;
    for (uint32_t c = 0; c < num_cores; ++c) {
        tile_offset_on_core[c] = row_prefix * tiles_per_row;
        row_prefix += rows_on_core[c];
    }

    std::cout << "══════════════════════════════════════\n"
              << " Wormhole 1-D FFT (fixed multicore)\n"
              << "══════════════════════════════════════\n"
              << " N           = " << N << "\n"
              << " num_rows    = " << num_rows << "\n"
              << " num_cores   = " << num_cores << "\n"
              << " log2(N)     = " << log2N << "\n"
              << " half_N      = " << half_N << "\n"
              << " tiles/row   = " << tiles_per_row << "\n"
              << " direction   = " << (inv ? "inverse" : "forward") << "\n"
              << "══════════════════════════════════════\n";

    for (uint32_t c = 0; c < num_cores; ++c) {
        std::cout << " core " << c
                  << " -> rows=" << rows_on_core[c]
                  << ", tile_offset=" << tile_offset_on_core[c] << "\n";
    }

    std::vector<float> in_r(total_N, 0.0f), in_i(total_N, 0.0f);
    if (!in_file.empty()) {
        std::cout << " Input: " << in_file << "\n";
        std::vector<float> row_r, row_i;
        if (!read_input(in_file, N, row_r, row_i)) {
            mesh->close();
            return 1;
        }
        for (uint32_t row = 0; row < num_rows; ++row) {
            for (uint32_t i = 0; i < N; ++i) {
                in_r[row * N + i] = row_r[i];
                in_i[row * N + i] = row_i[i];
            }
        }
    } else {
        for (uint32_t row = 0; row < num_rows; ++row) {
            for (uint32_t i = 0; i < N; ++i) {
                in_r[row * N + i] =
                    std::sin(2.0f * PI * 4.0f * static_cast<float>(i) / static_cast<float>(N)) +
                    0.5f * std::sin(2.0f * PI * 8.0f * static_cast<float>(i) / static_cast<float>(N));
            }
        }
    }

    std::vector<float> ref_r(in_r), ref_i(in_i);
    for (uint32_t row = 0; row < num_rows; ++row) {
        std::vector<float> rr(ref_r.begin() + row * N, ref_r.begin() + (row + 1) * N);
        std::vector<float> ri(ref_i.begin() + row * N, ref_i.begin() + (row + 1) * N);
        cpu_fft(rr, ri, inv);
        for (uint32_t i = 0; i < N; ++i) {
            ref_r[row * N + i] = rr[i];
            ref_i[row * N + i] = ri[i];
        }
    }

    std::vector<uint32_t> all_er(total_elems, f2u(0.0f));
    std::vector<uint32_t> all_ei(total_elems, f2u(0.0f));
    std::vector<uint32_t> all_or(total_elems, f2u(0.0f));
    std::vector<uint32_t> all_oi(total_elems, f2u(0.0f));

    for (uint32_t row = 0; row < num_rows; ++row) {
        StagedInput si = prepare_stage0(in_r, in_i, row * N, N, log2N, tiles_per_row);
        const uint32_t base = row * tiles_per_row * TILE_SIZE;
        for (uint32_t i = 0; i < si.er.size(); ++i) {
            all_er[base + i] = si.er[i];
            all_ei[base + i] = si.ei[i];
            all_or[base + i] = si.or_[i];
            all_oi[base + i] = si.oi[i];
        }
    }

    std::vector<uint32_t> stage_tw_r;
    std::vector<uint32_t> stage_tw_i;
    build_stage_twiddles(N, inv, log2N, tiles_per_row, stage_tw_r, stage_tw_i);

    using namespace tt::tt_metal::distributed;
    DeviceLocalBufferConfig dram_tile_cfg{
        .page_size = TILE_BYTES,
        .buffer_type = BufferType::DRAM,
    };

    const uint32_t buf_bytes = total_elems * sizeof(uint32_t);
    auto mk = [&](uint32_t bytes) {
        ReplicatedBufferConfig rc{.size = bytes};
        return MeshBuffer::create(rc, dram_tile_cfg, mesh.get());
    };

    auto b_er  = mk(buf_bytes);
    auto b_ei  = mk(buf_bytes);
    auto b_or  = mk(buf_bytes);
    auto b_oi  = mk(buf_bytes);
    auto b_oer = mk(buf_bytes);
    auto b_oei = mk(buf_bytes);
    auto b_oor = mk(buf_bytes);
    auto b_ooi = mk(buf_bytes);

    const uint32_t tw_bytes = static_cast<uint32_t>(stage_tw_r.size() * sizeof(uint32_t));
    ReplicatedBufferConfig rc_tw{.size = tw_bytes};
    DeviceLocalBufferConfig dram_tw_cfg{
        .page_size = TILE_BYTES,
        .buffer_type = BufferType::DRAM,
    };
    auto b_tw_r = MeshBuffer::create(rc_tw, dram_tw_cfg, mesh.get());
    auto b_tw_i = MeshBuffer::create(rc_tw, dram_tw_cfg, mesh.get());

    Program prog = CreateProgram();
    CoreRange core_range({0, 0}, {num_cores - 1, 0});

    constexpr const char* KPATH =
        "tt_metal/programming_examples/fft_float32_multicore_optimised/fft_multi_core/kernels/";

    KernelHandle reader_k = CreateKernel(
        prog,
        std::string(KPATH) + "dataflow/reader.cpp",
        core_range,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
        });

    KernelHandle writer_k = CreateKernel(
        prog,
        std::string(KPATH) + "dataflow/writer.cpp",
        core_range,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
        });

    KernelHandle compute_k = CreateKernel(
        prog,
        std::string(KPATH) + "compute/compute.cpp",
        core_range,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .math_approx_mode = false,
        });

    for (uint32_t c = 0; c < num_cores; ++c) {
        const CoreCoord cc = {c, 0};
        const uint32_t rows_this_core = rows_on_core[c];
        const uint32_t tile_offset = tile_offset_on_core[c];

        make_cb(prog, cc,  0, tiles_per_row);
        make_cb(prog, cc,  1, tiles_per_row);
        make_cb(prog, cc,  2, tiles_per_row);
        make_cb(prog, cc,  3, tiles_per_row);
        make_cb(prog, cc,  4, tiles_per_row);
        make_cb(prog, cc,  5, tiles_per_row);
        make_cb(prog, cc,  6, tiles_per_row);
        make_cb(prog, cc,  7, tiles_per_row);
        make_cb(prog, cc,  8, tiles_per_row);
        make_cb(prog, cc,  9, tiles_per_row);
        make_cb(prog, cc, 10, 1);
        make_cb(prog, cc, 11, 1);
        make_cb(prog, cc, 12, 1);
        make_cb(prog, cc, 13, 1);
        make_cb(prog, cc, 16, 1);
        make_cb(prog, cc, 17, 1);
        make_cb(prog, cc, 18, 1);
        make_cb(prog, cc, 19, 1);

        SetRuntimeArgs(prog, reader_k, cc, std::vector<uint32_t>{
            b_er->address(),
            b_ei->address(),
            b_or->address(),
            b_oi->address(),
            b_tw_r->address(),
            b_tw_i->address(),
            tiles_per_row,
            tile_offset,
            log2N,
            half_N,
            rows_this_core,
        });

        SetRuntimeArgs(prog, compute_k, cc, std::vector<uint32_t>{
            log2N,
            tiles_per_row,
            rows_this_core,
        });

        SetRuntimeArgs(prog, writer_k, cc, std::vector<uint32_t>{
            b_oer->address(),
            b_oei->address(),
            b_oor->address(),
            b_ooi->address(),
            tiles_per_row,
            log2N,
            half_N,
            tile_offset,
            rows_this_core,
        });
    }

    MeshWorkload wl;
    MeshCoordinateRange rng = MeshCoordinateRange(mesh->shape());
    wl.add_program(rng, std::move(prog));

    std::cout << "Writing inputs to DRAM...\n";
    EnqueueWriteMeshBuffer(cq, b_er, all_er, false);
    EnqueueWriteMeshBuffer(cq, b_ei, all_ei, false);
    EnqueueWriteMeshBuffer(cq, b_or, all_or, false);
    EnqueueWriteMeshBuffer(cq, b_oi, all_oi, false);
    EnqueueWriteMeshBuffer(cq, b_tw_r, stage_tw_r, false);
    EnqueueWriteMeshBuffer(cq, b_tw_i, stage_tw_i, false);
    Finish(cq);

    std::cout << "Launching FFT on " << num_cores << " core(s), "
              << num_rows << " row(s) of " << N << " points...\n";
    EnqueueMeshWorkload(cq, wl, true);
    std::cout << "Kernel complete.\n";

    std::vector<uint32_t> raw_oer(total_elems), raw_oei(total_elems);
    std::vector<uint32_t> raw_oor(total_elems), raw_ooi(total_elems);
    EnqueueReadMeshBuffer(cq, raw_oer, b_oer, true);
    EnqueueReadMeshBuffer(cq, raw_oei, b_oei, true);
    EnqueueReadMeshBuffer(cq, raw_oor, b_oor, true);
    EnqueueReadMeshBuffer(cq, raw_ooi, b_ooi, true);

    std::vector<float> res_r(total_N), res_i(total_N);
    for (uint32_t row = 0; row < num_rows; ++row) {
        const uint32_t tile_base = row * tiles_per_row * TILE_SIZE;
        for (uint32_t i = 0; i < half_N; ++i) {
            res_r[row * N + i] = u2f(raw_oer[tile_base + i]);
            res_i[row * N + i] = u2f(raw_oei[tile_base + i]);
            res_r[row * N + i + half_N] = u2f(raw_oor[tile_base + i]);
            res_i[row * N + i + half_N] = u2f(raw_ooi[tile_base + i]);
        }
    }

    if (inv) {
        const float scale = 1.0f / static_cast<float>(N);
        for (auto& v : res_r) v *= scale;
        for (auto& v : res_i) v *= scale;
    }

    float max_err_r = 0.0f;
    float max_err_i = 0.0f;
    float mean_err = 0.0f;
    uint32_t worst_row = 0;
    float worst_val = 0.0f;

    for (uint32_t row = 0; row < num_rows; ++row) {
        float row_worst = 0.0f;
        for (uint32_t i = 0; i < N; ++i) {
            const float er = std::fabs(res_r[row * N + i] - ref_r[row * N + i]);
            const float ei = std::fabs(res_i[row * N + i] - ref_i[row * N + i]);
            max_err_r = std::max(max_err_r, er);
            max_err_i = std::max(max_err_i, ei);
            mean_err += er + ei;
            row_worst = std::max(row_worst, er + ei);
        }
        if (row_worst > worst_val) {
            worst_val = row_worst;
            worst_row = row;
        }
    }
    mean_err /= (2.0f * static_cast<float>(total_N));

    const float threshold = std::max(0.5f, 0.005f * static_cast<float>(N));
    const bool passed = (max_err_r < threshold) && (max_err_i < threshold);

    std::cout << "\n══════════════════════════════════════\n"
              << " VALIDATION (" << num_rows << " row(s))\n"
              << "══════════════════════════════════════\n"
              << " Max error (real) : " << max_err_r << "\n"
              << " Max error (imag) : " << max_err_i << "\n"
              << " Mean error       : " << mean_err << "\n"
              << " Worst row        : " << worst_row << "\n"
              << " Threshold        : " << threshold << "\n"
              << " Result           : " << (passed ? "PASS" : "FAIL") << "\n";

    std::cout << "\n══════════════════════════════════════\n"
              << " FIRST 16 BINS (row 0) okok\n"
              << "══════════════════════════════════════\n"
              << std::fixed << std::setprecision(4);
    for (uint32_t i = 0; i < 16 && i < N; ++i) {
        std::cout << " X[" << std::setw(3) << i << "] = "
                  << std::setw(10) << res_r[i]
                  << (res_i[i] >= 0.0f ? " + " : " - ")
                  << std::setw(10) << std::fabs(res_i[i]) << "j"
                  << "  ref: "
                  << std::setw(10) << ref_r[i]
                  << (ref_i[i] >= 0.0f ? " + " : " - ")
                  << std::setw(10) << std::fabs(ref_i[i]) << "j\n";
    }

    mesh->close();
    return passed ? 0 : 1;
}
