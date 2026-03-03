#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <sys/time.h>
#include <time.h>
#include <vector>
#include <cmath>
#include <memory>

#define PI 3.14159265358979323846264338327950288

using namespace tt;
using namespace tt::tt_metal;

enum FFTDirection {
    FFT_FORWARD = 0,
    FFT_BACKWARD = 1
};

void compare(float*, float*, float*, float*, int);
void moveorigin(float*, float*, int);
void descale(float*, float*, int);
int checkIfPowerOfTwo(int);
CBHandle createCB(Program&, CoreCoord&, uint32_t, uint32_t, uint32_t);
float* computeTwiddleFactors(int);
[[maybe_unused]] static double getElapsedTime(struct timeval);

tt::tt_metal::Program create_fft_program(
    CoreCoord core,
    uint32_t domain_size,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> in_data_r_dram,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> in_data_i_dram,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> twiddle_dram,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> result_r_dram,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> result_i_dram,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> read_in_l1,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> twiddle_l1,
    FFTDirection direction) {

    Program program = CreateProgram();

    uint32_t cb_tile_size = 1024 * 4;
    createCB(program, core, CBIndex::c_0, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_1, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_2, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_3, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_4, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_5, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_16, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_17, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_18, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_19, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_20, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_21, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_23, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_24, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_25, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_6, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_7, 1, cb_tile_size);

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/basic/kernels/dataflow/reader.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/basic/kernels/dataflow/writer.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    KernelHandle compute_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/basic/kernels/compute/compute.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = {},
        });

    SetRuntimeArgs(
        program,
        reader_kernel_id,
        core,
        {   in_data_r_dram->address(),
            in_data_i_dram->address(),
            twiddle_dram->address(),
            0, 0, 0,
            read_in_l1->address(),
            twiddle_l1->address(),
            domain_size});

    SetRuntimeArgs(
        program,
        compute_kernel_id,
        core,
        {(uint32_t)direction, domain_size});

    SetRuntimeArgs(
        program,
        writer_kernel_id,
        core,
        {   result_r_dram->address(),
            result_i_dram->address(),
            0, 0,
            domain_size});

    return program;
}

void fft_mesh(
    tt::tt_metal::distributed::MeshCommandQueue& cq,
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> device,
    tt::tt_metal::Program program,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> in_data_r_dram,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> in_data_i_dram,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> twiddle_dram,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> result_data_r_dram,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> result_data_i_dram,
    std::vector<float>& input_r,
    std::vector<float>& input_i,
    std::vector<float>& twiddles,
    std::vector<float>& result_r,
    std::vector<float>& result_i,
    uint32_t domain_size,
    FFTDirection direction) {

    struct timeval start_time;

    gettimeofday(&start_time, NULL);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, in_data_r_dram, input_r, false);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, in_data_i_dram, input_i, false);
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(cq, twiddle_dram, twiddles, false);
    tt::tt_metal::distributed::Finish(cq);
    double xfer_on_time = getElapsedTime(start_time);

    gettimeofday(&start_time, NULL);
    tt::tt_metal::distributed::MeshWorkload workload;
    workload.add_program(tt::tt_metal::distributed::MeshCoordinateRange(device->shape()), std::move(program));
    tt::tt_metal::distributed::EnqueueMeshWorkload(cq, workload, false);
    tt::tt_metal::distributed::Finish(cq);
    double exec_time = getElapsedTime(start_time);

    gettimeofday(&start_time, NULL);
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(cq, result_r, result_data_r_dram, false);
    tt::tt_metal::distributed::EnqueueReadMeshBuffer(cq, result_i, result_data_i_dram, false);
    tt::tt_metal::distributed::Finish(cq);
    double xfer_off_time = getElapsedTime(start_time);

    double total_time = xfer_on_time + exec_time + xfer_off_time;
    printf("%s FFT of size %d: total time %.4f sec. %.4f sec transfer on, %.4f sec execution, %.4f sec transfer off\n", 
            direction == FFT_FORWARD ? "Forwards" : "Backwards", domain_size, total_time, xfer_on_time, exec_time, xfer_off_time);
}

int main(int argc, char** argv) {
    if (argc != 2) {
      fprintf(stderr, "You must provide the size of the domain as an argument\n");
      return -1;
    }

    int domain_size = atoi(argv[1]);
    if (!checkIfPowerOfTwo(domain_size)) {
      fprintf(stderr, "%d provided as domain size, but this must be a power of two\n", domain_size);
      return -1;
    }

    auto device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
    tt::tt_metal::distributed::MeshCommandQueue& cq = device->mesh_command_queue();
    CoreCoord core = {0, 0};

    uint32_t dram_tile_size = 4 * domain_size;
    
    tt::tt_metal::distributed::DeviceLocalBufferConfig dram_config{
        .page_size = dram_tile_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM
    };
    tt::tt_metal::distributed::ReplicatedBufferConfig buffer_config{
        .size = dram_tile_size
    };

    auto in_data_r_dram_buffer = tt::tt_metal::distributed::MeshBuffer::create(buffer_config, dram_config, device.get());
    auto in_data_i_dram_buffer = tt::tt_metal::distributed::MeshBuffer::create(buffer_config, dram_config, device.get());
    auto result_data_r_dram_buffer = tt::tt_metal::distributed::MeshBuffer::create(buffer_config, dram_config, device.get());
    auto result_data_i_dram_buffer = tt::tt_metal::distributed::MeshBuffer::create(buffer_config, dram_config, device.get());
    auto twiddle_dram_buffer = tt::tt_metal::distributed::MeshBuffer::create(buffer_config, dram_config, device.get());

    uint32_t cb_tile_size = 1024 * 4;
    tt::tt_metal::distributed::DeviceLocalBufferConfig l1_config{
        .page_size = cb_tile_size,
        .buffer_type = tt::tt_metal::BufferType::L1
    };
    tt::tt_metal::distributed::ReplicatedBufferConfig l1_buffer_config{
        .size = cb_tile_size
    };

    auto read_in_buffer = tt::tt_metal::distributed::MeshBuffer::create(l1_buffer_config, l1_config, device.get());
    auto twiddle_buffer = tt::tt_metal::distributed::MeshBuffer::create(l1_buffer_config, l1_config, device.get());

    float * golden_r = (float*) malloc(sizeof(float) * domain_size);
    float * golden_i = (float*) malloc(sizeof(float) * domain_size);
    for (int i=0;i<domain_size;i++) {
	    golden_r[i]=0.0f;
	    golden_i[i]=0.0f;
    }
    golden_r[domain_size/2]=(float) domain_size;
    golden_i[domain_size/2]=(float) domain_size*2;
    
    float * twiddle_factors = computeTwiddleFactors(domain_size);
    std::vector<float> twiddle_vec(twiddle_factors, twiddle_factors + domain_size);

    std::vector<float> data_r_vec(golden_r, golden_r + domain_size);
    std::vector<float> data_i_vec(golden_i, golden_i + domain_size);
    
    std::vector<float> result_r_vec(domain_size, 0.0f);
    std::vector<float> result_i_vec(domain_size, 0.0f);

    tt::tt_metal::Program program_fwd = create_fft_program(
        core, domain_size, in_data_r_dram_buffer, in_data_i_dram_buffer, twiddle_dram_buffer,
        result_data_r_dram_buffer, result_data_i_dram_buffer, read_in_buffer, twiddle_buffer, FFT_FORWARD
    );

    fft_mesh(cq, device, std::move(program_fwd), in_data_r_dram_buffer, in_data_i_dram_buffer, twiddle_dram_buffer,
             result_data_r_dram_buffer, result_data_i_dram_buffer, data_r_vec, data_i_vec, twiddle_vec, 
             result_r_vec, result_i_vec, domain_size, FFT_FORWARD);

    tt::tt_metal::Program program_bck = create_fft_program(
        core, domain_size, in_data_r_dram_buffer, in_data_i_dram_buffer, twiddle_dram_buffer,
        result_data_r_dram_buffer, result_data_i_dram_buffer, read_in_buffer, twiddle_buffer, FFT_BACKWARD
    );

    // Feed forward results as backward input
    data_r_vec = result_r_vec;
    data_i_vec = result_i_vec;

    fft_mesh(cq, device, std::move(program_bck), in_data_r_dram_buffer, in_data_i_dram_buffer, twiddle_dram_buffer,
             result_data_r_dram_buffer, result_data_i_dram_buffer, data_r_vec, data_i_vec, twiddle_vec, 
             result_r_vec, result_i_vec, domain_size, FFT_BACKWARD);

    moveorigin(result_r_vec.data(), result_i_vec.data(), domain_size);
    descale(result_r_vec.data(), result_i_vec.data(), domain_size);
    compare(result_r_vec.data(), result_i_vec.data(), golden_r, golden_i, domain_size);

    device->close();

    free(twiddle_factors);
    free(golden_r);
    free(golden_i);
}

void compare(float * a_data_r, float * a_data_i, float * b_data_r, float * b_data_i, int domain_size) {
  int matching = 0, missmatching = 0;
  for (int i=0; i<domain_size;i++) {
    float a_r=a_data_r[i];
    float a_i=a_data_i[i];
    float b_r=b_data_r[i];
    float b_i=b_data_i[i];

    if (a_r != b_r || a_i != b_i) {
      if(missmatching < 5) printf("Miss match index %d: (%.2f, %.2f) vs (%.2f, %.2f)\n", i, a_r, a_i, b_r, b_i);
      missmatching++;
    } else {
      matching++;
    }
  }
  printf("Checked %d elements: %d match and %d missmatched\n", domain_size, matching, missmatching);
}

void moveorigin(float* data_r, float* data_i, int domain_size) {
  for (int i=0;i<domain_size;i++) {
    data_r[i]=data_r[i] * pow(-1, i);
    data_i[i]=data_i[i] * pow(-1, i);
  }
}

void descale(float* data_r, float* data_i, int domain_size) {
  for (int i=0;i<domain_size;i++) {
    data_r[i]=data_r[i] / domain_size;
    data_i[i]=-(data_i[i] / domain_size);
  }
}

int checkIfPowerOfTwo(int v) {
  return (v != 0) && ((v & (v - 1)) == 0);
}

CBHandle createCB(Program & program, CoreCoord & core, uint32_t cb_index, uint32_t num_tiles, uint32_t tile_size) {
    CircularBufferConfig cb_config = 
        CircularBufferConfig(num_tiles * tile_size, {{cb_index, tt::DataFormat::Float32}})
            .set_page_size(cb_index, tile_size);
    CBHandle cb = tt_metal::CreateCircularBuffer(program, core, cb_config);
    return cb;
}

float* computeTwiddleFactors(int n) {
   int num_twiddle_factors=n/2;
   float * twiddle_factors=(float*) malloc(sizeof(float) * num_twiddle_factors * 2);

   for (int i=0;i<num_twiddle_factors;i++) {
     float base_factor=(2.0 * PI * i)/(float) n;
     twiddle_factors[i*2]=(float) cos((double) base_factor);
     twiddle_factors[(i*2)+1]=(float) -sin((double) base_factor);
   }

   return twiddle_factors;
}

[[maybe_unused]] static double getElapsedTime(struct timeval start_time) {
  struct timeval curr_time;
  gettimeofday(&curr_time, NULL);
  long int elapsedtime = (curr_time.tv_sec * 1000000 + curr_time.tv_usec) - (start_time.tv_sec * 1000000 + start_time.tv_usec);
  return elapsedtime / 1000000.0;
}
