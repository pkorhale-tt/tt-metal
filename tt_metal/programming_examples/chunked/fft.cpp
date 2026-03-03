#include "host_api.hpp"
#include "device.hpp"
#include <sys/time.h>
#include <time.h>

#define PI 3.14159265358979323846264338327950288

using namespace tt;
using namespace tt::tt_metal;

enum FFTDirection {
    FFT_FORWARD=0,
    FFT_BACKWARD=1
};

struct TTExecution {
    Program *program;
    CoreCoord *core;
    KernelHandle *read_kernel, *write_kernel, *compute_kernel;
    std::shared_ptr<tt::tt_metal::Buffer> in_data_r_dram_buffer, in_data_i_dram_buffer, twiddle_dram_buffer, result_data_r_dram_buffer, result_data_i_dram_buffer;
    std::shared_ptr<tt::tt_metal::Buffer> read_in_r_buffer, read_in_i_buffer, twiddle_buffer;
};

void fft(CommandQueue&, TTExecution*, float*, float*, float*, float*, float*, uint32_t, enum FFTDirection);
void compare(float*, float*, float*, float*, int);
void moveorigin(float*, float*, int);
void descale(float*, float*, int);
int checkIfPowerOfTwo(int);
CBHandle createCB(Program&, CoreCoord&, uint32_t, uint32_t, uint32_t);
float* computeTwiddleFactors(int);
static double getElapsedTime(struct timeval);

int main(int argc, char** argv) {
    if (argc != 2) {
      fprintf(stderr, "You must provide the size of the domain as an argument\n");
      return -1;
    }

    int domain_size=atoi(argv[1]);
    if (!checkIfPowerOfTwo(domain_size)) {
      fprintf(stderr, "%d provided as domain size, but this must be a power of two\n", domain_size);
      return -1;
    }

    /* Silicon accelerator setup */
    IDevice* device = CreateDevice(0);

    /* Setup program to execute along with its buffers and kernels to use */
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    uint32_t problem_mem_size = 4 * domain_size;
    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = problem_mem_size,
        .page_size = problem_mem_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    std::shared_ptr<tt::tt_metal::Buffer> in_data_r_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<tt::tt_metal::Buffer> in_data_i_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<tt::tt_metal::Buffer> result_data_r_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<tt::tt_metal::Buffer> result_data_i_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<tt::tt_metal::Buffer> twiddle_dram_buffer = CreateBuffer(dram_config);

    /* Use L1 circular buffers to set input and output buffers that the compute engine will use */
    uint32_t cb_tile_size=1024 * 2;
    uint32_t cb_total_size=problem_mem_size > cb_tile_size ? problem_mem_size: cb_tile_size;
    uint32_t num_chunks=4; //cb_total_size / cb_tile_size;
    // Data 0 into compute
    createCB(program, core, CBIndex::c_0, num_chunks, cb_tile_size);
    createCB(program, core, CBIndex::c_1, num_chunks, cb_tile_size);
    // Data 1 into compute
    createCB(program, core, CBIndex::c_2, num_chunks, cb_tile_size);
    createCB(program, core, CBIndex::c_3, num_chunks, cb_tile_size);
    // Twiddle factors
    createCB(program, core, CBIndex::c_4, num_chunks, cb_tile_size);
    createCB(program, core, CBIndex::c_5, num_chunks, cb_tile_size);
    // Data 0 out from compute
    createCB(program, core, CBIndex::c_6, num_chunks, cb_tile_size);
    createCB(program, core, CBIndex::c_7, num_chunks, cb_tile_size);
    // Data 1 out from compute
    createCB(program, core, CBIndex::c_8, num_chunks, cb_tile_size);
    createCB(program, core, CBIndex::c_9, num_chunks, cb_tile_size);
    // Data 0 rearranged from writer
    // This must be two as when we pipeline the writer is writing the current iteration to
    // the next CB and reader is reading from the current CB. The same applies to the 
    // data 1 CB (next one) too
    createCB(program, core, CBIndex::c_10, 2, cb_total_size);
    // Data 1 rearranged from writer
    createCB(program, core, CBIndex::c_11, 2, cb_total_size);
    // Intermediate results
    // The CB size is all one below here as these are used internally by the compute core
    // as intermediate results
    createCB(program, core, CBIndex::c_12, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_13, 1, cb_tile_size);
    createCB(program, core, CBIndex::c_14, 1, cb_tile_size);
    // f0
    createCB(program, core, CBIndex::c_15, 1, cb_tile_size);
    // f1
    createCB(program, core, CBIndex::c_16, 1, cb_tile_size);

    tt::tt_metal::InterleavedBufferConfig l1_read_buffer_config{
        .device= device,
        .size = problem_mem_size,
        .page_size = problem_mem_size,
        .buffer_type = tt::tt_metal::BufferType::L1};

    std::shared_ptr<tt::tt_metal::Buffer> read_in_r_buffer = CreateBuffer(l1_read_buffer_config);
    std::shared_ptr<tt::tt_metal::Buffer> read_in_i_buffer = CreateBuffer(l1_read_buffer_config);
    // Whilst we have n/2 twiddle factors, pack real and imaginary in so the data size is the same
    std::shared_ptr<tt::tt_metal::Buffer> twiddle_buffer = CreateBuffer(l1_read_buffer_config);

    /* Specify data movement kernels for reading/writing data to/from DRAM */
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "kernels/dataflow/reader.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "kernels/dataflow/writer.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    /* Set the parameters that the compute kernel will use */
    std::vector<uint32_t> compute_kernel_args = {};

    /* Use the add_tiles operation in the compute kernel */
    KernelHandle compute_kernel_id = CreateKernel(
        program,
        "kernels/compute/compute.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
        });

    /* Create source data and write to DRAM */
    float * golden_r=(float*) malloc(sizeof(float) * domain_size);
    float * golden_i=(float*) malloc(sizeof(float) * domain_size);
    for (int i=0;i<domain_size;i++) {
	    golden_r[i]=0.0f;
	    golden_i[i]=0.0f;
    }
    golden_r[domain_size/2]=(float) domain_size;
    golden_i[domain_size/2]=(float) domain_size*2;

    float * twiddle_factors=computeTwiddleFactors(domain_size);

    float * data_r=(float*) malloc(sizeof(float) * domain_size);
    float * data_i=(float*) malloc(sizeof(float) * domain_size);

    memcpy(data_r, golden_r, sizeof(float) * domain_size);
    memcpy(data_i, golden_i, sizeof(float) * domain_size);

    /* Configure program and runtime kernel arguments, then execute */

    TTExecution exec={
        .program=&program,
        .core=&core,
        .read_kernel=&reader_kernel_id,
        .write_kernel=&writer_kernel_id,
        .compute_kernel=&compute_kernel_id,
        .in_data_r_dram_buffer=in_data_r_dram_buffer,
        .in_data_i_dram_buffer=in_data_i_dram_buffer,
        .twiddle_dram_buffer=twiddle_dram_buffer,
        .result_data_r_dram_buffer=result_data_r_dram_buffer,
        .result_data_i_dram_buffer=result_data_i_dram_buffer,
        .read_in_r_buffer=read_in_r_buffer,
        .read_in_i_buffer=read_in_i_buffer,
        .twiddle_buffer=twiddle_buffer
    };

    //for (int i=0;i<1;i++) {
    //    printf("Iteration %d:\n", i);
        // We reuse the data arrays for the results
        fft(cq, &exec, data_r, data_i, twiddle_factors, data_r, data_i, domain_size, FFT_FORWARD);
        fft(cq, &exec, data_r, data_i, twiddle_factors, data_r, data_i, domain_size, FFT_BACKWARD);
    //}

    moveorigin(data_r, data_i, domain_size);
    descale(data_r, data_i, domain_size);

    //compare(data_r, data_i, golden_r, golden_i, domain_size);

    CloseDevice(device);

    free(data_r);
    free(data_i);
    free(twiddle_factors);
    free(golden_r);
    free(golden_i);
}

void fft(CommandQueue& cq, TTExecution * device_descriptor, float * input_r, float * input_i, float * twiddle_factors, float * result_r, float * result_i, uint32_t domain_size, enum FFTDirection direction) {        
    // Since all interleaved buffers have size == page_size, they are entirely contained in the first DRAM bank
    uint32_t in_data_r_dram_bank_id = 0;
    uint32_t in_data_i_dram_bank_id = 0;
    uint32_t result_data_r_dram_bank_id = 0;
    uint32_t result_data_i_dram_bank_id = 0;
    uint32_t twiddle_dram_bank_id = 0;

    const std::vector<uint32_t> read_kernel_runtime_args = {
            device_descriptor->in_data_r_dram_buffer->address(),
            device_descriptor->in_data_i_dram_buffer->address(),
            device_descriptor->twiddle_dram_buffer->address(),
            in_data_r_dram_bank_id,
            in_data_i_dram_bank_id,
            twiddle_dram_bank_id,
            device_descriptor->read_in_r_buffer->address(),
            device_descriptor->read_in_i_buffer->address(),
            device_descriptor->twiddle_buffer->address(),
            domain_size};

    const std::vector<uint32_t> write_kernel_runtime_args = {
            device_descriptor->result_data_r_dram_buffer->address(),
            device_descriptor->result_data_i_dram_buffer->address(),
            result_data_r_dram_bank_id,
            result_data_i_dram_bank_id,
            domain_size};

    SetRuntimeArgs(
        *(device_descriptor->program),
        *(device_descriptor->read_kernel),
        *(device_descriptor->core),
        read_kernel_runtime_args);

    SetRuntimeArgs(
        *(device_descriptor->program),
        *(device_descriptor->compute_kernel),
        *(device_descriptor->core),
        {direction, domain_size});

    SetRuntimeArgs(
        *(device_descriptor->program),
        *(device_descriptor->write_kernel),
        *(device_descriptor->core),
        write_kernel_runtime_args);

    struct timeval start_time;

    gettimeofday(&start_time, NULL);
    EnqueueWriteBuffer(cq, device_descriptor->in_data_r_dram_buffer, input_r, false);
    EnqueueWriteBuffer(cq, device_descriptor->in_data_i_dram_buffer, input_i, false);
    EnqueueWriteBuffer(cq, device_descriptor->twiddle_dram_buffer, twiddle_factors, false);
    Finish(cq);
    double xfer_on_time=getElapsedTime(start_time);

    gettimeofday(&start_time, NULL);
    EnqueueProgram(cq, *(device_descriptor->program), false);
    Finish(cq);
    double exec_time=getElapsedTime(start_time);

    gettimeofday(&start_time, NULL);
    EnqueueReadBuffer(cq, device_descriptor->result_data_r_dram_buffer, result_r, false);
    EnqueueReadBuffer(cq, device_descriptor->result_data_i_dram_buffer, result_i, false);
    Finish(cq);
    double xfer_off_time=getElapsedTime(start_time);

    double total_time=xfer_on_time+exec_time+xfer_off_time;
    printf("%s FFT of size %d: total time %.6f sec. %.6f sec transfer on, %.6f sec execution, %.6f sec transfer off\n", 
            direction == 0 ? "Forwards" : "Backwards", domain_size, total_time, xfer_on_time, exec_time, xfer_off_time);
}

void compare(float * a_data_r, float * a_data_i, float * b_data_r, float * b_data_i, int domain_size) {
  int matching, missmatching;
  matching=missmatching=0;
  for (int i=0; i<domain_size;i++) {
    float a_r=a_data_r[i];
    float a_i=a_data_i[i];
    float b_r=b_data_r[i];
    float b_i=b_data_i[i];

    if (a_r != b_r || a_i != b_i) {
      printf("Miss match index %d: (%.2f, %.2f) vs (%.2f, %.2f)\n", i, a_r, a_i, b_r, b_i);
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

static double getElapsedTime(struct timeval start_time) {
  struct timeval curr_time;
  gettimeofday(&curr_time, NULL);
  long int elapsedtime = (curr_time.tv_sec * 1000000 + curr_time.tv_usec) - (start_time.tv_sec * 1000000 + start_time.tv_usec);
  return elapsedtime / 1000000.0;
}

