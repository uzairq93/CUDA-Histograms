#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"

#define THREAD_DIM_X 32
#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 32
#define PAD_WIDTH ((INPUT_WIDTH + 128) & 0xFFFFFF80) // See alloc_2d in util.cpp
#define BIN_COUNT HISTO_HEIGHT*HISTO_WIDTH
#define INPUT_COUNT INPUT_HEIGHT*INPUT_WIDTH
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

/* Start of my device functions. All functions to be executed on the device must go here, above opt_2dhisto(), because they won't 
   be included in the corresponding header file. */
__global__ void Baseline_Kernel(uint32_t *input, uint32_t *kernel_bins){
    /* Blocks were 32x32 when evaluating this kernel, so memory accesses were coalesced */
    __shared__ uint32_t sub_hist[BIN_COUNT];
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    /* STEP 1: Initialize your assigned bins inside of shared memory
                There are no bank conflicts here because each warp edits a unique offset
    */
    for (int pos = threadIdx.x; pos < BIN_COUNT; pos += blockDim.x)
        sub_hist[pos] = 0;
    __syncthreads();

    /* STEP 2: Add your element to the block's sub histogram
                There may be bank conflicts when writing to shared memory
                This must be written in a for loop if thread coarsening is desired */
    if (row < INPUT_HEIGHT && col < INPUT_WIDTH)
    {
        uint32_t index = input[col + row * PAD_WIDTH];

        // The Always_AtomicAdd kernel just removes this if statement
        if (kernel_bins[index] < 255)
            atomicAdd(kernel_bins + index, 1);
    }
    __syncthreads();

    /* STEP 3: Merge your results with global memory
                There are no bank conflicts here because each warp reads a continuous 32-integer slot
    */
    for (int pos = threadIdx.x; pos < BIN_COUNT; pos += blockDim.x)
    {
        atomicAdd(kernel_bins + pos, sub_hist[pos]);
        if (kernel_bins[pos] > 255)
            atomicExch(kernel_bins + pos, 255);
    }
}


__global__ void NoSharedMem_Kernel(uint32_t *input, uint32_t *kernel_bins) {
    /* Blocks were 32x32 when evaluating this kernel for runtime, so memory accesses were coalesced */
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < INPUT_HEIGHT && col < INPUT_WIDTH)
    {
        uint32_t index = input[col + row * PAD_WIDTH];

        // The Always_AtomicAdd kernel just removes this if statement
        if (kernel_bins[index] < 255) 
            atomicAdd(kernel_bins + index, 1);
    }
}
__global__ void Shuffle(uint32_t *input, uint32_t *shuffle_input) {
    /* Blocks were 32x32 when evaluating this kernel for runtime, so memory accesses were coalesced */
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    __shared__ uint32_t temp[32][32];
    //first, shuffle the input
    if (row < INPUT_HEIGHT && col < PAD_WIDTH)
    {
        
        temp[threadIdx.x][threadIdx.y] = input[col + row * PAD_WIDTH];
    }
    __syncthreads();
    shuffle_input[col+row*PAD_WIDTH] = temp[threadIdx.y][threadIdx.x];
    __syncthreads();
    /*
    //then increment bins
    if (row < INPUT_HEIGHT && col < INPUT_WIDTH)
    {
        uint32_t index = input[col + row * PAD_WIDTH];

        // The Always_AtomicAdd kernel just removes this if statement
        if (kernel_bins[index] < 255) 
            atomicAdd(kernel_bins + index, 1);
    }*/
}
__global__ void TLPNoSharedMem_Kernel(uint32_t *input, uint32_t *kernel_bins)
{
    // NOTE: This is failing because I'm including padded spaces as bins, but it still can
    //       show off the time to say we tried in in the report. 

    /* Threads are covering 32 elements each, so memory accesses are coalesced */
    int iblock = blockIdx.x + blockIdx.y * gridDim.x;
    int index = threadIdx.x + 2 * iblock * gridDim.x;

    uint32_t element1 = input[index];
    uint32_t element2 = input[index + blockDim.x];
    uint32_t element3 = input[index + 2 * blockDim.x];

    // The Always_AtomicAdd kernel just removes this if statement
    if (kernel_bins[element1] < 255)
        atomicAdd(kernel_bins + element1, 1);
    if (kernel_bins[element2] < 255)
        atomicAdd(kernel_bins + element2, 1);
    if (kernel_bins[element2] < 255)
        atomicAdd(kernel_bins + element3, 1);
}

/* End of my device functions */

/* The following function was included for us, and must keep this name. We can call it with any set of parameters our kernels and 
   optimizations may need. However, all memory allocations and transfers must be done outside of this function. This function 
   should only contain a call to the GPU histogramming kernel. */
//needed to remove shuffling from opt2d
void opt_2dhisto(uint32_t *dev_input, uint32_t *dev_kernel_bins/*, uint32_t *shuffle_input*/)
{
    cudaMemset(dev_kernel_bins, 0, BIN_COUNT * sizeof(uint32_t)); // Reset bins every time so the timing works properly
    
    dim3 gridDims(MAX(PAD_WIDTH / BLOCK_DIM_X, 1), MAX(INPUT_HEIGHT / BLOCK_DIM_Y, 1), 1);
    dim3 blockDims(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
    //Shuffle<<<gridDims, blockDims>>>(dev_input, shuffle_input);

    //cudaDeviceSynchronize();
    Baseline_Kernel<<<gridDims, blockDims>>>(dev_input, dev_kernel_bins);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));
}

/* The functions below were written explicitly by Kyle Williams, and are called in test_harness.cpp for setup/teardown of the 
   GPU histogramming kernel. That file is to be compiled with gcc, so we need all calls to CUDA functions to be contained here 
   instead. */

uint32_t* allocate_input_on_device() {
    uint32_t* device_input;
    cudaMalloc((void **)&device_input, INPUT_HEIGHT * PAD_WIDTH * sizeof(uint32_t));
    return device_input;
  
}

//Need to allocate space for the shuffled input, REMOVE IF NOT SHUFFLING
uint32_t* allocate_bins_on_device() {
    uint32_t* device_bins;
    cudaMalloc((void **)&device_bins, BIN_COUNT * sizeof(uint32_t));
    return device_bins;
}

void copy_input_and_initialize_bins(uint32_t **input, uint32_t *device_input, uint32_t *device_bins) {
    /* STEP 1: Copy the elements from the input into the device input */
    for (int i = 0; i < INPUT_HEIGHT; i++) {
        int offset = i * PAD_WIDTH;
        cudaMemcpy(&device_input[offset], input[i], PAD_WIDTH * sizeof(uint32_t), cudaMemcpyHostToDevice);
    }
    /* STEP 3: Initialize the bins to 0 */
    cudaMemset(device_bins, 0, BIN_COUNT * sizeof(uint32_t));
}

void transfer_bins(uint8_t* output_bins, uint32_t* device_bins) {
    /* STEP 1: Transfer the computed frequencies in device_bins to a copy array in local memory */
    uint32_t kernel_bins_copy[BIN_COUNT];
    cudaMemcpy(&kernel_bins_copy, device_bins, BIN_COUNT * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    /* STEP 2: Transfer those computed frequencies back to the expected uint8_t object. Account for 
               overflow and cap frequencies at 255 */
    for (int i = 0; i < BIN_COUNT; i++) {
        if (kernel_bins_copy[i] > 255) output_bins[i] = 255;
        else output_bins[i] = kernel_bins_copy[i];
    }
}
void free_device_memory(uint32_t *device_input, uint32_t *device_bins) {
    cudaFree(device_input);
    cudaFree(device_bins);
}
void free_shuffle_memory(uint32_t *shuffle_input) {
    cudaFree(shuffle_input);   
}
