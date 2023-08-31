#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#include <cuda.h>
#include <cutil.h>

#include "util.h"
#include "ref_2dhisto.h"
#include "opt_2dhisto.h"

#define SQRT_2    1.4142135623730950488
#define SPREAD_BOTTOM   (2)
#define SPREAD_TOP      (6)

#define NEXT(init_, spread_)\
    (init_ + (int)((drand48() - 0.5) * (drand48() - 0.5) * 4.0 * SQRT_2 * SQRT_2 * spread_));

#define CLAMP(value_, min_, max_)\
    if (value_ < 0)\
        value_ = (min_);\
    else if (value_ > (max_))\
        value_ = (max_);

// Generate another bin for the histogram.  The bins are created as a random walk ...
static uint32_t next_bin(uint32_t pix)
{
    const uint16_t bottom = pix & ((1<<HISTO_LOG)-1);
    const uint16_t top   = (uint16_t)(pix >> HISTO_LOG);

    int new_bottom = NEXT(bottom, SPREAD_BOTTOM);
    CLAMP(new_bottom, 0, HISTO_WIDTH-1);

    int new_top = NEXT(top, SPREAD_TOP);
    CLAMP(new_top, 0, HISTO_HEIGHT-1);

    const uint32_t result = (new_bottom | (new_top << HISTO_LOG)); 

    return result; 
}

// Return a 2D array of histogram bin-ids.  This function generates
// bin-ids with correlation characteristics similar to some actual images.
// The key point here is that the pixels (and thus the bin-ids) are *NOT*
// randomly distributed ... a given pixel tends to be similar to the
// pixels near it.
static uint32_t **generate_histogram_bins()
{
    uint32_t **input = (uint32_t**)alloc_2d(INPUT_HEIGHT, INPUT_WIDTH, sizeof(uint32_t));

    input[0][0] = HISTO_WIDTH/2 | ((HISTO_HEIGHT/2) << HISTO_LOG);
    for (int i = 1; i < INPUT_WIDTH; ++i)
        input[0][i] =  next_bin(input[0][i - 1]);
    for (int j = 1; j < INPUT_HEIGHT; ++j)
    {
        input[j][0] =  next_bin(input[j - 1][0]);
        for (int i = 1; i < INPUT_WIDTH; ++i)
            input[j][i] =  next_bin(input[j][i - 1]);
    }

    return input;
}

int main(int argc, char* argv[])
{
    /* Case of 0 arguments: Default seed is used */
    if (argc < 2){
	srand48(0);
    }
    /* Case of 1 argument: Seed is specified as first command line argument */ 
    else {
	int seed = atoi(argv[1]);
	srand48(seed);
    }

    uint8_t *gold_bins = (uint8_t*)malloc(HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint8_t));

    // Use kernel_bins for your final result
    uint8_t *kernel_bins = (uint8_t*)malloc(HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint8_t));

    // A 2D array of histogram bin-ids.  One can think of each of these bins-ids as
    // being associated with a pixel in a 2D image.
    uint32_t **input = generate_histogram_bins();

    /* Until my kernel is error-free, just run this reference code once. Change it back to 1000 eventually! */
    TIME_IT("ref_2dhisto",
            2,
            ref_2dhisto(input, INPUT_HEIGHT, INPUT_WIDTH, gold_bins);)

    /* Include your setup code below (temp variables, function calls, etc.) */

    /* SETUP STEP 1: Allocate space on the device for our input and output 
       NOTES: Unlike the original input, device_input will be 1D instead of a 2D multi-level array
              The final output should be a uint8_t*, but we can only use atomicAdd on uint32_t*, so we will convert
              the computed bins back in the teardown code. */
    uint32_t* device_input = allocate_input_on_device(); 
    uint32_t* device_bins = allocate_bins_on_device(); 
    /*allocate shuffle space if necessary
    uint32_t* shuffle_input = allocate_input_on_device(); */

    /* SETUP STEP 2: Copy input elements and initialize all bins to 0 */
    copy_input_and_initialize_bins(input, device_input, device_bins);

    /* End of setup code */

    /* This is the call you will use to time your parallel implementation.
       NOTE: opt_2dhisto is responsible for calling cudaDeviceSynchronize() */
       // CHANGE THE NUMBER OF ITERATIONS BACK TO 1, EVENTUALLY
    
    TIME_IT("opt_2dhisto",
            10,
            opt_2dhisto(device_input, device_bins/*, shuffle_input*/);)

    /* Include your teardown code below (temporary variables, function calls, etc.) */

    /* TEARDOWN STEP 1: Transfer the calculated device bins back to host memory */
    transfer_bins(kernel_bins, device_bins);

    /* TEARDOWN STEP 2: Free device memory */
    free_device_memory(device_input, device_bins);
    /*TEARDOWN STEP 3 IF NECESSARY: Free shuffle device memory*/
    //free_shuffle_memory(shuffle_input);

    

    /* End of teardown code */

    int passed=1;

    for (int i=0; i < HISTO_HEIGHT*HISTO_WIDTH; i++){
        if (gold_bins[i] != kernel_bins[i]){
            passed = 0;
            //break;
        }
        printf("\n golker %u %u", gold_bins[i], kernel_bins[i]);
    }

    (passed) ? printf("\n    Test PASSED\n") : printf("\n    Test FAILED\n");

    free(gold_bins);
    free(kernel_bins);
}