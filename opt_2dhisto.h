#ifndef OPT_KERNEL
#define OPT_KERNEL

void opt_2dhisto(uint32_t *dev_input, uint *dev_kernel_bins/*, uint32_t *shuffle_input*/);

/* Include below the function headers of any other functions that you implement */
uint32_t* allocate_input_on_device();
uint32_t* allocate_bins_on_device();
void copy_input_and_initialize_bins(uint32_t **input, uint32_t *device_input, uint32_t *device_bins);
void transfer_bins(uint8_t *output_bins, uint32_t *device_bins);
void free_device_memory(uint32_t *device_input, uint32_t *device_bins);
void free_shuffle_memory(uint32_t *shuffle_input);
#endif