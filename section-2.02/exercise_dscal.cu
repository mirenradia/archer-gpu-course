/*
 * Introduction.
 *
 * Implement the simple operation x := ax for a vector x of type double
 * and a constant 'a'.
 *
 * This part introduces the kernel.
 *
 * Part 1. write a kernel of prototype
 *         __global__ void mykernel(double a, double * x)
 *         which performs the relevant operation for one block.
 * Part 2. in the main part of the program, declare and initialise
 *         variables of type dim3 to hold the number of blocks, and
 *         the number of threads per block. Use one block and
 *         THREADS_PER_BLOCK in the first instance.
 * Part 3. Generalise the kernel to treat any number of blocks,
 *         and problem sizes which are not a whole number of blocks.
 *
 * Training material originally developed by James Perry and Alan Gray
 * Copyright EPCC, The University of Edinburgh, 2010-2023
 */

#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"

/* Error checking routine and macro. */

__host__ void myErrorHandler(cudaError_t ifail, const char * file,
                             int line, int fatal);

#define CUDA_ASSERT(call) { myErrorHandler((call), __FILE__, __LINE__, 1); }


/* The number of integer elements in the array */
#define ARRAY_LENGTH 1000

/* Suggested kernel parameters */
#define NUM_BLOCKS  4
#define THREADS_PER_BLOCK 256

/* Device kernel */
__global__ void myKernel(double a, double * x);

/* Main routine */

int main(int argc, char *argv[]) {

  size_t sz = ARRAY_LENGTH*sizeof(double);

  double a = 2.0;          /* constant a */
  double * h_x = NULL;     /* input array (host) */
  double * h_out = NULL;   /* output array (host) */
  double * d_x = NULL;     /* array (device) */

  /* Check we have a GPU, and get device name from the cudaDeviceProp
   * structure. This is for information. */

  int ndevice = 0;
  int deviceNum = -1;
  cudaDeviceProp prop;

  CUDA_ASSERT( cudaGetDeviceCount(&ndevice) );

  if (ndevice == 0) {
     printf("No GPU available!\n");
     exit(0);
  }

  CUDA_ASSERT( cudaGetDevice(&deviceNum) );
  CUDA_ASSERT( cudaGetDeviceProperties(&prop, deviceNum) );
  printf("Device %d name: %s\n", deviceNum, prop.name);
  printf("Maximum number of threads per block: %d\n", prop.maxThreadsPerBlock);


  /* allocate memory on host; assign some initial values */

  h_x   = (double *) malloc(sz);
  h_out = (double *) malloc(sz);
  assert(h_x);
  assert(h_out);

  for (int i = 0; i < ARRAY_LENGTH; i++) {
    h_x[i] = 1.0*i;
    h_out[i] = 0;
  }

  /* allocate memory on device */

  CUDA_ASSERT( cudaMalloc(&d_x, sz) );

  /* copy input array from host to GPU */

  CUDA_ASSERT( cudaMemcpy(d_x, h_x, sz, cudaMemcpyHostToDevice) );

  /* ... kernel will be here  ... */
  dim3 threads_per_block = { THREADS_PER_BLOCK, 1, 1};
  dim3 num_blocks { 1 + (ARRAY_LENGTH - 1)/THREADS_PER_BLOCK, 1, 1};

  myKernel<<< num_blocks, threads_per_block >>>(a, d_x);

  CUDA_ASSERT( cudaPeekAtLastError() );
  CUDA_ASSERT( cudaDeviceSynchronize() );

  /* copy the result array back to the host output array */

  CUDA_ASSERT( cudaMemcpy(h_out, d_x, sz, cudaMemcpyDeviceToHost) );

  /* We can now check the results ... */
  printf("Results:\n");
  {
    int ncorrect = 0;
    for (int i = 0; i < ARRAY_LENGTH; i++) {
      /* The print statement can be uncommented for debugging... */
      /* printf("%9d %5.2f\n", i, h_out[i]); */
      if (fabs(h_out[i] - a*h_x[i]) < DBL_EPSILON) ncorrect += 1;
    }
    printf("No. elements %d, and correct: %d\n", ARRAY_LENGTH, ncorrect);
  }

  /* free device buffer */

  CUDA_ASSERT( cudaFree(d_x) );

  /* free host buffers */
  free(h_x);
  free(h_out);

  return 0;
}

/* It is important to check the return code from API calls, so the
 * follow function/macro allow this to be done concisely as
 *
 *   CUDA_ASSERT(cudaRunTimeAPIFunction(...));
 *
 * Return codes may be asynchronous, and thus misleading! */

__host__ void myErrorHandler(cudaError_t ifail, const char * file,
                             int line, int fatal) {

  if (ifail != cudaSuccess) {
    fprintf(stderr, "Line %d (%s): %s: %s\n", line, file,
            cudaGetErrorName(ifail), cudaGetErrorString(ifail));
    if (fatal) exit(ifail);
  }

  return;
}

__global__ void myKernel(double a, double * x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < ARRAY_LENGTH) { 
    x[i] *= a;
  }

}
