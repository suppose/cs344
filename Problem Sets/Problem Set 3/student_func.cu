/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <cstdio>

__global__ void reduce_kernel(float* d_out, const float* const d_in, const int arraySize, bool isMax) {
    extern __shared__ float sdata[];
    const int myId = blockIdx.x*blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    
    //if(myId >= arraySize)
    //    return;
    
	if(myId < arraySize)
		sdata[tid] = d_in[myId];
    __syncthreads();
    int maxId = 0;
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && myId + s < arraySize) {
			float val1 = sdata[tid];
			float val2 = sdata[tid+s];
			sdata[tid] = isMax ? (val1 > val2 ? val1 : val2) : (val1 < val2 ? val1 : val2);
			maxId = isMax ? (val1 > val2 ? myId : myId+s) : (val1 < val2 ? myId : myId+s);
        }
        __syncthreads();
    }
    
    if (tid == 0)
        d_out[blockIdx.x] = sdata[0];
}


__global__ void histo_kernel(unsigned int* d_histogram, 
                             const float* const d_lum, 
                             const float lumMin, 
                             const float lumRange, 
                             const size_t numBins,
                             const size_t arraySize)
{
    const int myId = blockIdx.x * blockDim.x + threadIdx.x;
    if (myId >= arraySize)
        return;
    unsigned int bin = min(static_cast<unsigned int>(numBins - 1),
                           static_cast<unsigned int>((d_lum[myId] - lumMin) / lumRange * static_cast<float>(numBins))); // for rounding error
    atomicAdd(&d_histogram[bin], 1);
}

//__global__ void blelloch_scan_reduce_kernel (unsigned int* const d_out, 
//                                             unsigned int* d_in,
//                                             const size_t numBins)
//{
//    extern __shared__ unsigned int reduce_data[];
//    const int numThreads = gridDim.x*blockDim.x;
//    const int chunkSize = ceil((float)numBins / (float)numThreads);
//    const int myId = (blockIdx.x * blockDim.x + threadIdx.x + 1) * chunkSize -1;
//    if (myId >= numBins)
//        return;
//    // Copy data to shared memory
//    const int tid = threadIdx.x;
//    reduce_data[tid] = d_in[myId];
//    __syncthreads();
//    
//    
//    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
//        if(tid - offset >= 0 && (tid+1)%(2*offset) == 0) {
//            reduce_data[tid] += reduce_data[tid-offset];
//        }
//        __syncthreads();
//    }
//    d_out[myId] = reduce_data[tid];
//}
//
//__global__ void blelloch_scan_down_sweep (unsigned int* d_in_out,
//                                          const size_t numBins)
//{
//    extern __shared__ unsigned int sweep_data[];
//    const int numThreads = gridDim.x*blockDim.x;
//    const int chunkSize = ceil((float)numBins / (float)numThreads);
//    const int myId = (blockIdx.x * blockDim.x + threadIdx.x + 1) * chunkSize -1;
//    if (myId >= numBins)
//        return;
//    // Copy data to shared memory
//    const int tid = threadIdx.x;
//    if(myId == numBins -1)
//        sweep_data[tid] = 0;
//    else
//        sweep_data[tid] = d_in_out[myId];
//    __syncthreads();
//    
//    
//    for (int offset = blockDim.x >> 1; offset >=1; offset >>= 1) {
//        if(tid - offset >= 0 && (blockDim.x-1-tid)%(offset << 1) == 0) {
//            int tmp = sweep_data[tid];
//            sweep_data[tid] += sweep_data[tid-offset];
//            sweep_data[tid-offset] = tmp;
//        }
//        __syncthreads();
//    }
//    d_in_out[myId] = sweep_data[tid];
//}

__global__ void hillis_steele_scan (unsigned int* const d_out, 
                                    unsigned int* d_in,
                                    const size_t numBins)
{
	extern __shared__ int scan_data[];
	const int myId = blockIdx.x * blockDim.x + threadIdx.x;
	if(myId >= numBins)
		return;
	const int tid = threadIdx.x;
	scan_data[tid] = d_in[myId];
	__syncthreads();

	for (unsigned int step = 1; step < numBins; step <<= 1) {
		if (tid >= step) {
			scan_data[tid] += scan_data[tid-step];
		}
		__syncthreads();
	}
	if (myId == 0)
		d_out[myId] = 0;
	else if(myId < numBins)
    {
        d_out[myId] = scan_data[tid-1];
    }

}



void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
    float* d_tmp;
    float* d_out;
    unsigned int* d_histogram;
    
    const size_t arraySize = numRows * numCols;
    const size_t arrayBytes = arraySize * sizeof(float);
    size_t numThreads = 512;
    size_t numBlocks = ceil(((float)arraySize) / ((float)numThreads));
    printf("Array size: %d\n", arraySize);
	printf("block number: %d\n", numBlocks);
    // Allocate memory
    checkCudaErrors(cudaMalloc(&d_tmp, sizeof(float) * numBlocks));
    checkCudaErrors(cudaMalloc(&d_out, sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_histogram, sizeof(unsigned int)*numBins));
    checkCudaErrors(cudaMemset(d_histogram, 0, sizeof(unsigned int)*numBins));
    
    reduce_kernel<<<numBlocks, numThreads, numThreads * sizeof(float)>>>
        (d_tmp, d_logLuminance, arraySize, false);
    
	int threads = 1;
	while (threads < numBlocks)
		threads <<= 1;
    reduce_kernel<<<1, threads, threads * sizeof(float)>>>  // # of threads must be power of 2!
        (d_out, d_tmp, numBlocks, false);
    
    checkCudaErrors(cudaMemcpy(&min_logLum, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    
    reduce_kernel<<<numBlocks, numThreads, numThreads * sizeof(float)>>>
        (d_tmp, d_logLuminance, arraySize, true);
    
    reduce_kernel<<<1, threads, threads * sizeof(float)>>> // # of threads must be power of 2!
        (d_out, d_tmp, numBlocks, true);
	
    checkCudaErrors(cudaMemcpy(&max_logLum, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("min: %f\n", min_logLum);
    printf("max: %f\n", max_logLum);
	
	cudaFree(d_tmp);
    cudaFree(d_out);

    float lumRange = max_logLum - min_logLum;
	printf("Bins: %d, lumRang: %f\n", numBins, lumRange);
    numBlocks = ceil(((float)arraySize) / ((float)numThreads));
    histo_kernel<<<numBlocks, numThreads>>>
                                (d_histogram, 
                                 d_logLuminance, 
                                 min_logLum, 
                                 lumRange, 
                                 numBins,
                                 arraySize);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    //// scan reduce
    //numBlocks = ceil( (float)numBins / (float)numThreads );
    //blelloch_scan_reduce_kernel<<<numBlocks, numThreads, numThreads*sizeof(int)>>>
    //    ( d_cdf,
    //      d_histogram,
    //      numBins );
    //
    //blelloch_scan_reduce_kernel<<<1, numBlocks, numBlocks*sizeof(int)>>>
    //    (d_cdf,
    //     d_cdf,
    //     numBins );
    //
    //// scan down sweep
    //blelloch_scan_down_sweep<<<1, numBlocks, numBlocks*sizeof(int)>>>
    //    (d_cdf,
    //     numBins);
    //blelloch_scan_down_sweep<<<numBlocks, numThreads, numThreads*sizeof(int)>>>
    //    (d_cdf,
    //     numBins);

	hillis_steele_scan<<<1, numBins, numBins*sizeof(unsigned int)>>>(d_cdf, d_histogram, numBins);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	cudaFree(d_histogram);
}

