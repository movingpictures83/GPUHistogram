#ifndef GPUHISTORGRAMPLUGIN_H
#define GPUHISTORGRAMPLUGIN_H

#include "Plugin.h"
#include "Tool.h"
#include "PluginProxy.h"
#include <string>
#include <map>

class GPUHistogramPlugin : public Plugin, public Tool {

	public:
		void input(std::string file);
		void run();
		void output(std::string file);
	private:
                std::string inputfile;
		std::string outputfile;
 //               std::map<std::string, std::string> parameters;
};


#define NUM_BINS 4096

#define CUDA_CHECK(ans)                                                   \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Privatized bins
  extern __shared__ unsigned int bins_s[];
  for (unsigned int binIdx = threadIdx.x; binIdx < num_bins;
       binIdx += blockDim.x) {
    bins_s[binIdx] = 0;
  }
  __syncthreads();

  // Histogram
  for (unsigned int i = tid; i < num_elements;
       i += blockDim.x * gridDim.x) {
    atomicAdd(&(bins_s[input[i]]), 1);
  }
  __syncthreads();

  // Commit to global memory
  for (unsigned int binIdx = threadIdx.x; binIdx < num_bins;
       binIdx += blockDim.x) {
    atomicAdd(&(bins[binIdx]), bins_s[binIdx]);
  }
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num_bins) {
    bins[tid] = min(bins[tid], 127);
  }
}

#endif
