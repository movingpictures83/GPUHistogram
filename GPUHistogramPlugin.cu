#include "GPUHistogramPlugin.h"

void histogram(unsigned int *input, unsigned int *bins,
               unsigned int num_elements, unsigned int num_bins) {

  // zero out bins
  cudaMemset(bins, 0, num_bins * sizeof(unsigned int));
  // Launch histogram kernel on the bins
  {
    dim3 blockDim(512), gridDim(30);
    histogram_kernel<<<gridDim, blockDim,
                       num_bins * sizeof(unsigned int)>>>(
        input, bins, num_elements, num_bins);
    cudaGetLastError();
    cudaDeviceSynchronize();
  }

  // Make sure bin values are not too large
  {
    dim3 blockDim(512);
    dim3 gridDim((num_bins + blockDim.x - 1) / blockDim.x);
    convert_kernel<<<gridDim, blockDim>>>(bins, num_bins);
    cudaGetLastError();
    cudaDeviceSynchronize();
  }
}

void GPUHistogramPlugin::input(std::string infile) {
readParameterFile(infile);
}

void GPUHistogramPlugin::run() {}

void GPUHistogramPlugin::output(std::string outfile) {
  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *deviceInput;
  unsigned int *deviceBins;
  inputLength = atoi(myParameters["N"].c_str());
  hostInput = (unsigned int*) malloc (inputLength*sizeof(unsigned int));
   std::ifstream myinput((std::string(PluginManager::prefix())+myParameters["data"]).c_str(), std::ios::in);
 int i;
 for (i = 0; i < inputLength; ++i) {
        int k;
        myinput >> k;
        hostInput[i] = k;
 }


 // hostInput = (unsigned int *)gpuTKImport(gpuTKArg_getInputFile(args, 0),
 //                                      &inputLength, "Integer");
  hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));

  //@@ Allocate GPU memory here
  cudaMalloc((void **)&deviceInput,
                        inputLength * sizeof(unsigned int));
  cudaMalloc((void **)&deviceBins, NUM_BINS * sizeof(unsigned int));
  cudaDeviceSynchronize();

  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput,
                        inputLength * sizeof(unsigned int),
                        cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  // Launch kernel
  // ----------------------------------------------------------

  histogram(deviceInput, deviceBins, inputLength, NUM_BINS);
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostBins, deviceBins,
                        NUM_BINS * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
        std::ofstream outsfile(outfile.c_str(), std::ios::out);
        int j;
        for (i = 0; i < NUM_BINS; ++i){
                outsfile << hostBins[i];//std::setprecision(0) << a[i*N+j];
                outsfile << "\n";
        }

  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);

  free(hostBins);
  free(hostInput);
}

PluginProxy<GPUHistogramPlugin> GPUHistogramPluginProxy = PluginProxy<GPUHistogramPlugin>("GPUHistogram", PluginManager::getInstance());

