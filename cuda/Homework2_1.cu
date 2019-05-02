#include <iostream>
#include <fstream>
#include <chrono>

__global__
void sumOne(int n,int *m,int *partialSum,int *sum){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int threadSum =0;
    for (int i = index; i < n; i += stride){
      if(m[i]==1){
          threadSum++;
      }
    }
    // Block Sum
    atomicAdd(&partialSum[blockIdx.x],threadSum);
    __syncthreads();

    if(threadIdx.x==0){
        // Global Sum;
        atomicAdd(&sum[0],partialSum[blockIdx.x]);
    }
}

int main(int argc,char **argv){

    //Read input matrix
    std::ifstream infile;
    infile.open(argv[1]);
    if (!infile.is_open()){
        std::cerr << "Couldn't read " << argv[1] << std::endl;
		return 0;
    }

    int w,h;
    infile >> w >> h;
    int N = w*h;
    int *m;
    
    //Unified memory allocation
    cudaMallocManaged(&m, N*sizeof(int));

    for(int i=0;i<N;i++){
        infile >> m[i];
    }
    infile.close();

    auto start = std::chrono::system_clock::now();
    //Block,Grid parameters
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    int *partialSum;
    int *sum;
    cudaMallocManaged(&partialSum,numBlocks*sizeof(int));
    cudaMallocManaged(&sum, sizeof(int));

    //prefetch input matrix
    int device = -1;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(m, N*sizeof(int), device, NULL);
    cudaMemPrefetchAsync(partialSum, numBlocks*sizeof(int), device, NULL);
    cudaMemPrefetchAsync(sum, sizeof(int), device, NULL);

    //Sum ones
    sumOne<<<numBlocks, blockSize>>>(N, m, partialSum,sum); 

    cudaDeviceSynchronize();        
    std::cout << sum[0] << std::endl;

    cudaFree(m);
    cudaFree(partialSum);
    cudaFree(sum);

    return 0;
}