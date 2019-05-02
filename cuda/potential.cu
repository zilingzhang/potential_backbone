#include <iostream>
#include <chrono>
#include <stdio.h>
#include <math.h>
#include <chrono>

struct Source{
    double x;
    double y;
    double z;
};

// Since sm35 is the targeted platform, and doesn't have float64 atomicAdd implemented,
// We need a custom atomicAdd function

__device__ double atomicAdd_sm35(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void
potential_reduce(
    struct Source query_point,
    struct Source *sources,
    const int N,
    double *partialSum,
    double *sum
){
    if(threadIdx.x==0){
        partialSum[blockIdx.x]=0;
    }
    __syncthreads();    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double threadSum =1;
    if(i<N){        
        // Compute point source contribution
        double r = sqrt(
            pow((sources[i].x-query_point.x),2)
            +pow((sources[i].y-query_point.y),2)
            +pow((sources[i].z-query_point.z),2)
        );
        threadSum = 1.0/r;
        // Block Sum
        atomicAdd_sm35(&partialSum[blockIdx.x],threadSum);
        __syncthreads();        
    }
    
    if(threadIdx.x==0){
        // Global Sum;
        atomicAdd_sm35(&sum[0],partialSum[blockIdx.x]);
    }
}

int main(int argc, char **argv)
{
    auto start = std::chrono::system_clock::now();
    int N = 31200;
    struct Source *sources;
    cudaMallocManaged(&sources,N * sizeof(struct Source));

    // Create a 10m x 2m x 2m box with 31200 point source on the surface
    int count = 0;
    for(int i=-100;i<100;i++){
		for(int j=-19;j<19;j++){
			double x=i*0.05+0.025;
			double y=-1.0;
			double z=j*0.05+0.025;
            sources[count].x = x;
            sources[count].y = y;
            sources[count].z = z;
            count++;
			y=1.0;
			sources[count].x = x;
            sources[count].y = y;
            sources[count].z = z;
            count++;
		}
	}
	for(int i=-100;i<100;i++){
		for(int j=-20;j<20;j++){
			double x=i*0.05+0.025;
			double y=j*0.05+0.025;
			double z=-1.0;
            sources[count].x = x;
            sources[count].y = y;
            sources[count].z = z;
            count++;
			z=1.0;
            sources[count].x = x;
            sources[count].y = y;
            sources[count].z = z;
            count++;
		}
    }
    
    int blockSize = 256;
    int numBlocks = (N+blockSize -1)/blockSize;
    double *partialSum;
    double *sum;
    cudaMallocManaged(&partialSum,numBlocks*sizeof(double));
    cudaMallocManaged(&sum,sizeof(double));

    struct Source query_point;
    query_point.x = -2.0;
    query_point.y = 0;
    query_point.z = 0;

    // auto start = std::chrono::system_clock::now();

    for(int i=0;i<10;i++){
        sum[0]=0;        
        potential_reduce<<<numBlocks,blockSize>>>(query_point,sources,N,partialSum,sum);
        cudaDeviceSynchronize();
        std::cout 
         << "---" << std::endl
         << query_point.x << std::endl
         << query_point.y << std::endl
         << query_point.z << std::endl
         << "---" << std::endl
         << sum[0] 
         << std::endl;
         query_point.x+=0.5;
    }

    
    auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;  
	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
}