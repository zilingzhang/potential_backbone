#include <stdio.h>
#include <stdlib.h>
#include <iostream>

__global__ void block_scan(
    unsigned long long *g_odata, 
    unsigned long long *g_idata, 
    unsigned long long n, 
    unsigned long long *block_sums){

    __shared__ unsigned int temp[1024];

    int tid = threadIdx.x;
    int offset = 1;
    int block_offset = blockIdx.x * 1024;

    //Parallel Load into shared memory, each thread load 2 elements
    if(block_offset + 2*tid<n){
        temp[2*tid] = g_idata[block_offset + 2*tid];
    }
    else{
        temp[2*tid] = 0;
    }
    if(block_offset + 2*tid+1<n){
        temp[2*tid+1] = g_idata[block_offset + 2*tid+1];
    }
    else{
        temp[2*tid+1]=0;
    }
    
    //Downward pass
    for (int d = 1024>>1;d>0;d>>=1)
    {
        __syncthreads();
        if(tid < d)
        {
            //Compute indices of 2 elements to be handled
            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    //Zero the last element
    if (tid == 0){
        temp[1023]=0;
    }

    //Upward pass
    for (int d=1;d<1024;d*=2)
    {
        offset >>= 1;
        __syncthreads();
        if(tid<d)
        {
            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;
            int swap = temp[ai];
            temp[ai] = temp[bi];
            temp[bi]+= swap;
        }
    }

    __syncthreads();
    
    if(block_offset + 2*tid<n){
        g_odata[block_offset + 2*tid] = temp[2*tid];
    }
    if(block_offset + 2*tid+1<n){
        g_odata[block_offset + 2*tid + 1] = temp[2*tid+1];
    }

    //Compute block sum
    if(tid==0){
        int bid = blockIdx.x;
        if(1024*bid+1023<n){
            block_sums[bid] = temp[1023]+g_idata[1024*bid+1023];
        }
        else{
            block_sums[bid] = temp[1023];
        }
    }
    __syncthreads();    
}

__global__ void add_block_sums(
    unsigned long long *A_gpu, 
    unsigned long long *a, 
    unsigned long long N,
    unsigned long long *block_sums){
    //Load block sums    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_offset = blockIdx.x * 1024;
    __shared__ long long blocksum;
    blocksum = block_sums[bid];    
    if(block_offset + 2*tid<N){
        A_gpu[block_offset + 2*tid] += blocksum;
    }
    if(block_offset + 2*tid+1<N){
        A_gpu[block_offset + 2*tid+1] += blocksum;
    }
    __syncthreads();
}

bool scan(
    unsigned long long *array_out,
    unsigned long long *array_in,
    unsigned long long N){
    // printf("scan %llu\n",N);
    //Allocate block sum
    unsigned long long numOfBlocks;    
    if(N>N/1024*1024){numOfBlocks = N/1024+1;}
    else{numOfBlocks = N/1024;}
    unsigned long long *block_sums;
    cudaMallocManaged(&block_sums,numOfBlocks*sizeof(unsigned long long));

    //Pascal+ GPU prefetch
    int device = -1;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(array_in,N*sizeof(unsigned long long),device, NULL);
    cudaMemPrefetchAsync(array_out,N*sizeof(unsigned long long),device, NULL);
    cudaMemPrefetchAsync(block_sums,numOfBlocks*sizeof(unsigned long long),device, NULL);

    //Scan 1024 element blocks
    block_scan<<<numOfBlocks,512>>>(array_out,array_in,N,block_sums);
    cudaDeviceSynchronize();
    // printf("%llu block scan completed!\n",numOfBlocks);

    //Scan block sums
    unsigned long long *block_sums_out;
    cudaMallocManaged(&block_sums_out,numOfBlocks*sizeof(unsigned long long));
    cudaMemPrefetchAsync(block_sums_out,numOfBlocks*sizeof(unsigned long long),device, NULL);
    
    if(numOfBlocks>1){
        scan(block_sums_out,block_sums,numOfBlocks);        
    }
    //Add block sums
    add_block_sums<<<numOfBlocks,512>>>(array_out,array_in,N,block_sums_out);
    cudaDeviceSynchronize();
    
    // printf("%llu block sum completed!\n",N);

    cudaFree(block_sums);
    cudaFree(block_sums_out);
    return true;
}

int main(int argc, char **argv)
{
    // 1) Take a positive integer N as an argument
    unsigned long long N;
    if(argc>1){
        N = std::stoll(argv[1]);
    }
    else{
        std::cerr << "Usage: ./homework4 N" << std::endl
            << "Testing with N=1000000" << std::endl;        
        N=1000000;
    }

    // 2) Create an input integer array a[N] of size N
    unsigned long long *a;    
    cudaMallocManaged(&a,N*sizeof(unsigned long long));

    // 3) Populate the array with random integers from he range [1,1000]
    for(unsigned long long i=0;i<N;i++){
        a[i]=rand()%1000+1;
    }

    // 4) Compute the scan output array A_cpu in sequential on the CPU
    unsigned long long *A_cpu;
    A_cpu = new unsigned long long[N];
    A_cpu[0]=0;
    for(unsigned long long i =1;i<N;i++){
        A_cpu[i]=A_cpu[i-1]+a[i-1];
    }

    // 5) Compute the scan output array A_gpu on the GPU   
    unsigned long long *A_gpu;
    cudaMallocManaged(&A_gpu,N*sizeof(unsigned long long));
    scan(A_gpu,a,N);    

    // 6) Compare A_cpu and A_gpu
    int error(0);
    for(int i=0;i<N;i++){
        if(A_cpu[i]!=A_gpu[i]){
            std::cout << "Not equal at " << i << ": "<< A_cpu[i] << " " << A_gpu[i] << std::endl;
            exit(0);
            error++;            
        }
    }
    if(error){
        printf("Scan Failed!\n");
    }
    else{
        printf("Scan Successful!\n");
    }

    cudaFree(A_gpu);
    cudaFree(a);
}
