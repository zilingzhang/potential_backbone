#include <iostream>
#include <fstream>

__global__ void transpose(const int *in, int *out, 
    const int w, const int h){

    //Parallel Load into shared memory
    __shared__ int tile[32][33];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    if (x < w && y < h){
        tile[threadIdx.y][threadIdx.x] = in[y*w + x];
    }
    __syncthreads();
    
    //Block-wise transpose
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    if (y < w && x < h){
        out[y*h + x] = tile[threadIdx.x][threadIdx.y];
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
    int *m_transposed;

    //Unified memory allocation
    cudaMallocManaged(&m, N*sizeof(int));
    cudaMallocManaged(&m_transposed, N*sizeof(int));

    for(int i=0;i<N;i++) infile >> m[i];    
    infile.close();

    //Block,Grid parameters
    dim3 dimGrid((w+32-1)/32,(h+32-1)/32,1);
    dim3 dimBlock(32,32,1);
    
    //prefetch data to GPU
    int device = -1;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(m, N*sizeof(int), device, NULL);
    cudaMemPrefetchAsync(m_transposed, N*sizeof(int), device, NULL);
 
    //Block-wise Matrix Transpose
    transpose<<<dimGrid, dimBlock>>>(m,m_transposed,w,h); 

    //Sync
    cudaDeviceSynchronize();        
    
    //Standard Output
    std::cout << h << " " << w << std::endl;
    for(int i=0;i<w;i++){
        for (int j=0;j<h;j++){
            std::cout << m_transposed[i*h+j] << " ";
        }
        std::cout << std::endl;
    }

    //Garbage collection
    cudaFree(m);
    cudaFree(m_transposed);            

    return 0;
}