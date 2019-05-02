#include <stdio.h>
#include <stdlib.h>
#include <iostream>

// The spmv kernel from nvidia document, with bug fix and modification
// https://www.nvidia.com/docs/IO/66889/nvr-2008-004.pdf

__global__ void
spmv_csr_vector_kernel(const int num_rows,
                       const int *ptr,
                       const int *indices,
                       const float *data,
                       cudaTextureObject_t b_tex,
                       float *y)
{
    //used 33 to avoid memory bank conflict
    __shared__ double vals[33];

    // 32 threads per row, each thread load matrix(data) and vector(x) 32 elements apart
    // memory access to global memory between 32 threads should be coalesced
    // To enable quicker memory reuse, cache vector b into a texture object

    int row = blockIdx.x;
    if (row < num_rows)
    {
        int row_start = ptr[row];
        int row_end = ptr[row + 1];
        
        // compute running sum per thread
        vals[threadIdx.x] = 0;
        for (int jj = row_start + threadIdx.x; jj < row_end; jj += 32)
            vals[threadIdx.x] += data[jj] * tex1Dfetch<float>(b_tex,indices[jj]);
        __syncthreads();

        // parallel reduction in shared memory
        if (threadIdx.x < 16)
            vals[threadIdx.x] += vals[threadIdx.x + 16];
        __syncthreads();
        if (threadIdx.x < 8)
            vals[threadIdx.x] += vals[threadIdx.x + 8];
        __syncthreads();
        if (threadIdx.x < 4)
            vals[threadIdx.x] += vals[threadIdx.x + 4];
        __syncthreads();            
        if (threadIdx.x < 2)
            vals[threadIdx.x] += vals[threadIdx.x + 2];
        __syncthreads();
        if (threadIdx.x < 1)
            vals[threadIdx.x] += vals[threadIdx.x + 1];
        __syncthreads();

        // first thread writes the result
        if (threadIdx.x == 0)
            y[row] += vals[threadIdx.x];
    }
}

int main(int argc, char **argv)
{
    FILE *fp;
    char line[1024];
    int *ptr, *indices;
    float *data, *b, *t, *t_gpu;
    int i, j;
    int n;  // number of nonzero elements in data
    int nr; // number of rows in matrix
    int nc; // number of columns in matrix

    // Open input file and read to end of comments
    if (argc != 2)
        abort();

    if ((fp = fopen(argv[1], "r")) == NULL)
    {
        abort();
    }

    fgets(line, 128, fp);
    while (line[0] == '%')
    {
        fgets(line, 128, fp);
    }

    // Read number of rows (nr), number of columns (nc) and
    // number of elements and allocate memory for ptr, indices, data, b and t.
    sscanf(line, "%d %d %d\n", &nr, &nc, &n);
    
    // Unified Memory Allocation
    cudaMallocManaged(&ptr,(nr + 1) * sizeof(int));
    cudaMallocManaged(&indices,n * sizeof(int));
    cudaMallocManaged(&data,n * sizeof(float));
    cudaMallocManaged(&b,nc * sizeof(float));
    cudaMallocManaged(&t_gpu,nr * sizeof(float));

    t = (float *) malloc(nr*sizeof(float));

    // Read data in coordinate format and initialize sparse matrix
    int lastr = 0;
    for (i = 0; i < n; i++)
    {
        int r;
        fscanf(fp, "%d %d %f\n", &r, &(indices[i]), &(data[i]));
        indices[i]--; // start numbering at 0
        if (r != lastr)
        {
            ptr[r - 1] = i;
            lastr = r;
        }
    }
    ptr[nr] = n;

    // initialize t to 0 and b with random data
    for (i = 0; i < nr; i++)
    {
        t[i] = 0.0;
    }

    for (i = 0; i < nc; i++)
    {
        b[i] = (float)rand() / 1111111111;
    }

    // MAIN COMPUTATION, SEQUENTIAL VERSION
    for (i = 0; i < nr; i++)
    {
        for (j = ptr[i]; j < ptr[i + 1]; j++)
        {
            t[i] = t[i] + data[j] * b[indices[j]];
        }
    }

    // Compute result on GPU and compare output

    int device = -1;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(ptr, (nr + 1) * sizeof(int), device, NULL);
    cudaMemPrefetchAsync(indices, n * sizeof(int), device, NULL);
    cudaMemPrefetchAsync(data, n * sizeof(float), device, NULL);    
    cudaMemPrefetchAsync(t_gpu, nr * sizeof(float), device, NULL);

    // Use read only texture object to cache vector b, saving ~100 cycle on read
    // refered to https://devblogs.nvidia.com/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility/
    
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = b;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32; // bits per channel
    resDesc.res.linear.sizeInBytes = nr*sizeof(float);

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
        
    cudaTextureObject_t tex=0;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);    

    //Call Kernel
    dim3 dimGrid(nr,1,1);
    dim3 dimBlock(32,1,1);
    spmv_csr_vector_kernel<<<dimGrid,dimBlock>>>(nr,ptr,indices,data,tex,t_gpu);

    //Sync
    cudaDeviceSynchronize();

    //Result Validation
    for(i=0;i<nr;i++){
        if(fabs(t[i]-t_gpu[i])>1e-5){
            std::cout<< "Not equal at " << i << "\n";
            std::cout<< t[i] << " " << t_gpu[i] << " " << "\n";
            abort();
        }
    }

    //Garbage collection
    cudaFree(ptr);
    cudaFree(indices);
    cudaFree(data);
    cudaFree(b);  
    cudaFree(t_gpu);
    cudaDestroyTextureObject(tex);
}
