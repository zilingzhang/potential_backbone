#include <iostream>
#include <chrono>
#include <stdio.h>

struct Source{
    double x;
    double y;
    double z;
};

__global__ void
potential_reduce(
    Source *query_point,
    Source *sources,
    const int N
){
    __shared__ double vals[33];

}

int main(int argc, char **argv)
{
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
    
    cudaDeviceSynchronize();

}