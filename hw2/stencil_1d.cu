#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <time.h>


#define N 1024
#define RADIUS 3
#define BLOCKSIZE 32

void print_array(int *out, int size){
    std::cout << "Result: \n";
    for(int i=0;i<size;++i) std::cout << out[i] << ",";
    std::cout << std::endl;
}

__global__ void stencil_1d(int* in, int* out){
    __shared__ int tmp[BLOCKSIZE + 2*RADIUS];
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int lidx = threadIdx.x + RADIUS;

    //start the data move to shared memory from GPU global memory
    tmp[lidx] = in[gidx];
    //now get the windows.
    if(threadIdx.x < RADIUS){
        tmp[lidx - RADIUS] = in[gidx - RADIUS];
	tmp[lidx + blockDim.x] = in[gidx + blockDim.x];
    }

    __syncthreads();

    if(gidx >= RADIUS && gidx < N+RADIUS){
        for(int i=1;i<=RADIUS;++i){
	    out[gidx] += tmp[lidx-i] + tmp[lidx+i];
	}
	out[gidx] += tmp[lidx];
    }
}

int  main(){
    
    int *in, *out;
    int *d_in, *d_out;

    clock_t t0, t1, t2;
    t0 = clock();

    int num_elem = N + 2*RADIUS;
    in = (int*)malloc(num_elem*sizeof(int));
    out = (int*)malloc(num_elem*sizeof(int));
    std::fill_n(in, num_elem, 1);
    std::fill_n(out, num_elem, 0);

    t1 = clock();
    // Phase 1: bring the data to device
    cudaMalloc(&d_in, num_elem*sizeof(int));
    cudaMalloc(&d_out, num_elem*sizeof(int));

    cudaMemcpy(d_in, in, num_elem*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out, num_elem*sizeof(int), cudaMemcpyHostToDevice);


    // Phase 2: Process the data
    int grid = (num_elem + BLOCKSIZE-1)/BLOCKSIZE;
    stencil_1d<<<grid, BLOCKSIZE>>>(d_in, d_out);

    cudaMemcpy(out, d_out, num_elem*sizeof(int), cudaMemcpyDeviceToHost);
    t2 = clock();

    std::cout << "Init took " << ((double)(t1-t0))/CLOCKS_PER_SEC << " seconds.\n";
    std::cout << "Compute took " << ((double)(t2-t1))/CLOCKS_PER_SEC << " seconds.\n";
    print_array(out, num_elem);
    return 0;
}

