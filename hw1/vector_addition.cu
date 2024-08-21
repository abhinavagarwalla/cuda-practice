#include <iostream>
#include <stdio.h>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

__global__ void add(float* a, float* b, float* c, int size){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < size) c[idx] = a[idx] + b[idx];
}

void print_array(float *c, int size){
    std::cout << "Result: \n";
    for(int i=0;i<size;++i) std::cout <<c[i] << ",";
    std::cout << std::endl;
    return;
}

#define VSIZE 1024
#define THREADS 512
int main(){
    float *a, *b, *c;
    a = new float[VSIZE];
    b = new float[VSIZE];
    c = new float[VSIZE];
    float *da, *db, *dc;

    for(int i=0;i<VSIZE;++i){
        a[i] = 1.0f * rand() / VSIZE;
        b[i] = 1.0f * rand() / VSIZE;
        c[i] = 0.0f;
    }
    //print_array(a, VSIZE);
    //print_array(b, VSIZE);

    cudaMalloc(&da, VSIZE*sizeof(float));
    cudaMalloc(&db, VSIZE*sizeof(float));
    cudaMalloc(&dc, VSIZE*sizeof(float));

    cudaCheckErrors("cudaMalloc error check.");

    cudaMemcpy(da, a, VSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, VSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dc, c, VSIZE*sizeof(float), cudaMemcpyHostToDevice);

    cudaCheckErrors("cudaMemcpy error check.");

    add<<<(VSIZE + THREADS-1)/THREADS, THREADS>>>(da, db, dc, VSIZE);
    cudaCheckErrors("kernel execution failure");
    
    cudaMemcpy(c, dc, VSIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    
    print_array(c, VSIZE);

}
