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

__global__ void multiply(float* a, float* b, float* c, int size){
    int idx_row = threadIdx.x + blockIdx.x * blockDim.x;
    int idx_col = threadIdx.y + blockIdx.y * blockDim.y;

    if((idx_row < size) && (idx_col < size)){
        for(int j=0;j<size;++j){
            c[idx_row*size + idx_col] += a[idx_row*size+j] * b[j*size+idx_col];
        }
    }
}

void print_matrix(float* c, int size){
    std::cout << "Result: \n";
    for(int r=0;r<size;++r){
        for(int i=0;i<size;++i) std::cout <<c[r*size+i] << ",";
        std::cout << std::endl;
    }
    return;
}

#define VSIZE 32
#define THREADS 4
int main(){
    float *a, *b, *c;
    a = new float[VSIZE*VSIZE];
    b = new float[VSIZE*VSIZE];
    c = new float[VSIZE*VSIZE];
    for(int i=0;i<VSIZE*VSIZE;++i){
        a[i] = 1.0f;
        b[i] = 2.0f;
        c[i] = 0.0f;
    }

    float *da, *db, *dc;

    //print_matrix(a, VSIZE);
    //print_matrix(b, VSIZE);

    cudaMalloc(&da, VSIZE*VSIZE*sizeof(float));
    cudaMalloc(&db, VSIZE*VSIZE*sizeof(float));
    cudaMalloc(&dc, VSIZE*VSIZE*sizeof(float));
    
    cudaCheckErrors("cudaMalloc error check.");

    cudaMemcpy(da, a, VSIZE*VSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, VSIZE*VSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dc, c, VSIZE*VSIZE*sizeof(float), cudaMemcpyHostToDevice);

    cudaCheckErrors("cudaMemcpy error check.");

    dim3 block(THREADS, THREADS);
    dim3 grid((VSIZE + THREADS - 1)/THREADS, (VSIZE + THREADS - 1)/THREADS);
    multiply<<<grid, block>>>(da, db, dc, VSIZE);
    //multiply<<<VSIZE/THREADS, THREADS>>>(da, db, dc, VSIZE);
    cudaCheckErrors("kernel execution failure");
    
    cudaMemcpy(c, dc, VSIZE*VSIZE*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    free(a); free(b);
    
    print_matrix(c, VSIZE);

}
