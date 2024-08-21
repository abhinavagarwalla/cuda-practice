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

#define VSIZE 32
#define BLOCKSIZE 4

// When thinking of shared memory, think in terms of blocksize.
// no dependence on the actual size of the tensors i.e. VSIZE.
// this can lead to huge memory reserve. might be possible to do this
// but for now, aim to BLOCKSIZE dependent matrix multiplication. Nice!
__global__ void multiply(float* a, float* b, float* c, int size){
    __shared__ float as[BLOCKSIZE][BLOCKSIZE];
    __shared__ float bs[BLOCKSIZE][BLOCKSIZE];

    int gidx_row = threadIdx.x + blockIdx.x * blockDim.x;
    int gidx_col = threadIdx.y + blockIdx.y * blockDim.y;

    if((gidx_row < size) && (gidx_col < size)){
        // start sliding the local box across row/col
        for(int k=0;k<size/BLOCKSIZE;++k){
	    // copy into shared memory
	    as[threadIdx.x][threadIdx.y] = a[gidx_row*size + k*BLOCKSIZE + threadIdx.y];
	    bs[threadIdx.x][threadIdx.y] = b[(gidx_col+ k*BLOCKSIZE+threadIdx.x)*size + k*BLOCKSIZE+threadIdx.y];
    }
    __syncthreads();

    if((gidx_row < size) && (gidx_col < size)){
        for(int j=0;j<size;++j){
            c[gidx_row*size + gidx_col] += row[j] * col[j];
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

    dim3 block(BLOCKSIZE, BLOCKSIZE);
    dim3 grid((VSIZE + BLOCKSIZE - 1)/BLOCKSIZE, (VSIZE + BLOCKSIZE - 1)/BLOCKSIZE);
    multiply<<<grid, block>>>(da, db, dc, VSIZE);
    cudaCheckErrors("kernel execution failure");
    
    cudaMemcpy(c, dc, VSIZE*VSIZE*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    free(a); free(b);
    
    print_matrix(c, VSIZE);

}
