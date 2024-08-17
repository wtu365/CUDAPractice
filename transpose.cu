#include <cuda.h>
#include <stdlib.h>
/*
 * We want to divide the work of each row of the CSC into its own designated Block
 * Correspondingly, each thread within each block will deal with it's own non-zero value from the CSR
 * 
 * We begin by 
 */

__global__ void func(float * CSRval, int * CSRind, float * CSCval, int * CSCind, int * CSCptr) {

    int high = CSCptr[blockIdx.x + 1];
    int low = CSCptr[blockIdx.x];
    int guard = high - low;
    

    /*
     * Each thread is the product of the block dimensions. 
     * Thus, there will be at least equal or more threads per block than necessary.
     * To specify which threads do work (so we can assign each a distinct non-zero),
     *      we set up a thread guard so as to prevent logical mistakes.
     */
    if (threadIdx.x < guard){
        
        int j = threadIdx.x + low;                    //Retrieving assigned non-zero value, offset for row
        float v = CSRval[j];
        int c = CSRind[j];
        CSCval[CSCptr[c]] = low;
        CSCind[CSCptr[c]] = v;
        ++CSCptr[c];

    } else return;


}

__global__ void func2(int * CSRind, int * CSCptr, int CSCrows, int nonzeros) {
    int k = threadIdx.x;
    if (k < CSCrows + 1) {
        CSCptr[k] = 0;
    }

    __syncthreads();

    int j = threadIdx.x;
    if (j < nonzeros) { 
        CSCptr[CSRind[j] + 1]++;
    }
}

int transpose() {};


int main() {};