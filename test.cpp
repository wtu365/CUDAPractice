#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda>
#include <stdexcept>

#define BLOCK_SIZE 16

// using namespace std;


inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

__global__ void VectorAdd(double * A, double * B, double * C){

    int i = threadIdx.x;
    printf("%d\n", i);
    C[i] = A[i] + B[i];
    //printf("%f\n", A[i]);
    //printf("%f\n", B[i]);
    //printf("%f\n", C[i]);
}


int main(){
    size_t size = 10;

    double * A = (double *) malloc((sizeof(double) * size));
    double * B = (double *) malloc((sizeof(double) * size));
    double * C = (double *) malloc((sizeof(double)) * size);

    for (int i = 0; i < size; ++i){
        A[i] = i;
        B[i] = i;
        C[i] = 0;
        //printf("%f\n", A[i]);
        //printf("%f\n", B[i]);
        //printf("%f\n", C[i]);
    }


    double *dA, *dB, *dC;
    gpuErrorCheck(cudaMalloc(&dA, sizeof(double) * size))
    gpuErrorCheck(cudaMalloc(&dB, sizeof(double) * size))
    gpuErrorCheck(cudaMalloc(&dC, sizeof(double) * size))
    
    gpuErrorCheck(cudaMemcpy(dA, A, sizeof(double) * size, cudaMemcpyHostToDevice))
    gpuErrorCheck(cudaMemcpy(dB, B, sizeof(double) * size, cudaMemcpyHostToDevice))

    VectorAdd<<< 1, size>>>(dA, dB, dC);

    gpuErrorCheck(cudaMemcpy(C, dC, sizeof(double) * size, cudaMemcpyDeviceToHost))

    for (int i = 0; i < size; ++i)
        // cout << i << endl;
        printf("%f\n", C[i]);

    gpuErrorCheck(cudaFree(dA))
    gpuErrorCheck(cudaFree(dB))
    gpuErrorCheck(cudaFree(dC))
    free(A);
    free(B);
    free(C);
}