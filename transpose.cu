#include <cuda.h>
#include <stdlib.h>
#include <stdexcept>
/*
 * We want to divide the work of each row of the CSC into its own designated Block
 * Correspondingly, each thread within each block will deal with it's own non-zero value from the CSR
 * 
 * We begin by establishing j value (unique non-zero identifier),
 *      then doing a lookup into said value and index of CSR. 
 * We then set the transposed position of value and index (using ptr row value) to said value and index.
 * 
 * A Final increment of the ptr value in question is necessary, to point to the next spot in the chain,
 *      for the next value index pair to be slotted in. Changes to ptr non-binding, b/c we won't cpy it back.
 */

typedef struct CSR {
    int nrows; // number of rows
    int ncols; // number of rows
    int * ind; // column ids
    float * val; // values
    int * ptr; // pointers (start of row in ind/val)\

    CSR()
    {
        nrows = ncols = 0;
        ind = nullptr;
        val = nullptr;
        ptr = nullptr;
    }

    void reserve(const int nrows, const int nnz)
    {
        if(nrows > this->nrows){
            if(ptr){
                ptr = (int*) realloc(ptr, sizeof(int) * (nrows+1));
            } else {
                ptr = (int*) malloc(sizeof(int) * (nrows+1));
                ptr[0] = 0;
            }
            if(!ptr){
                throw std::runtime_error("Could not allocate ptr array.");
            }
        }
        if(nnz > ptr[this->nrows]){
            if(ind){
                ind = (int*) realloc(ind, sizeof(int) * nnz);
            } else {
                ind = (int*) malloc(sizeof(int) * nnz);
            }
            if(!ind){
                throw std::runtime_error("Could not allocate ind array.");
            }
            if(val){
                val = (float*) realloc(val, sizeof(float) * nnz);
            } else {
                val = (float*) malloc(sizeof(float) * nnz);
            }
            if(!val){
                throw std::runtime_error("Could not allocate val array.");
            }
        }
        this->nrows = nrows;
    }

    ~CSR() {
        if (ind) {
            free(ind);
        }
        if (val) {
            free(val);
        }
        if (ptr) {
            free(ptr);
        }
    }
} CSR;

__global__ void transposition(float * CSRval, int * CSRind, float * CSCval, int * CSCind, int * CSCptr) {

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
        
        int j = threadIdx.x + low;                      //Retrieving assigned non-zero value, offset for row
        float v = CSRval[j];                            //Grab future value
        int c = CSRind[j];                              //Grab future row
        /*
         * Set future value (position determined by current spot open in row) properly
         * Set future index similarly, with Block ID (row num in CSR) accordingly
         * Only works b/c we increment the open position within each row
         */
        CSCval[CSCptr[c]] = v;
        CSCind[CSCptr[c]] = low;
        ++CSCptr[c];

    } else return;
}

__global__ void func2(int * CSRind, int * CSCptr, int CSCrows, int nonzeros) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < CSCrows + 1) {
        CSCptr[k] = 0;
    }

    __syncthreads();

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < nonzeros) { 
        CSCptr[CSRind[j] + 1]++;
    }
}

int transpose() {};


int main() {};