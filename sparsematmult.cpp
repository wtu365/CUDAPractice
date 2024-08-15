#include <iostream>
#include <omp.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <cstring>      /* strcasecmp */
#include <cstdint>
#include <assert.h>
#include <vector>       // std::vector
#include <algorithm>    // std::random_shuffle
#include <random>
#include <stdexcept>

using namespace std;

using idx_t = std::uint32_t;
using val_t = float;
using ptr_t = std::uintptr_t;

/**
 * CSR structure to store search results
 */
typedef struct csr_t {
  idx_t nrows; // number of rows
  idx_t ncols; // number of rows
  idx_t * ind; // column ids
  val_t * val; // values
  ptr_t * ptr; // pointers (start of row in ind/val)

  csr_t()
  {
    nrows = ncols = 0;
    ind = nullptr;
    val = nullptr;
    ptr = nullptr;
  }

  /**
   * Reserve space for more rows or non-zeros. Structure may only grow, not shrink.
   * @param nrows Number of rows
   * @param nnz   Number of non-zeros
   */
  void reserve(const idx_t nrows, const ptr_t nnz)
  {
    if(nrows > this->nrows){
      if(ptr){
        ptr = (ptr_t*) realloc(ptr, sizeof(ptr_t) * (nrows+1));
      } else {
        ptr = (ptr_t*) malloc(sizeof(ptr_t) * (nrows+1));
        ptr[0] = 0;
      }
      if(!ptr){
        throw std::runtime_error("Could not allocate ptr array.");
      }
    }
    if(nnz > ptr[this->nrows]){
      if(ind){
        ind = (idx_t*) realloc(ind, sizeof(idx_t) * nnz);
      } else {
        ind = (idx_t*) malloc(sizeof(idx_t) * nnz);
      }
      if(!ind){
        throw std::runtime_error("Could not allocate ind array.");
      }
      if(val){
        val = (val_t*) realloc(val, sizeof(val_t) * nnz);
      } else {
        val = (val_t*) malloc(sizeof(val_t) * nnz);
      }
      if(!val){
        throw std::runtime_error("Could not allocate val array.");
      }
    }
    this->nrows = nrows;
  }

  /** 
   * Transpose matrix
   */
  csr_t* transpose()
  {
    auto mat = new csr_t();
    mat->reserve(this->ncols, this->ptr[this->nrows]);
    mat->ncols = this->nrows;

    // printf("%d\n", mat->nrows);
    #pragma omp parallel
    {
      //#pragma omp single
      {
        #pragma omp for
        for(idx_t k = 0; k < mat->nrows + 1; ++k) mat->ptr[k] = 0;

        // #pragma omp single
        // {
        //   for(idx_t k = 0; k < mat->nrows + 1; ++k){
        //     printf("%lu\n", mat->ptr[k]);
        //     printf("%d\n", k);
        //   } 
        // }

        #pragma omp single
        {
          for(idx_t j = 0; j < this->ptr[this->nrows]; ++j){
            mat->ptr[this->ind[j] + 1]++;
            // printf("%d\n", this->ind[j] + 1);
          }
        }

        // #pragma omp single
        // {
        //   for(idx_t k = 0; k < mat->nrows + 1; ++k){
        //     printf("%lu\n", mat->ptr[k]);
        //     printf("%d\n", k);
        //   } 
        // }

        #pragma omp single
        {
          for(idx_t j = 1; j < mat->nrows + 1; ++j){
            mat->ptr[j] += mat->ptr[j - 1];
          }
        }
        // #pragma omp single
        // {
        //   for(idx_t k = 0; k < mat->nrows + 1; ++k){
        //     printf("%lu\n", this->ptr[k]);
        //     printf("%d\n", k);
        //   } 
        // }
      }
    }
    ptr_t * temp_ptr = (ptr_t*) malloc(sizeof(ptr_t) * (mat->nrows+1));
    #pragma omp parallel
    {
      {
        #pragma omp for
        for(idx_t k = 0; k < mat->nrows + 1; ++k) temp_ptr[k] = 0;

        #pragma omp for
        for(idx_t k = 0; k < mat->nrows + 1; ++k) temp_ptr[k] = mat->ptr[k];

        // #pragma omp single
        // {
        //   for(idx_t k = 0; k < mat->nrows + 1; ++k){
        //     printf("%lu\n", temp_ptr[k]);
        //     printf("%d\n", k);
        //   } 
        // }
        #pragma omp single
        {
          idx_t v,c = 0;
          for(idx_t i = 0; i < this->nrows; ++i){
            for(idx_t j = this->ptr[i]; j < this->ptr[i + 1]; ++j){
              v = this->val[j];
              c = this->ind[j];
              // printf("%d\n", c);
              // printf("%lu\n", temp_ptr[c]);
              mat->ind[temp_ptr[c]] = i;
              mat->val[temp_ptr[c]] = v;
              temp_ptr[c]++;
            }
          }
        }
      }
    }

    return mat;
  }

  /**
   * Reserve space for more rows or non-zeros. Structure may only grow, not shrink.
   * @param nrows Number of rows
   * @param ncols Number of columns
   * @param factor   Sparsity factor
   */
  static csr_t * random(const idx_t nrows, const idx_t ncols, const double factor)
  {
    ptr_t nnz = (ptr_t) (factor * nrows * ncols);
    if(nnz >= nrows * ncols / 2.0){
      throw std::runtime_error("Asking for too many non-zeros. Matrix is not sparse.");
    }
    auto mat = new csr_t();
    mat->reserve(nrows, nnz);
    mat->ncols = ncols;

    /* fill in ptr array; generate random row sizes */
    unsigned int seed = (unsigned long) mat;
    long double sum = 0;
    for(idx_t i=1; i <= mat->nrows; ++i){
      mat->ptr[i] = rand_r(&seed) % ncols;
      sum += mat->ptr[i];
    }
    for(idx_t i=0; i < mat->nrows; ++i){
      double percent = mat->ptr[i+1] / sum;
      mat->ptr[i+1] = mat->ptr[i] + (ptr_t)(percent * nnz);
      if(mat->ptr[i+1] > nnz){
        mat->ptr[i+1] = nnz;
      }
    }
    if(nnz - mat->ptr[mat->nrows-1] <= ncols){
      mat->ptr[mat->nrows] = nnz;
    }

    /* fill in indices and values with random numbers */
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      unsigned int seed = (unsigned long) mat * (1+tid);
      std::vector<int> perm;
      for(idx_t i=0; i < ncols; ++i){
        perm.push_back(i);
      }
      std::random_device seeder;
      std::mt19937 engine(seeder());

      #pragma omp for
      for(idx_t i=0; i < nrows; ++i){
        std::shuffle(perm.begin(), perm.end(), engine);
        for(ptr_t j=mat->ptr[i]; j < mat->ptr[i+1]; ++j){
          mat->ind[j] = perm[j - mat->ptr[i]];
          mat->val[j] = ((double) rand_r(&seed)/rand_r(&seed));
        }
      }
    }

    return mat;
  }

  string info(const string name="") const
  {
    return (name.empty() ? "CSR" : name) + "<" + to_string(nrows) + ", " + to_string(ncols) + ", " +
      (ptr ? to_string(ptr[nrows]) : "0") + ">";
  }

  ~csr_t()
  {
    if(ind){
      free(ind);
    }
    if(val){
      free(val);
    }
    if(ptr){
      free(ptr);
    }
  }
} csr_t;

/**
 * Ensure the matrix is valid
 * @param mat Matrix to test
 */
// void test_matrix(csr_t * mat){
//   auto nrows = mat->nrows;
//   auto ncols = mat->ncols;
//   assert(mat->ptr);
//   auto nnz = mat->ptr[nrows];
//   for(idx_t i=0; i < nrows; ++i){
//     // printf("%lu\n", mat->ptr[i]);
//     // printf("%lu\n", nnz);
//     assert(mat->ptr[i] <= nnz);
//   }
//   for(ptr_t j=0; j < nnz; ++j){
//     assert(mat->ind[j] < ncols);
//   }
// }

val_t sparsevectorvector(csr_t * A, csr_t * B, ptr_t arow, ptr_t brow){
  val_t sum = 0;
  // printf("Entered function\n");
  // #pragma omp parallel
  // {
    idx_t i = 0;
    idx_t j = 0;
    // #pragma omp for
    for(i = A->ptr[arow], j = B->ptr[brow]; i < A->ptr[arow + 1] && j < B->ptr[brow + 1];){
      // printf("index i is: %d and index j is: %d\n", i, j);
      if (A->ind[i] == B->ind[j]){
        sum += A->val[i] * B->val[j];
        ++i; ++j;
      } else if (A->ind[i] < B->ind[j]){
        ++i;
      } else ++j;
    }
  // }

  // for(idx_t i = A->ptr[arow]; i < A->ptr[arow + 1];){
  //   for(idx_t j = B->ptr[brow]; j < B->ptr[brow + 1];){
  //     printf("index i is: %d and index j is: %d\n", i, j);
  //     if (A->ind[i] == B->ind[j]){
  //       sum += A->val[i] * B->val[j];
  //       ++i; ++j;
  //     } else if (A->ind[i] < B->ind[j]){
  //       ++i;
  //     } else ++j;
  //   }
  // }
  // printf("Exiting\n");
  return sum;
}

/**
 * Multiply A and B and write output in C.
 * Note that C has no data allocations (i.e., ptr, ind, and val pointers are null).
 * Use `csr_t::reserve` to increase C's allocations as necessary.
 * @param A  Matrix A.
 * @param B The transpose of matrix B.
 * @param C  Output matrix
 */
void sparsematmult(csr_t * A, csr_t * B, csr_t *C)
{
  // C->reserve(A->nrows, A->nrows * B->nrows);
  // C->ncols = B->nrows;

  // C->ptr[0] = 0;                                                        //init ptr[0] = 0
  // C->ptr[1] = 0;

  // idx_t countCval = 0;                                                            //counting current cvalues

  // for(idx_t rowA = 0; rowA < A->nrows; ++rowA){                                   //iterate through rows of A
  //   for(idx_t rowB = 0; rowB < B->nrows; ++rowB){                                 //iterating through rows of B
  //     bool addToPTR = false;                                                      //whether or not there is a new value
  //     for(idx_t Bindex = B->ptr[rowB]; Bindex < B->ptr[rowB + 1]; ++Bindex){      //iterating through every value in row B
  //       for(idx_t Aindex = A->ptr[rowA]; Aindex < A->ptr[rowA + 1]; ++Aindex){    //iterating through every value in row A
  //         if(A->ind[Aindex] == B->ind[Bindex]){                                   //if the column ids match up, multiply them
  //           C->val[countCval] += A->val[Aindex] * B->val[Bindex];                 //part of dot product
  //           C->ind[countCval] = rowB;                                             //setting the column of the incremented value
  //           addToPTR = true;                                                      //Yes, there will be a value
  //         }
  //       }
  //     }
  //     if(addToPTR){                                                               //if there is a non-zero value
  //       ++countCval;                                                              //prep for the next value slot
  //     }
  //     addToPTR = false;                                                           //reset
  //   }
  //   C->ptr[rowA + 1] = countCval;                                                 //ptr is current amount of value available
  // }

  C->reserve(A->nrows, A->nrows * B->nrows);
  C->ncols = B->nrows;

  C->ptr[0] = 0;
  C->ptr[1] = 0;

  val_t temp = 0;
  idx_t currentVal = 0;
  #pragma omp parallel
  {
    #pragma omp for
    for(idx_t rowA = 0; rowA < A->nrows; ++rowA){
      for(idx_t rowB = 0; rowB < B->nrows; ++rowB){
        // printf("Row A is: %d and Row B is: %d\n", rowA, rowB);
        temp = sparsevectorvector(A, B, rowA, rowB);
        if (temp != 0){
          C->val[currentVal] = temp;
          C->ind[currentVal] = rowB;
          ++currentVal;
          temp = 0;
        }
      }
      C->ptr[rowA + 1] = (ptr_t) currentVal;

    }  
  }
  // #pragma omp single
  // {
  //   for (idx_t i = 0; i < C->nrows; ++i){
  //     printf("%lu\n", C->ptr[i]);
  //   }
  // }
}



int main(int argc, char *argv[])
{
  // if(argc < 4){
  //   cerr << "Invalid options." << endl << "<program> <A_nrows> <A_ncols> <B_ncols> <fill_factor> [-t <num_threads>]" << endl;
  //   exit(1);
  // }
    int nrows = 100;
    int ncols = 10000;
    int ncols2 = 10000;
    double factor = .2;
  if(argc > 4){
    nrows = atoi(argv[1]);
    ncols = atoi(argv[2]);
    ncols2 = atoi(argv[3]);
    factor = atof(argv[4]);
  }
  int nthreads = 16;
  if(argc == 7 && strcasecmp(argv[5], "-t") == 0){
    nthreads = atoi(argv[6]);
    omp_set_num_threads(nthreads);
  }
  cout << "A_nrows: " << nrows << endl;
  cout << "A_ncols: " << ncols << endl;
  cout << "B_nrows: " << ncols << endl;
  cout << "B_ncols: " << ncols2 << endl;
  cout << "factor: " << factor << endl;
  cout << "nthreads: " << nthreads << endl;

  /* initialize random seed: */
  srand (time(NULL));

  auto A = csr_t::random(nrows, ncols, factor);
  auto B = csr_t::random(ncols, ncols2, factor); // Note B is not transposed yet.
  // test_matrix(A);
  // test_matrix(B);
  auto C = new csr_t(); // Note that C has no data allocations so far.

  // cout << A->info("A") << endl;
  // cout << B->info("B") << endl;

  auto t1 = omp_get_wtime();
  /* Optionally transpose matrix B (Must implement) */
  // B->transpose();
  auto Btranspose = B->transpose();
  // test_matrix(Btranspose);
  cout << Btranspose->info("Btranspose") << endl;
  sparsematmult(A, Btranspose, C);

  auto t2 = omp_get_wtime();

  // test_matrix(C);
  cout << C->info("C") << endl;

  cout << "Execution time: " << (t2-t1) << endl;

  delete A;
  delete B;
  delete Btranspose;
  delete C;

  return 0;
}