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
    auto mat = new csr_t(); // The CSC that the CSR will be transposed into
    mat->reserve(this->ncols, this->ptr[this->nrows]); // Reserve space for the CSC
    mat->ncols = this->nrows;

    #pragma omp parallel
    {
      {
        #pragma omp for
        for(idx_t k = 0; k < mat->nrows + 1; ++k) mat->ptr[k] = 0; // Set each value in colptr to 0

        #pragma omp single
        {
          for(idx_t j = 0; j < this->ptr[this->nrows]; ++j){ //For each nonzero...
            mat->ptr[this->ind[j] + 1]++; // ...check which column that nonzero is in. The corresponding column in the CSC should be incremented by 1 for # of nonzeros in that column
          } // Maybe possible race condition?
        }

        #pragma omp single
        {
          for(idx_t j = 1; j < mat->nrows + 1; ++j){
            mat->ptr[j] += mat->ptr[j - 1]; // Each value in the colptr array should be added up to show the number of nonzeros accounted for reading column by column
          }
        }
      }
    }
    ptr_t * temp_ptr = (ptr_t*) malloc(sizeof(ptr_t) * (mat->nrows+1)); //Set up temporary colptr
    #pragma omp parallel
    {
      {
        #pragma omp for
        for(idx_t k = 0; k < mat->nrows + 1; ++k) temp_ptr[k] = 0; // Copying over colptr to temporary colptr

        #pragma omp for
        for(idx_t k = 0; k < mat->nrows + 1; ++k) temp_ptr[k] = mat->ptr[k];

        #pragma omp single
        {
          idx_t v,c = 0;
          for(idx_t i = 0; i < this->nrows; ++i){ // For each row in the CSR...
            for(idx_t j = this->ptr[i]; j < this->ptr[i + 1]; ++j){ // ...and for each nonzero in each row
              v = this->val[j]; // Get the value of the nonzero
              c = this->ind[j]; // Get the column index of that nonzero
              mat->ind[temp_ptr[c]] = i; // Set the index in colind to be the row number
              mat->val[temp_ptr[c]] = v; // Set the value in colval to be the value of the nonzero
              temp_ptr[c]++; // Increment the temporary colptr's column index by 1 to put the next nonzero in that column after the previously placed nonzero
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
  auto B = csr_t::random(ncols, ncols2, factor); // Note B is not transposed yet.
  
  auto Btranspose = B->transpose();
  cout << Btranspose->info("Btranspose") << endl;

  delete B;
  delete Btranspose;

  return 0;
}