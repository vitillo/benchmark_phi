#include <iostream>
#include <omp.h>
#include <getopt.h>
#include <cstdlib>
#include <immintrin.h>
#include <cassert>

#include "common.h"
#include "gemm.h"

using namespace std;

static const double frequency = 1.0e-9;
static const int flops_per_calc = 2;
static const int dim = 5;
static const int nmat = 4992;
static const int iters = 1000;
static const int block_size = 16;

scalar_t A[nmat*dim*dim] __attribute__((aligned(ALIGNMENT)));
scalar_t B[nmat*dim*dim] __attribute__((aligned(ALIGNMENT)));
scalar_t C[nmat*dim*dim] __attribute__((aligned(ALIGNMENT)));

void benchmark(){
  double gflop;
  double t0;

  cout << "nmatrices:" << nmat << " block_size:" << block_size << " dimension:" << dim << " iterations:" << iters << endl;
  gflop = frequency * flops_per_calc * iters * dim * dim * dim * nmat;

  assert(nmat % block_size == 0); // Ideal use case

// Compare 1 thread ICC vs 1 thread OpenCL + device fission
// I am only interested in the quality of the generated assembly

//#pragma omp parallel
//#pragma omp master
  t0 = omp_get_wtime();

  for(int i = 0; i < iters; i++){
//#pragma omp parallel for
    for(int z = 0; z < nmat; z+= block_size){
      gemm(A, B, C, dim, dim, dim, nmat, z, block_size);
    }
  }

  double t = omp_get_wtime() - t0;
  cout << "GFlop = " << gflop << ", Secs = " << t <<", GFlops per sec = " << gflop/t << endl;
}

int main(int argc, char *argv[]){
  benchmark();
  return 0;
}
