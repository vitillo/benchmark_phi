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

scalar_t A[nmat*dim*dim] __attribute__((aligned(ALIGNMENT)));
scalar_t B[nmat*dim*dim] __attribute__((aligned(ALIGNMENT)));
scalar_t C[nmat*dim*dim] __attribute__((aligned(ALIGNMENT)));

void benchmark(){
  double gflop;
  double t0;

  cout << "nmatrices:" << nmat << " dimension:" << dim << " iterations:" << iters << endl;
  gflop = frequency * flops_per_calc * iters * dim * dim * dim * nmat;

// Compare 1 thread ICC vs 1 thread OpenCL + device fission
// I am only interested in the quality of the generated assembly
  t0 = omp_get_wtime();

  for(int i = 0; i < iters; i++){
    gemm(A, B, C, dim, dim, dim, nmat);
  }

  double t = omp_get_wtime() - t0;
  cout << "GFlop = " << gflop << ", Secs = " << t <<", GFlops per sec = " << gflop/t << endl;
}

int main(int argc, char *argv[]){
  benchmark();
  return 0;
}
