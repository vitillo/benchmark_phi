#include <iostream>
#include <omp.h>
#include <getopt.h>
#include <cstdlib>
#include <immintrin.h>
#include <cassert>

#if USE_DOUBLE
typedef double scalar_t;
#else
typedef float scalar_t;
#endif

using namespace std;

static const double frequency = 1.0e-9;
static const int flops_per_calc = 2;
static const int dim = 5;
static const int nmat = 4992;
static const int iters = 100000;
static const int block_size = 32;

scalar_t A[nmat*dim*dim] __attribute__((aligned(ALIGNMENT)));
scalar_t B[nmat*dim*dim] __attribute__((aligned(ALIGNMENT)));
scalar_t C[nmat*dim*dim] __attribute__((aligned(ALIGNMENT)));

template <int M, int N, int K>
void gemm(const scalar_t *A, const scalar_t *B, scalar_t *C, int nmat, int base, int block_size){
  for(int i = 0; i < M; i++){
    for(int j = 0; j < N; j++){
      for(int k = 0; k < K; k++){
	const scalar_t *sub_A = A + base + i*K*nmat + k*nmat;
	const scalar_t *sub_B = B + base + k*N*nmat + j*nmat;
	scalar_t *sub_C = C + base + i*N*nmat + j*nmat;

#pragma unroll(4)
#pragma vector aligned
#pragma simd assert
	for(int idx = 0; idx < block_size; idx++){
	  C[idx] += A[idx]*B[idx];
	}
      }
    }
  }
}

void benchmark(){
  double gflop;
  double t0;

  cout << "nmatrices: " << nmat << " block_size: " << block_size << " dimension: " << dim << " iterations:" << iters << endl;
  gflop = frequency * flops_per_calc * iters * dim * dim * dim * nmat;

  assert(nmat % block_size == 0); // Ideal use case

#pragma omp parallel
#pragma omp master
  t0 = omp_get_wtime();

  for(int i = 0; i < iters; i++){
#pragma omp parallel for
    for(int z = 0; z < nmat; z+= block_size){
      gemm<dim, dim, dim>(A, B, C, nmat, z, block_size);
    }
  }

  double t = omp_get_wtime() - t0;
  cout << "GFlop = " << gflop << ", Secs = " << t <<", GFlops per sec = " << gflop/t << endl;
}

int main(int argc, char *argv[]){
  benchmark();
  return 0;
}
