// Use a separate compilation unit for a fair comparison vs OpenCL

#include "gemm.h"

void gemm(const scalar_t *A, const scalar_t *B, scalar_t *C, int M, int N, int K, int nmat, int base, int block_size){
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
