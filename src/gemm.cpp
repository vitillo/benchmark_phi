// Use a separate compilation unit for a fair comparison vs OpenCL

#include "gemm.h"

void gemm(const scalar_t *A, const scalar_t *B, scalar_t *C, int M, int N, int K, int nmat){
  for(int i = 0; i < M; i++){
    for(int j = 0; j < N; j++){
      for(int k = 0; k < K; k++){
	const scalar_t *sub_A = A + i*K*nmat + k*nmat;
	const scalar_t *sub_B = B + k*N*nmat + j*nmat;
	scalar_t *sub_C = C + i*N*nmat + j*nmat;

#pragma unroll(4)
#pragma vector aligned
#pragma simd assert
	for(int idx = 0; idx < nmat; idx++){
	  C[idx] += A[idx]*B[idx];
	}
      }
    }
  }
}
