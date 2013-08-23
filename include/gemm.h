#ifndef GEMM_H
#define GEMM_H

#include "common.h"

void gemm(const scalar_t *A, const scalar_t *B, scalar_t *C, int M, int N, int K, int nmat, int base, int block_size);

template <int M, int N, int K>
void gemm_fast(const scalar_t *A, const scalar_t *B, scalar_t *C, int nmat, int base, int block_size){
  for(int i = 0; i < M; i++){
    for(int j = 0; j < N; j++){
      for(int k = 0; k < K; k++){
	const scalar_t *sub_A = A + base + i*K*nmat + k*nmat;
	const scalar_t *sub_B = B + base + k*N*nmat + j*nmat;
	scalar_t *sub_C = C + base + i*N*nmat + j*nmat;

#pragma unroll(4)
#pragma vector unaligned
#pragma simd assert
	for(int idx = 0; idx < block_size; idx++){
	  C[idx] += A[idx]*B[idx];
	}
      }
    }
  }
}



#endif
