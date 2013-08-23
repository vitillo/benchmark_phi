#include "include/common.h"

void mxm(__global scalar_t *A, __global scalar_t *B, __global scalar_t *C, int M, int N, int K, int num, int idx){
  for(int i = 0; i < M; i++){
    for(int j = 0; j < N; j++){
      scalar_t sum = 0;

      for(int k = 0; k < K; k++){
	sum += A[i*K*num + k*num + idx] * B[k*N*num + j*num + idx];
      }

      C[i*N*num + j*num + idx] = sum;
    }
  }
}

__kernel void gemm(__global scalar_t *A, __global scalar_t *B, __global scalar_t *C, int M, int N, int K, int num, int iters){
  const int idx = get_global_id(0);

  for(int i = 0; i < iters; i++){
    mxm(A, B, C, M, N, K, num, idx);
  }
}

