#include <stdio.h>

#define TILE_SIZE 16

__global__ void mysgemm(int m, int n, int p, const float *A, const float *B, float *C) {
    __shared__ float h[TILE_SIZE][TILE_SIZE], v[TILE_SIZE][TILE_SIZE];

    int carry = int(p / TILE_SIZE);
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float ts = 0;

    for (int pos = 0; pos <= carry; pos++) {
        if ((row < m) && (pos * TILE_SIZE + threadIdx.x < p)) {
            h[threadIdx.y][threadIdx.x] = A[row * p + pos * TILE_SIZE + threadIdx.x];
        } else {
            h[threadIdx.y][threadIdx.x] = 0;
        }

        if ((col < n) && (pos * TILE_SIZE + threadIdx.y < p)) {
            v[threadIdx.y][threadIdx.x] = B[(pos * TILE_SIZE + threadIdx.y) * n + col];
        } else {
            v[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        if (row < m && col < n) {
            for (int idx = 0; idx < TILE_SIZE; idx++) {
                ts += h[threadIdx.y][idx] * v[idx][threadIdx.x];
            }
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * n + col] += ts;
    }
}

void basicSgemm(int m, int n, int k, const float *A, const float *B, float *C) {
    const unsigned int BLOCK_SIZE = TILE_SIZE;
    dim3 DimGrid((n - 1) / BLOCK_SIZE + 1, (m - 1) / BLOCK_SIZE + 1, 1);
    dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    mysgemm<<<DimGrid, DimBlock>>>(m, n, k, A, B, C);
}
