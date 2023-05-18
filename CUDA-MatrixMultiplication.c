#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>


#define A_ROWS 8192
#define A_COLS 2048
#define B_ROWS 2048
#define B_COLS 8192
#define C_ROWS A_ROWS
#define C_COLS B_COLS

#define TILE_WIDTH 32

__global__ void kernelMatrixMultiplication(int* A, int* B, int* C) {
	// Kernel for Matrix Multiplication
	__shared__ int ds_M[TILE_WIDTH][TILE_WIDTH];
	__shared__ int ds_N[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;
	int Row = by * blockDim.y + ty;
	int Col = bx * blockDim.x + tx;
	int Pvalue = 0;

	for (int p = 0; p < A_COLS / TILE_WIDTH; ++p) {
		ds_M[ty][tx] = A[Row * A_COLS + p * TILE_WIDTH + tx];
		ds_N[ty][tx] = B[(p * TILE_WIDTH + ty) * B_COLS + Col];
		__syncthreads();
		for (int i = 0; i < TILE_WIDTH; ++i) {
			Pvalue += ds_M[ty][i] * ds_N[i][tx];
		}
		__syncthreads();
	}
	C[Row * C_COLS + Col] = Pvalue;
}


int main() {
	printf("hey");
	// Initial matrix compatibility check
	if (A_COLS != B_ROWS) {
		exit(1);
	}

	// Allocate matrices
	int* A_h;
	int* B_h;
	int* C_h;

	A_h = (int*)malloc(A_ROWS * A_COLS * sizeof(int));
	B_h = (int*)malloc(B_ROWS * B_COLS * sizeof(int));
	C_h = (int*)malloc(C_ROWS * C_COLS * sizeof(int));

	if (A_h == NULL || B_h == NULL || C_h == NULL) {
		exit(1);
	}

	// Fill matrices with data
	for (int i = 0; i < A_ROWS; i++) {
		for (int j = 0; j < A_COLS; j++) {
			int index = i * A_ROWS + j;
			A_h[index] = 1;
		}
	}

	for (int i = 0; i < B_ROWS; i++) {
		for (int j = 0; j < B_COLS; j++) {
			int index = i * B_ROWS + j;
			B_h[index] = 1;
		}
	}

	/*for (int i = 0; i < C_ROWS; i++) {
		for (int j = 0; j < C_COLS; j++) {
			int index = i * C_ROWS + j;
			C_h[index] = 0;
		}
	}*/

	// Copy data to device
	int* A_d;
	int* B_d;
	int* C_d;

	cudaMalloc((void**)&A_d, A_ROWS * A_COLS * sizeof(int));
	cudaMalloc((void**)&B_d, B_ROWS * B_COLS * sizeof(int));
	cudaMalloc((void**)&C_d, C_ROWS * C_COLS * sizeof(int));

	cudaMemcpy(A_d, A_h, A_ROWS * A_COLS * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, B_ROWS * B_COLS * sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(C_d, C_h, C_ROWS * C_COLS * sizeof(int));

	// Define grid and invoke kernel function
	dim3 blockDim(16, 16);
	dim3 gridDim(C_COLS / 16, C_ROWS / 16);

	kernelMatrixMultiplication << <gridDim, blockDim >> > (A_d, B_d, C_d);

	// Copy result to host
	cudaMemcpy(C_h, C_d, C_ROWS * C_COLS * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < C_ROWS; i++) {
		for (int j = 0; j < C_COLS; j++) {
			int index = i * C_ROWS + j;
			printf("%d ", C_h[i]);
		}
	}

	// Free memories
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);

	free(A_h);
	free(B_h);
	free(C_h);

	return 0;
}