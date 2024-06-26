#pragma once

#include <omp.h>

struct matrix {
	float* data;
	int row_num;
	int columns_num;
};

void print_mat(matrix* mat) {
	for (int i = 0; i < mat->columns_num; i++) {
		for (int j = 0; j < mat->row_num; j++) {
			std::cout << mat->data[i * mat->row_num + j] << " ";
		}
		std::cout << "\n";
	}
}

matrix init_mat(float3 data, int N, int M) {
	matrix matrix;
	matrix.data = (float*)calloc(N * M, sizeof(float));
	matrix.row_num = N;
	matrix.columns_num = M;

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			matrix.data[i * M + j] = data[i * M + j];
		}
	}

	return matrix;
}

matrix transpose_mat(matrix* mat) {
	int N = mat->row_num;
	int M = mat->columns_num;

	matrix transposed;
	transposed.data = (float*)calloc(N * M, sizeof(float));
	transposed.row_num = M;
	transposed.columns_num = N;

	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			transposed.data[i * N + j] = mat->data[j * M + i];
		}
	}

	return transposed;
}

void multiply_mat(matrix* mat1, matrix* mat2, matrix* dest, int A_offset) {
	if (mat1->columns_num != mat2->row_num) { std::cout << "can't multiply these matrixes"; exit(1); }
	int N = mat1->row_num;
	int M = mat1->columns_num;
	int K = mat2->columns_num;
	matrix transposed = transpose_mat(mat2);

	matrix* mat = new matrix();
	mat->data = (float*)calloc(N * K, sizeof(float));
	mat->row_num = N;
	mat->columns_num = K;

#pragma omp parallel
	{
		#pragma omp for
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < K; j++) {
				int dest = i * K + j;
				int src1 = i * M;
				int src2 = j * M;
				for (int s = 0; s < M; s++) {
					mat->data[dest] += mat1->data[src1 + s + A_offset] * transposed.data[src2 + s];
				}
			}
		}
	}
	dest->data = (float*)realloc(dest->data, N * K * sizeof(float));
	memcpy(dest->data, mat->data, N * K * sizeof(float));
	dest->columns_num = mat->columns_num;
	dest->row_num = mat->row_num;
	free(mat->data);
	delete mat;
}

matrix add_mat(matrix* mat1, matrix* mat2, int B_offset) {
	int N = mat1->row_num;
	int M = mat1->columns_num;

	if(M != mat2->columns_num || N != mat2->row_num) { 
		std::cout << "can't add these matrix"; 
		exit(1); 
	}

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			mat1->data[i * M + j] += mat2->data[i * M + j + B_offset];
		}
	}

	return *mat1;
}

void sin_mat(matrix* mat, int N, int M) {
	const float w = 30.0;

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			mat->data[i * M + j] = sin(w * mat->data[i * M + j]);
		}
	}
}
