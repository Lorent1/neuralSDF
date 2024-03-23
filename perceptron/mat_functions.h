#pragma once

#include <omp.h>

void print_mat(float* mat, int N, int M) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			std::cout << mat[i * M + j] << " ";
		}
		std::cout << "\n";
	}
}

std::vector<float> init_mat(std::vector<float> data, int N, int M) {
	std::vector<float> mat(N * M);

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			mat[i * M + j] = data[i * M + j];
		}
	}

	return mat;
}

void transpose_mat(float* mat, int N, int M) {
	std::vector<float> new_mat(M * N);

	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			new_mat[i * N + j] = mat[j * M + i];
		}
	}

	mat = new_mat.data();
}

std::vector<float> multiply_mat(float* mat1, float* mat2, int N, int M, int K, int A_offset) {
	transpose_mat(mat2, M, K);

	std::vector<float> res(N * K);

#pragma omp parallel
	{
		#pragma omp for collapse(2)
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < K; j++) {
				int dest = i * K + j;
				res[dest] = 0;
				int src1 = i * M;
				int src2 = j * M;
				for (int s = 0; s < M; s++) {
					res[dest] += mat1[src1 + s + A_offset] * mat2[src2 + s];
				}
			}
		}
	}

	return res;
}

std::vector<float> add_mat(float* mat1, float* mat2, int N, int M, int B_offset) {
	std::vector<float> res(N * M);

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			res[i * M + j] = mat1[i * M + j] + mat2[i * M + j + B_offset];
		}
	}

	return res;
}

void sin_mat(std::vector<float>* mat, int N, int M) {
	const float w = 30;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			(*mat)[i * M + j] = sin(w * (*mat)[i * M + j]);
		}
	}
}
