
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<random>
#include<cassert>
#include<iostream>
#include"time.h"

using namespace std;


__global__ void matrixMul(int *a, int *b, int *c, int N) {
	
	//calcular o tamanho de cada thread
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < N && col < N)
	{
		int tmp = 0;

		for (int i = 0; i < N; i++)
		{
			tmp += a[row * N + i] * b[i * N + col];
		}

		c[row * N + col] = tmp;
	}
	
}


//inicializar as matrizes com numeros aletorios
void init_matrix(int* m, int N)
{
	for (int i = 0; i < N * N; i++)
	{
		m[i] = rand() % 100;
	}
}


//CPU
void cpu_result(int* a, int* b, int* c, int N)
{
	
	int tmp;
	for (int i = 0; i < N; i++)//row
	{
		for (int j = 0; j < N; j++)//cpl
		{
			tmp = 0;
			for (int k = 0; k < N; k++)//elemento no row-col
			{
				tmp += a[i * N + k] * b[k * N + j];
			}

			//check results
			//assert(tmp == c[i * N + j]);
			c[i * N + j] = tmp;
		}
	}
	
}
int main() {
	//tamanho da matriz
	int N = 1 << 10;
	size_t bytes = N * N * sizeof(int);

	Timer t1, t2;
	double timer1, timer2;
	//alocar memoria pras matrizes
	int* a, *b,  *c;
	cudaMallocManaged(&a, bytes);
	cudaMallocManaged(&b, bytes);
	cudaMallocManaged(&c, bytes);

	//inicializar as matrizes
	init_matrix(a, N);
	init_matrix(b, N);

	int threads = 16;
	int blocks = (N + threads - 1) / threads;

	//setup kernel parameters
	dim3 THREADS(threads, threads);
	dim3 BLOCKS(blocks, blocks);
	t1.reset();
	t1.start();
	matrixMul << <BLOCKS, THREADS >> > (a, b, c, N);
	cudaDeviceSynchronize();
	t1.finish();

	t2.reset();
	t2.start();
	cpu_result(a, b, c, N);
	t2.finish();

	cout << "PROGRAM COMPLETED" << endl;
	timer1 = t1.getElapsedTimeMs();
	timer2 = t2.getElapsedTimeMs();
	cout << "Timer GPU: " << timer1 << endl << "Timer CPU: " << timer2 << endl;
	return 0;

}