#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <cuda_runtime.h>

__global__ void Max_Interleaved_Addressing_Shared(float* data, float* data_copy, int* maxidx, int data_size){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float sdata[7];
    if (idx < data_size){

        /*copy to shared memory*/
        sdata[threadIdx.x] = data[idx];
        __syncthreads();

        for(int stride=1; stride < blockDim.x; stride *= 2) {
            if (threadIdx.x % (2*stride) == 0) {
                float lhs = sdata[threadIdx.x];
                float rhs = sdata[threadIdx.x + stride];
                sdata[threadIdx.x] = lhs < rhs ? rhs : lhs;
            }
            __syncthreads();
        }
    }
    if (idx == 0) data[0] = sdata[0];
    if (data_copy[idx] == sdata[0]) maxidx[0] = idx;
}

/*
__global__ void find_maximum_kernel(float *angle, float *max, int *mutex, unsigned int n, int *ans)
{
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride = gridDim.x*blockDim.x;
	unsigned int offset = 0;

	__shared__ float cache[2];

	float temp = -1.0;
	while(index + offset < n){
		temp = fmaxf(temp, angle[index + offset]);
		offset += stride;
	}

	cache[threadIdx.x] = temp;

	__syncthreads();

	// reduction
	unsigned int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
			cache[threadIdx.x] = fmaxf(cache[threadIdx.x], cache[threadIdx.x + i]);
		}

		__syncthreads();
		i /= 2;
	}

	if(threadIdx.x == 0){
		while(atomicCAS(mutex,0,1) != 0);  //lock
		*max = fmaxf(*max, cache[0]);
		atomicExch(mutex, 0);  //unlock
	}

	for (i=0;i<7;i++) {
		if (*max == angle[i]) {
			*ans = i;
		}
	}

  // *max 為最大值結果
}
*/

extern "C" float test(float *angle) {
    // set matrix dimension
    int nx = 7;
    int ny = 1;
    int nxy = nx*ny;
    float nBytes = nxy * sizeof(float);
    int NBytes = sizeof(int);

    // malloc host memory
    float *h_A;
    float *h_B;
    int *h_C;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_C = (int *)malloc(NBytes);

    // malloc device memory
    float *d_MatA;
    float *d_MatB;
    int *d_MatC;
    cudaMalloc((float **)&d_MatA, nBytes);
    cudaMalloc((float **)&d_MatB, nBytes);
    cudaMalloc((int **)&d_MatC, NBytes);

    // transfer data from host to device
    cudaMemcpy(d_MatA, angle, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, angle, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatC, h_C, NBytes, cudaMemcpyHostToDevice);

    Max_Interleaved_Addressing_Shared <<< 1, 7>>> (d_MatA, d_MatB, d_MatC, nx);

    // copy kernel result back to host side
    cudaMemcpy(h_B, d_MatA, nBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C, d_MatC, NBytes, cudaMemcpyDeviceToHost);
    
    // free host and devide memory
    cudaFree(d_MatA);
    cudaFree(d_MatB);
    cudaFree(d_MatC);
    free(h_A);
    free(h_B);
    return (h_C[0]);
    
/*
    unsigned int N = 7;
    //float *h_angle;
    float *d_angle;
    float *h_max;
    float *d_max;
    int *d_mutex;
    int *d_ans;
    int *h_ans; 
    // allocate memory
    //h_angle = (float*)malloc(N*sizeof(float));
    h_max = (float*)malloc(sizeof(float));
    h_ans = (int*)malloc(sizeof(int));
    cudaMalloc((void**)&d_angle, N*sizeof(float));
    cudaMalloc((void**)&d_max, sizeof(float));
    cudaMalloc((void**)&d_mutex, sizeof(int));
    cudaMalloc((void**)&d_ans, sizeof(int));
    cudaMemset(d_max, 0, sizeof(float));
    cudaMemset(d_mutex, 0, sizeof(float)); 
    cudaMemset(d_ans, 0,sizeof(int));

    // copy from host to device
    cudaMemcpy(d_angle, h_angle, N*sizeof(float), cudaMemcpyHostToDevice);

    // call kernel
    dim3 gridSize = 2;
    dim3 blockSize = 2;
    find_maximum_kernel<<< gridSize, blockSize >>>(d_angle, d_max, d_mutex, N, d_ans);

    // copy from device to host
    cudaMemcpy(h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ans, d_ans, sizeof(int), cudaMemcpyDeviceToHost);

    // free memory
    //free(h_angle);
    //free(h_max);
    //free(h_ans);
    cudaFree(d_angle);
    cudaFree(d_max);
    cudaFree(d_mutex);

    cudaFree(d_ans);

    printf("\n\n this message is from *.cu file !!! ans = %d \n\n",*h_ans);
    return(*h_ans);
*/
}

