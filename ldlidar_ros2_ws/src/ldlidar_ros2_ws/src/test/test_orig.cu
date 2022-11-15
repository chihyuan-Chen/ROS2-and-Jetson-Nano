
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <cuda_runtime.h>
//#include "test.cuh"

__global__ void sumArraysOnGPU(float *A, float *B, float *C) {
 int i = threadIdx.x;
 C[i] = A[i] + B[i] + 2.0;
}

void initialData(float *ip,int size) {
 // generate different seed for random number
 time_t t;
 srand((unsigned int) time(&t));
 for (int i=0; i<size; i++) {
   ip[i] = (float)( rand() & 0xFF )/10.0;
 }
}

void print_output(float *output, int size) {
	printf("\n Lab7 Output:\n");

	for(int idx=0; idx<size; idx++) {
		printf(" %.2f ",  output[idx]);
	}

}

//int main(int argc, char **argv) {
extern "C" int test() {
//int test() {
 // set up data size of vectors
 int nElem = 16;
 size_t nBytes = nElem * sizeof(float);

 float *h_A, *h_B, *h_C;
 float *d_A, *d_B, *d_C;

 // malloc host memory
 h_A = (float *)malloc(nBytes);
 h_B = (float *)malloc(nBytes);
 h_C = (float *)malloc(nBytes);
 
 // malloc device global memory
cudaMalloc((float**)&d_A, nBytes);
cudaMalloc((float**)&d_B, nBytes);
cudaMalloc((float**)&d_C, nBytes);

 // initialize data at host side
 initialData(h_A, nElem);
 initialData(h_B, nElem);

 // transfer data from host to device
cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

 // use GPU to calculate
 sumArraysOnGPU<<<1, nElem>>>(d_A, d_B, d_C);

 // copy kernel result back to host side
 cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost);
 
 // display the data 
 print_output(h_A, nElem);
 print_output(h_B, nElem);
 print_output(h_C, nElem);

 // free device global memory


cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);

 // free host memory
 free(h_A);
 free(h_B);
 free(h_C);
 printf("\n\n this message is from *.cu file !!! \n\n");
 return(0);
}

