#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <iostream>
#include <fstream>

static const int NUM_BLOCKS = 64;
static const int NUM_THREADS = 1024; // per block	
static const int MAX_ARRAY_SIZE = NUM_BLOCKS * NUM_THREADS;

__global__ void square(double * d_out, int start){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double f = (double) (start + idx);
    d_out[idx] = f * f;
}

void print_err() {
	printf("Usage: ./square_user [number of squares] [optional flags: -q]\n\n");
	exit(0);
}

void exec_kernel(int start, int end, int N, bool quiet) {	
	int ARRAY_BYTES = (end - start) * sizeof(double);
	double * d_out;
	double * h_out;
	
	// allocate GPU memory
	cudaMalloc((void**) &d_out, ARRAY_BYTES);
	h_out = (double *)malloc(ARRAY_BYTES);

	// launch the kernel
	square<<<NUM_BLOCKS, NUM_THREADS>>>(d_out, start);

	cudaError err;
	if ( cudaSuccess != (err = cudaGetLastError()) ){
		fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString( err ) );
		exit(-2);
	}

	// copy back the result array to the CPU
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	// print out the resulting array
	/*
	for (int i = start; i < end; i++) {
		if (!quiet) {
			printf("%lf\n\n", h_out[i - start]);
		}
	//	if (i>510000) {printf("%d\n", i);}
		if (quiet && (i >= (N-4))) {
			printf("%lf\n\n", h_out[i - start]);
		}
	}
	*/

	cudaFree(d_out);
	free(h_out);
}

int num_digits(int num) {
	int num_digits = 0;
	while (num != 0) {
		num = num / 10;
		num_digits = num_digits + 1;
	}
	return num_digits;
}

int main(int argc, char ** argv) { 
	bool quiet = false;

	printf("\n");
	if (argc < 2) { 
		print_err(); 
	}
	if (argv[1] == NULL) { 
		print_err(); 
	} 
	if (argc > 2) {
		for (int i = 2; i < argc; i++) {
			if (!strcmp(argv[i], "-q")) {
				quiet = true;
			}
		}
	}

	int MAX_N = atoi(argv[1]);
	
	std::ofstream out_file;
	out_file.open("square_user.csv");

	clock_t start;
	clock_t end;
	int curr_N;
	int cpu_time_ms;
	char * str1;
	char * str2;

	for (int N = 0; N < MAX_N; N = N + (MAX_N / 100)) { 
		start = clock();
		
		curr_N = N % MAX_ARRAY_SIZE;
		if (curr_N != 0) {
			exec_kernel(0, curr_N, N, quiet);
		}
		for (; curr_N < N; curr_N = curr_N + MAX_ARRAY_SIZE) {
			exec_kernel(curr_N, curr_N + MAX_ARRAY_SIZE, N, quiet);
		}

		end = clock();
		cpu_time_ms = (int)( ((double)(end - start)) / CLOCKS_PER_SEC * 1000 ); // * 1000 for s --> ms 

		str1 = (char *) malloc(num_digits(N) * sizeof(char));
		str2 = (char *) malloc(num_digits(cpu_time_ms) * sizeof(char));
		sprintf(str1, "%d", N);
		sprintf(str2, "%d", cpu_time_ms);
		out_file << str1;
		out_file << ",";
		out_file <<  str2;
		out_file << "\n";
		free(str1);
		free(str2);
	}
	out_file.close();	
	return 0;
}
