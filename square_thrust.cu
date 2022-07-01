#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

static const int NUM_BLOCKS = 64;
static const int NUM_THREADS = 1024; // per block	
static const int MAX_ARRAY_SIZE = NUM_BLOCKS * NUM_THREADS;

void print_err() {
	printf("Usage: ./square_user [number of squares] [optional flags: -q]\n\n");
	exit(0);
}


void exec_kernel(int start, int end, int N, bool quiet) {	
	int ARRAY_SIZE = end - start;
//	double * d_out;
//	double * h_out;
	
	// allocate GPU memory
//	cudaMalloc((void**) &d_out, ARRAY_BYTES);
//	h_out = (double *)malloc(ARRAY_BYTES);
	thrust::device_vector<double> d_in(ARRAY_SIZE);	
	thrust::device_vector<double> d_out(ARRAY_SIZE);
	thrust::host_vector<double> h_out(ARRAY_SIZE);

	thrust::sequence(d_in.begin(), d_in.end(), start);
	thrust::transform(d_in.begin(), d_in.end(), d_out.begin(), thrust::square<double>());

	cudaError err;
	if ( cudaSuccess != (err = cudaGetLastError()) ){
		fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString( err ) );
		exit(-2);
	}

	// copy back the result array to the CPU
	// cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	thrust::copy(d_out.begin(), d_out.end(), h_out.begin());

	// print out the resulting array
	for (int i = start; i < end; i++) {
		if (!quiet) {
			printf("%lf\n\n", h_out[i - start]);
		}
	//	if (i>510000) {printf("%d\n", i);}
		if (quiet && (i >= (N-4))) {
			printf("%lf\n\n", h_out[i - start]);
		}
	}
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

	int N = atoi(argv[1]);
	int curr_N;
	
	curr_N = N % MAX_ARRAY_SIZE;
	if (curr_N != 0) {
		exec_kernel(0, curr_N, N, quiet);
	}
	for (; curr_N < N; curr_N = curr_N + MAX_ARRAY_SIZE) {
		exec_kernel(curr_N, curr_N + MAX_ARRAY_SIZE, N, quiet);
	}

	return 0;
}
