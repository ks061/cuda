#include <stdio.h>
#include <string.h>

static const int NUM_BLOCKS = 64;
static const int NUM_THREADS = 1024; // per block	
static const int MAX_ARRAY_SIZE = NUM_BLOCKS * NUM_THREADS;

void square(double * h_out, int start, int end){
	for (int i = start; i < end; i++) {
		h_out[i - start] = ((double)i) * ((double)i);
	}
}

void print_err() {
	printf("Usage: ./square_user [number of squares] [optional flags: -q]\n\n");
	exit(0);
}

void exec_square(int start, int end, int N, bool quiet) {	
	int ARRAY_BYTES = (end - start) * sizeof(double);
	double * h_out;
	h_out = (double *)malloc(ARRAY_BYTES);

	square(h_out, start, end);

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

	free(h_out);
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
		exec_square(0, curr_N, N, quiet);
	}
	for (; curr_N < N; curr_N = curr_N + MAX_ARRAY_SIZE) {
		exec_square(curr_N, curr_N + MAX_ARRAY_SIZE, N, quiet);
	}

	return 0;
}
