#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include <thrust/transform_scan.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

struct is_alphabet {
    __host__ __device__
    int operator()(const char c) const {
        return (c != ' ') ? 1 : 0;
    }
};

struct cont_acc {
    __host__ __device__
    int operator()(const int prev, const int curr) const {
        return (curr == 0) ? 0 : (curr + prev);
    }
};

char input[11] = { 'f', 'o', 'o', ' ', ' ', 'b', 'a', ' ', 'b', 'a', 'z' };
int output[11] = { 0 };

void CountPosition1(const char *text, int *pos, int text_size)
{
    for (int i = 0; i < 11; i++) {
        fprintf(stderr, "%d\t", output[i]);
    }
    fprintf(stderr, "\n");

    // count them
    thrust::transform_inclusive_scan(
        input,               // beginning of the input sequence
        input + 11,   // end of the input sequence
        output,                // beginning of the output sequence
        is_alphabet(),
        cont_acc()
    );

    for (int i = 0; i < 11; i++) {
        fprintf(stderr, "%d\t", output[i]);
    }
    fprintf(stderr, "\n");
}

void CountPosition2(const char *text, int *pos, int text_size)
{
}
