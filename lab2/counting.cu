#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

struct is_alphabet {
    __host__ __device__
    int operator()(const char c) const {
        return (c != '\n') ? 1 : 0;
    }
};

void CountPosition1(const char *text, int *pos, int text_size)
{
    thrust::transform(
        thrust::device,
        text,
        text + text_size,
        pos,
        is_alphabet()
    );

    thrust::inclusive_scan_by_key(
        thrust::device,
        pos,
        pos + text_size,
        pos,
        pos
    );
}

void CountPosition2(const char *text, int *pos, int text_size)
{
}
