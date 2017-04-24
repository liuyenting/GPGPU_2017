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
    __device__
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

namespace lab2 {

__global__
void transform(
    const char *input,
    int *output,
    const int n
) {
    for (
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        i < n;
        i += blockDim.x * gridDim.x
    ){
        output[i] = (input[i] != '\n') ? 1 : 0;
    }
}

__global__
void scan(
    const int *input,
    int *flag,
    int *output,
    const int n
) {
    extern __shared__ int temp[];

    int tid = threadIdx.x;
    int offset = 1;

    // pre-load data into the shared memory
    temp[2*tid] = input[2*tid];
    temp[2*tid+1] = input[2*tid+1];

    __syncthreads();
    if (tid == 0) {
        printf(" -- loaded --\n");
        for (int i = 0; i < n; i++) {
            printf("%d\t", temp[i]);
        }
        printf("\n");
    }

    // up-sweep
    if (tid == 0) {
        printf(" -- up-sweep --\n");
    }
    for (int d = n>>1; d > 0; d >>= 1) {
        __syncthreads();

        if (tid < d) {
            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;

            if (flag[bi] == 0) {
                temp[bi] += temp[ai];
            }
            flag[bi] |= flag[ai];
        }
        offset *= 2;

        __syncthreads();
        if (tid == 0) {
            for (int i = 0; i < n; i++) {
                printf("(%d,%d)\t", flag[i], temp[i]);
            }
            printf("\n");
        }
    }

    // remove the root element
    if (tid == 0) {
        temp[n-1] = 0;
        flag[n-1] = 0;
    }

    __syncthreads();
    if (tid == 0) {
        printf(" -- root removed --\n");
        for (int i = 0; i < n; i++) {
            printf("(%d,%d)\t", flag[i], temp[i]);
        }
        printf("\n");
    }

    // down-sweep
    if (tid == 0) {
        printf(" -- down-sweep --\n");
    }
    for (int d = 1;  d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();

        if (tid < d) {
            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;

            int t = temp[ai];
            temp[ai] = temp[bi];
            if (flag[ai+1] == 1) {
                temp[bi] = 0;
            } else if (flag[ai] == 1) {
                temp[bi] = t;
            } else {
                temp[bi] += t;
            }
            flag[ai] = 0;
        }

        __syncthreads();
        if (tid == 0) {
            for (int i = 0; i < n; i++) {
                printf("(%d,%d)\t", flag[i], temp[i]);
            }
            printf("\n");
        }
    }

    __syncthreads();

    // save the result
    output[2*tid] = temp[2*tid];
    output[2*tid+1] = temp[2*tid+1];

    __syncthreads();
    if (tid == 0) {
        printf(" -- output --\n");
        for (int i = 0; i < n; i++) {
            printf("%d\t", output[i]);
        }
        printf("\n");
    }
}

}

void CountPosition2(const char *text, int *pos, int text_size)
{
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

    lab2::transform<<<32*numSMs, 256>>>(text, pos, text_size);

    int data[] = { 4, 2, 1, 3, 0, 2, 1, 5 };
    int flag[] = { 1, 0, 0, 1, 0, 0, 1, 0 };
    int *d_data, *d_flag;
    cudaMalloc(&d_data, 8 * sizeof(int));
    cudaMalloc(&d_flag, 8 * sizeof(int));
    cudaMemcpy(d_data, data, 8 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flag, flag, 8 * sizeof(int), cudaMemcpyHostToDevice);
    lab2::scan<<<1, 8, 8*sizeof(int)>>>(d_data, d_flag, d_data, 8);
}
