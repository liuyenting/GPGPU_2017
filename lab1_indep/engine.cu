#include <iostream>
#include <cuda_runtime.h>

#define EPS     1e-4f
#define PI      3.14159265359f

#define gpuErrChk(func) { gpuAssert((func), __FILE__, __LINE__); }
inline void gpuAssert(
    cudaError_t ret,
    const char *fname,
    int line,
    bool forceStop=true
) {
    if (ret != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(ret);
        std::cerr << ' ' << fname;
        std::cerr << " ln" << line << std::endl;
        if (forceStop) {
            exit(ret);
        }
    }
}

int main() {
    const int width = 320, height = 240;

    float3 *h_frame = new float3[width*height];
    float3 *d_frame;
    gpuErrChk(cudaMalloc(&d_frame, width*height * sizeof(float3)));

    gpuErrChk(cudaFree(d_frame));
    delete[] h_frame;

    return 0;
}
