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


struct Ray {
    float3 orig;    // origin
    float3 dir;     // direction

    __device__
    Ray(const float3 _orig, const float3 _dir)
        : orig(_orig), dir(_dir) {
    }
};

/*
 * Reflection type
 */
enum Refl_t {
    DIFF = 0,   // diffuse
    SPEC,       // speckle
    REFR        // refract
};

struct Object {
    float3 pos;     // position in world coordinate
    float3 emi;     // RGB emission
    float3 color;   // RGB
    Refl_t refl;    // type of reflection

    __device__
    virtual bool distance(const Ray &ray) const = 0;
    __device__
    virtual bool distance(const Object &obj) const = 0;
};

int main() {
    const int width = 320, height = 240;

    float3 *h_frame = new float3[width*height];
    float3 *d_frame;
    gpuErrChk(cudaMalloc(&d_frame, width*height * sizeof(float3)));

    gpuErrChk(cudaFree(d_frame));
    delete[] h_frame;

    return 0;
}
