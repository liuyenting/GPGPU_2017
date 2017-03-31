// Library
#include <cuda_runtime.h>

// Internal
#include "cutil_math.h"

// System
#include <iostream>

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
    DIFF = 1,   // diffuse
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

struct Sphere : Object {
    float rad;  // radius

    __device__
    virtual bool distance(const Ray &ray) const {
        /*
         * ray equation: p(x, y, z) = ray.orig + t*ray.dir
         * sphere equation: x^2 + y^2 + z^2 = rad^2
         *
         * quadratic: ax^2 + bx + c = 0 -> x = (-b +- sqrt(b^2 - 3ac)) / 2a
         *
         * solve t^2*ray.dir*ray.dir + 2*t*(orig-p)*ray.dir + (orig-p)*(orig-p) - rad*rad = 0
         */
        float3 dist = pos - ray.orig;
        float b = dot(dist, ray.dir);

        // discriminant
        float disc = b*b - dot(dist, dist) + rad*rad;

        if (disc < 0) {
            // ignroe complex solution
            return 0;
        } else {
            disc = sqrtf(disc);
        }
        // return the closest point relative the the origin of light ray
        float t;
        return ((t = b - disc) > EPS) ? t : ( ((t = b + disc) > EPS) ? t : 0 );
    }
};

/*
 * rendering equation:
 * outgoing radiance at a point = emitted radiance + reflected radiance
 *
 * reflected radiance = sum of (incoming radiance from hemisphere above point)
 *                      * reflectance function of material
 *                      * cosine incident angle
 */
__device__
float3 radiance(Ray r, unsigned int ) {
    float3 color = make_float3(0.0f);   // accumulated color
    float3 mask = make_float3(1.0f);

    // bounce the ray
    for (int count = 0; count < 4; count++) {
        int objId = 0;  // id of the closest object;
        float dist;     // distance to the closest object;

        // test whether the scence is intersected
        if (!intersect_scene(r, t, id)) {
            // return black if missed
            return make_float3(0.0f);
        }

        // compute impact location and normal vector
        const Object &obj =
        float3 p = ray.orig + r.dir * t;    // impact location
        float3 n = normaliz(p - obj.pos);   // normal
        // convert to front facing
        n = (dot(n, ray.dir) < 0) ? n : n * (-1);

        // add the photons from current object to accumulate the color
        color += mask * obj.emi;

        /*
         * generate new diffuse ray
         *     .orig = impac location
         *     .dir = random direction above the impact location
         */
        float r1 = 2 * PI * getrandom(s1, s2);  // random on unit sphere
        float r2 = getrandom(s1, s2);
        float r2s = sqrtf(r2);

        // compute local orthonormal basis uvw at hitpoint to use for calculation random ray direction
        // first vector = normal at hitpoint, second vector is orthogonal to first, third vector is orthogonal to first two vectors
        float3 w = nl;
        float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
        float3 v = cross(w,u);

        // compute random ray direction on hemisphere using polar coordinates
        // cosine weighted importance sampling (favours ray directions closer to normal direction)
        float3 d = normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrtf(1 - r2));

        // new ray origin is intersection point of previous ray with scene
        r.orig = x + nl*0.05f; // offset ray origin slightly to prevent self intersection
        r.dir = d;

        mask *= obj.col;    // multiply with colour of object
        mask *= dot(d,nl);  // weigh light contribution using cosine of angle between incident light and normal
        mask *= 2;          // fudge factor
    }

    return color;
}

__global__
void render(float3 *frame) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
}

int main() {
    const int width = 320, height = 240;

    float3 *h_frame = new float3[width*height];
    float3 *d_frame;
    gpuErrChk(cudaMalloc(&d_frame, width*height * sizeof(float3)));

    const dim3 threads(8, 8);
    // TODO: Use DivUp macro from previous project
    const dim3 blocks(width/threads.x, height/threads.y);

    render<<<blocks, threads>>>>(d_frame);

    gpuErrChk(cudaFree(d_frame));
    delete[] h_frame;

    return 0;
}
