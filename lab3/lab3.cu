#include "lab3.h"
#include <cstdio>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
		}
	}
}

__global__ void PoissonClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
) {
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int ct = wt*yt+xt;

	// ignore out-of-range coordinates, 1px spacing
	if (yt < 1 or yt >= ht-1 or xt < 1 or xt >= wt-1) {
		return;
	}

	const int yb = oy+yt, xb = ox+xt;
	const int cb = wb*yb+xb;

	// calculate target N, S, W, E linear index
	const int nt = wt*(yt+1)+xt;
	const int st = wt*(yt-1)+xt;
	const int wt = wt*yt+(xt-1);
	const int et = wt*yt+(xt+1);
	// calculate background N, S, W, E linear index
	const int nb = wb*(yb+1)+xb;
	const int sb = wb*(yb-1)+xb;
	const int wb = wb*yb+(xb+1);
	const int eb = wb*yb+(xb-1);

	// surrounding pixel sum
	const float surPx0 = target[nt*3+0] + target[st*3+0] + target[wt*3+0] + target[t*3+0];
	const float surPx1 = target[nt*3+1] + target[st*3+1] + target[wt*3+1] + target[t*3+1];
	const float surPx2 = target[nt*3+2] + target[st*3+2] + target[wt*3+2] + target[t*3+2];

	// constant neighbor pixel
	const float conPx0 = 0.0f, conPx1 = 0.0f, conPx2 = 0.0f;
	// variate neighbor pixel
	const float varPx0 = 0.0f, varPx1 = 0.0f, varPx2 = 0.0f;
	// accumulate the pixels
	if (mask[nt] > 127.0f) {
		varPx0 += output[nt*3+0];
		varPx1 += output[nt*3+1];
		varPx2 += output[nt*3+2];
	} else {
		conPx0 += background[nt*3+0];
		conPx0 += background[nt*3+1];
		conPx0 += background[nt*3+2];
	}



	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
		}
	}
}

__global__ void CalculateFixed(
	const float *background,
	const float *target,
	const float *mask,
	float *fixed,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
) {

}

__global__ void PoissonImageCloningIteration(
	const float *fixed,
	const float *mask,
	const float *in,
	float *out,
	const int wt, const int ht
) {

}

void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	// setup
	float *fixed, *buf1, *buf2;
	cudaMalloc(&fixed, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf1, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf2, 3*wt*ht*sizeof(float));

	// initialize the iteration
	dim3 gdim(CeilDiv(wt, 32), CeilDiv(ht, 16)), bdim(32, 16);
	CalculateFixed<<<gdim, bdim>>>(
		background, target, mask, fixed,
		wb, hb, wt, ht, oy, ox
	);
	cudaMemcpy(buf1, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);

	// iterate
	for (int i = 0; i < 100; i++) {
		PoissonImageCloningIteration<<<gdim, bdim>>>(
			fixed, mask, buf1, buf2, wt, ht
		);
		PoissonImageCloningIteration<<<gdim, bdim>>>(
			fixed, mask, buf2, buf1, wt, ht
		);
	}

	// copy the image back
	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	SimpleClone<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
		background, target, mask, output,
		wb, hb, wt, ht, oy, ox
	);

	// clean up
	cudaFree(fixed);
	cudaFree(buf1);
	cudaFree(buf2);
}
