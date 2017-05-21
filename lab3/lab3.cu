#include "lab3.h"
#include <cstdio>
#include <cassert>

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

__global__ void CalculateFixed(
	const float *background,
	const float *target,
	const float *mask,
	float *fixed,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
) {
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int clt = wt*yt+xt;

	// ignore out-of-range pixels
	if (yt >= ht or xt >= wt) {
		return;
	}

	const int yb = oy+yt, xb = ox+xt;

	// calculate target N, S, W, E linear index
	const int nlt = wt*(yt+1)+xt;
	const int slt = wt*(yt-1)+xt;
	const int wlt = wt*yt+(xt-1);
	const int elt = wt*yt+(xt+1);
	// calculate background N, S, W, E linear index
	const int nlb = wb*(yb+1)+xb;
	const int slb = wb*(yb-1)+xb;
	const int wlb = wb*yb+(xb-1);
	const int elb = wb*yb+(xb+1);

	int npx = 4;
	// surrounding pixel sum
	float surPx0 = 0.0f, surPx1 = 0.0f, surPx2 = 0.0f;
	// constant neighbor pixel
	float conPx0 = 0.0f, conPx1 = 0.0f, conPx2 = 0.0f;
	if (yt < ht-1) {
		surPx0 += target[nlt*3+0];
		surPx1 += target[nlt*3+1];
		surPx2 += target[nlt*3+2];

		if (mask[nlt] <= 127.0f) {
			conPx0 += background[nlb*3+0];
			conPx1 += background[nlb*3+1];
			conPx2 += background[nlb*3+2];
		}
	} else {
		npx--;
	}
	if (yt > 1) {
		surPx0 += target[slt*3+0];
		surPx1 += target[slt*3+1];
		surPx2 += target[slt*3+2];

		if (mask[slt] <= 127.0f) {
			conPx0 += background[slb*3+0];
			conPx1 += background[slb*3+1];
			conPx2 += background[slb*3+2];
		}
	} else {
		npx--;
	}
	if (xt > 1) {
		surPx0 += target[wlt*3+0];
		surPx1 += target[wlt*3+1];
		surPx2 += target[wlt*3+2];

		if (mask[wlt] <= 127.0f) {
			conPx0 += background[wlb*3+0];
			conPx1 += background[wlb*3+1];
			conPx2 += background[wlb*3+2];
		}
	} else {
		npx--;
	}
	if (xt < wt-1) {
		surPx0 += target[elt*3+0];
		surPx1 += target[elt*3+1];
		surPx2 += target[elt*3+2];

		if (mask[elt] <= 127.0f) {
			conPx0 += background[elb*3+0];
			conPx1 += background[elb*3+1];
			conPx2 += background[elb*3+2];
		}
	} else {
		npx--;
	}

	// fill the constant value
	fixed[clt*3+0] = (npx*target[clt*3+0] - surPx0 + conPx0)/npx;
	fixed[clt*3+1] = (npx*target[clt*3+1] - surPx1 + conPx1)/npx;
	fixed[clt*3+2] = (npx*target[clt*3+2] - surPx2 + conPx2)/npx;
}

__global__ void PoissonImageCloningIteration(
	const float *fixed,
	const float *mask,
	const float *in,
	float *out,
	const int wt, const int ht
) {
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int clt = wt*yt+xt;

	// ignore out-of-range pixels
	if (yt >= ht or xt >= wt) {
		return;
	}

	// calculate target N, S, W, E linear index
	const int nlt = wt*(yt+1)+xt;
	const int slt = wt*(yt-1)+xt;
	const int wlt = wt*yt+(xt-1);
	const int elt = wt*yt+(xt+1);

	// total pixels
	int npx = 4;
	// constant neighbor pixel
	float varPx0 = 0.0f, varPx1 = 0.0f, varPx2 = 0.0f;
	// accumulate the background pixels
	if (yt < ht-1) {
		if (mask[nlt] > 127.0f) {
			varPx0 += in[nlt*3+0];
			varPx1 += in[nlt*3+1];
			varPx2 += in[nlt*3+2];
		}
	} else {
		npx--;
	}
	if (yt > 1) {
		if (mask[slt] > 127.0f) {
			varPx0 += in[slt*3+0];
			varPx1 += in[slt*3+1];
			varPx2 += in[slt*3+2];
		}
	} else {
		npx--;
	}
	if (xt > 1) {
		if (mask[wlt] > 127.0f) {
			varPx0 += in[wlt*3+0];
			varPx1 += in[wlt*3+1];
			varPx2 += in[wlt*3+2];
		}
	} else {
		npx--;
	}
	if (xt < wt-1) {
		if (mask[elt] > 127.0f) {
			varPx0 += in[elt*3+0];
			varPx1 += in[elt*3+1];
			varPx2 += in[elt*3+2];
		}
	} else {
		npx--;
	}
	assert(npx > 0);

	// fill the result
	out[clt*3+0] = fixed[clt*3+0] + varPx0/npx;
	out[clt*3+1] = fixed[clt*3+1] + varPx1/npx;
	out[clt*3+2] = fixed[clt*3+2] + varPx2/npx;
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
	for (int i = 0; i < 10000; i++) {
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
		background, buf1, mask, output,
		wb, hb, wt, ht, oy, ox
	);

	// clean up
	cudaFree(fixed);
	cudaFree(buf1);
	cudaFree(buf2);
}
