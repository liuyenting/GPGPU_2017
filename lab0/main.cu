#include <cstdio>
#include <cstdlib>
#include "SyncedMemory.h"

#define DIVUP(a, b)     ((a+b-1)/b)

#define CHECK {\
	auto e = cudaDeviceSynchronize();\
	if (e != cudaSuccess) {\
		printf("At " __FILE__ ":%d, %s\n", __LINE__, cudaGetErrorString(e));\
		abort();\
	}\
}

const int W = 40;
const int H = 12;

__global__ void Draw(char *frame) {
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (y < H and x < W) {
		char c;
		if (x == W-1) {
			c = y == H-1 ? '\0' : '\n';
		} else if (y == 0 or y == H-1 or x == 0 or x == W-2) {
			c = ':';
		} else {
			if (y < 5 or x < 8) {
				c = ' ';
			} else if (x == 33) {
				// the pole
				if (y == H-2) {
					c = '#';
				} else {
					c = '|';
				}
			} else if (x == 32 and y == 5) {
				// the flag
				c = '<';
			} else if (x > 7 and x < 22) {
				// the stairs
				if (-x/2+14 <= y) {
					c = '#';
				} else {
					c = ' ';
				}
			} else {
				c = ' ';
			}
		}
		frame[y*W+x] = c;
	}
}

int main(int argc, char **argv)
{
	MemoryBuffer<char> frame(W*H);
	auto frame_smem = frame.CreateSync(W*H);
	CHECK;

	dim3 threads(16, 12);
	dim3 blocks(DIVUP(W, threads.x), (DIVUP(H, threads.y)));
	Draw<<<blocks, threads>>>(frame_smem.get_gpu_wo());
	CHECK;

	puts(frame_smem.get_cpu_ro());
	CHECK;
	return 0;
}
