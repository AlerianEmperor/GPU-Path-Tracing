#include <gl\glew.h>
#include <gl\freeglut.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cuda.h>
#include <math_functions.h>
#include <vector_types.h>
#include <vector_functions.h>
#include "device_launch_parameters.h"
#include "cutil_math.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>
#include "render.h"
#include "camera.h"
union Colour  // 4 bytes = 4 chars = 1 float
{
	float c;
	uchar4 components;
};

__global__ void CoreLoopPathTracingKernel(vec3* output, vec3* accumbuffer, unsigned int framenumber, unsigned int hashedframenumber)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	// global threadId, see richiesams blogspot
	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	// create random number generator and initialise with hashed frame number, see RichieSams blogspot
	curandState randState; // state of the random number generator, to prevent repetition
	curand_init(hashedframenumber + threadId, 0, 0, &randState);

	vec3 finalcol; // final pixel colour  
	int i = (height - y - 1)*width + x; // pixel index in buffer
	int pixelx = x; // pixel x-coordinate on screen
	int pixely = height - y - 1; // pixel y-coordintate on screen

	/*Render Here*/
	finalcol = vec3(1.0f, 0.0f, 0.0f); // reset colour to zero for every pixel	

	accumbuffer[i] = finalcol;

	Colour fcolour;
	vec3 colour = vec3(clamp(finalcol.x, 0.0f, 1.0f), clamp(finalcol.y, 0.0f, 1.0f), clamp(finalcol.z, 0.0f, 1.0f));
	
	fcolour.components = make_uchar4((unsigned char)(powf(colour.x, 1 / 2.2f) * 255), (unsigned char)(powf(colour.y, 1 / 2.2f) * 255), (unsigned char)(powf(colour.z, 1 / 2.2f) * 255), 1);
	
	output[i] = vec3(x, y, fcolour.c);
}

void renderer(vec3* dptr, vec3* accumulatebuffer, unsigned framenumber, unsigned hashedframes)
{
	dim3 block(16, 16, 1);   // dim3 CUDA specific syntax, block and grid are required to schedule CUDA threads over streaming multiprocessors
	dim3 grid(width / block.x, height / block.y, 1);

	// Configure grid and block sizes:
	int threadsPerBlock = 256;
	
	int fullBlocksPerGrid = ((width * height) + threadsPerBlock - 1) / threadsPerBlock;
	
	CoreLoopPathTracingKernel << <grid, block >> >(dptr, accumulatebuffer, framenumber, hashedframes);
}
