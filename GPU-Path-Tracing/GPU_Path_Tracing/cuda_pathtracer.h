#ifndef __CUDA_PATHTRACER_H_
#define __CUDA_PATHTRACER_H_

//#include "vec3.h"
//#include "geometry.h"
//#include "bvh.h"
//#include "New_bvh.h"
#include "camera.h"
#include <ctime>
 
#define BVH_STACK_SIZE 32
#define width 1280	// screenwidth
#define height 720 // screenheight

#define DBG_PUTS(level, msg) \
    do { if (level <= 1) { puts(msg); fflush(stdout); }} while (0)

// global variables
/*extern unsigned g_verticesNo;
extern Vertex* g_vertices;
extern unsigned g_trianglesNo;
extern Triangle* g_triangles;
extern BVHNode* g_pSceneBVH;
extern unsigned g_triIndexListNo;
extern int* g_triIndexList;
extern unsigned g_pCFBVH_No;
extern CacheFriendlyBVHNode* g_pCFBVH;*/

//extern int total_triangle;
//extern int total_node;


// The gateway to CUDA, called from C++ (src/main.cpp)

//void cudarender(vec3* dptr, vec3* accumulatebuffer, Triangle* cudaTriangles, int* cudaBVHindexesOrTrilists,
//	float* cudaBVHlimits, float* cudaTriangleIntersectionData, int* cudaTriIdxList, unsigned framenumber, unsigned hashedframes, Camera* cudaRendercam); 

//void cudarender(vec3* dptr, vec3* accumulatebuffer, float* cuda_triangle_values, int* cuda_triangle_indices, float* cuda_bvh_boxes,
//int* cuda_bvh_indices, int total_triangle, int total_node, unsigned framenumber, unsigned hashedframes, Camera* cudaRendercam);

extern int num_triangles;
extern int num_nodes;

void cudarender(vec3* dptr, vec3* accumulatebuffer, float* cuda_triangle_values, int* cuda_triangle_indices, float* cuda_bvh_boxes,
	int* cuda_bvh_indices, unsigned framenumber, unsigned hashedframes, Camera* cudaRendercam);

struct Clock {
	unsigned firstValue;
	Clock() { reset(); }
	void reset() { firstValue = clock(); }
	unsigned readMS() { return (clock() - firstValue) / (CLOCKS_PER_SEC / 1000); }
};


#endif
