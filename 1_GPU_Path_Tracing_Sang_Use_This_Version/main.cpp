/*
*  CUDA based triangle mesh path tracer using BVH acceleration by Sam lapere, 2016
*  BVH implementation based on real-time CUDA ray tracer by Thanassis Tsiodras,
*  http://users.softlab.ntua.gr/~ttsiod/cudarenderer-BVH.html
*  Interactive camera with depth of field based on CUDA path tracer code
*  by Peter Kutz and Yining Karl Li, https://github.com/peterkutz/GPUPathTracer
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation; either version 2 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this program; if not, write to the Free Software
*  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/
#include <cuda.h>
#include <cuda_runtime.h>
#include <gl\glew.h>
#include <gl\freeglut.h>
#include <cuda_gl_interop.h>
#include <sstream>
#include <iostream>
#include <math.h>
#include "cuda_pathtracer.h"
#include "bvh.h"
//#include "loader.h"
//#include "camera.h"
//#include "New_bvh.h"

#ifndef M_PI
#define M_PI 3.14156265
#endif

using namespace std;

unsigned int framenumber = 0;
GLuint vbo;
void *d_vbo_buffer = NULL;

// CUDA arrays
/*Vertex* cudaVertices2 = NULL;
Triangle* cudaTriangles2 = NULL;

float *cudaTriangleIntersectionData2 = NULL;
int* cudaTriIdxList2 = NULL;
float *cudaBVHlimits2 = NULL;
int *cudaBVHindexesOrTrilists2 = NULL;*/

Camera* cudaRendercam2 = NULL;
bool buffer_reset = false;

void Timer(int obsolete) {

	glutPostRedisplay();
	glutTimerFunc(10, Timer, 0);
}

__device__ float timer = 0.0f;

// image buffer storing accumulated pixel samples
vec3* accumulatebuffer;
// final output buffer storing averaged pixel samples
vec3* finaloutputbuffer;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -30.0;

// TODO: Delete stuff at some point!!!
InteractiveCamera* interactiveCamera = NULL;
Camera* hostRendercam = NULL;  
Clock watch;

float scalefactor = 1.2f;

// this hash function calculates a new random number generator seed for each frame, based on framenumber  
unsigned int WangHash(unsigned int a) {
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}

// initialise camera on the CPU
void initCamera()
{
	delete interactiveCamera;
	interactiveCamera = new InteractiveCamera();

	interactiveCamera->setResolution(width, height);
	interactiveCamera->setFOVX(45);
}

// create OpenGL vertex buffer object for CUDA to store calculated pixels
void createVBO(GLuint* vbo)
{
	//Create vertex buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	//Initialize VBO
	unsigned int size = width * height * sizeof(vec3);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//Register VBO with CUDA
	cudaGLRegisterBufferObject(*vbo);
}
float* cuda_triangle_values;
int* cuda_triangle_indices;
float* cuda_bvh_boxes;
int* cuda_bvh_indices;



// display function called by glutMainLoop(), gets executed every frame 
void disp(void)   
{
	// if camera has moved, reset the accumulation buffer
	if (buffer_reset){ cudaMemset(accumulatebuffer, 1, width * height * sizeof(vec3)); framenumber = 0; }

	buffer_reset = false;
	framenumber++;

	// build a new camera for each frame on the CPU
	interactiveCamera->buildRenderCamera(hostRendercam); 

	// copy the CPU camera to a GPU camera
	cudaMemcpy(cudaRendercam2, hostRendercam, sizeof(Camera), cudaMemcpyHostToDevice);

	cudaThreadSynchronize();

	// maps a buffer object for acces by CUDA
	cudaGLMapBufferObject((void**)&finaloutputbuffer, vbo);									

	//clear all pixels:
	glClear(GL_COLOR_BUFFER_BIT);

	// calculate a new seed for the random number generator, based on the framenumber
	unsigned int hashedframes = WangHash(framenumber);

	// gateway from host to CUDA, passes all data needed to render frame (triangles, BVH tree, camera) to CUDA for execution
	//cudarender(finaloutputbuffer, accumulatebuffer, cudaTriangles2, cudaBVHindexesOrTrilists2, cudaBVHlimits2, cudaTriangleIntersectionData2, 
	//				cudaTriIdxList2, framenumber, hashedframes, cudaRendercam2); 

	cudarender(finaloutputbuffer, accumulatebuffer, cuda_triangle_values, cuda_triangle_indices, cuda_bvh_boxes,
		cuda_bvh_indices, framenumber, hashedframes, cudaRendercam2);

	cudaThreadSynchronize();
	cudaGLUnmapBufferObject(vbo);
	//glFlush();
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(2, GL_FLOAT, 12, 0);
	glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid*)8);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_POINTS, 0, width * height);
	glDisableClientState(GL_VERTEX_ARRAY);

	glutSwapBuffers();
	//glutPostRedisplay();
}

// keyboard interaction
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	switch (key) {
	
	case(27) : exit(0);
	case(' ') : initCamera(); buffer_reset = true; break;
	case('a') : interactiveCamera->strafe(-0.05f); buffer_reset = true; break;
	case('d') : interactiveCamera->strafe(0.05f); buffer_reset = true; break;
	case('r') : interactiveCamera->changeAltitude(0.05f); buffer_reset = true; break;
	case('f') : interactiveCamera->changeAltitude(-0.05f); buffer_reset = true; break;
	case('w') : interactiveCamera->goForward(0.05f); buffer_reset = true; break;
	case('s') : interactiveCamera->goForward(-0.05f); buffer_reset = true; break;
	case('g') : interactiveCamera->changeApertureDiameter(0.1); buffer_reset = true; break;
	case('h') : interactiveCamera->changeApertureDiameter(-0.1); buffer_reset = true; break;
	case('t') : interactiveCamera->changeFocalDistance(0.1); buffer_reset = true; break;
	case('y') : interactiveCamera->changeFocalDistance(-0.1); buffer_reset = true; break;
	}
}

void specialkeys(int key, int, int){

	switch (key) {

	case GLUT_KEY_LEFT: interactiveCamera->changeYaw(0.02f); buffer_reset = true; break;
	case GLUT_KEY_RIGHT: interactiveCamera->changeYaw(-0.02f); buffer_reset = true; break;
	case GLUT_KEY_UP: interactiveCamera->changePitch(0.02f); buffer_reset = true; break;
	case GLUT_KEY_DOWN: interactiveCamera->changePitch(-0.02f); buffer_reset = true; break;

	}
}

// mouse event handlers

int lastX = 0, lastY = 0;
int theButtonState = 0;
int theModifierState = 0;

// camera mouse controls in X and Y direction
void motion(int x, int y)
{
	int deltaX = lastX - x;
	int deltaY = lastY - y;

	if (deltaX != 0 || deltaY != 0) {

		if (theButtonState == GLUT_LEFT_BUTTON)  // Rotate
		{
			interactiveCamera->changeYaw(deltaX * 0.01);
			interactiveCamera->changePitch(-deltaY * 0.01);
		}
		else if (theButtonState == GLUT_MIDDLE_BUTTON) // Zoom
		{
			interactiveCamera->changeAltitude(-deltaY * 0.01);
		}

		if (theButtonState == GLUT_RIGHT_BUTTON) // camera move
		{
			interactiveCamera->changeRadius(-deltaY * 0.01);
		}

		lastX = x;
		lastY = y;
		buffer_reset = true;
		glutPostRedisplay(); 

	}
}

void mouse(int button, int state, int x, int y)
{
	theButtonState = button;
	theModifierState = glutGetModifiers();
	lastX = x;
	lastY = y;

	motion(x, y);
}
/*
// initialises scene data, builds BVH
void prepCUDAscene(){
	
	const char* scenefile = "dragon_vrip_res4.ply";  // teapot.ply, big_atc.ply
	
	// load scene
	float maxi = load_object(scenefile);

	// build the BVH
	UpdateBoundingVolumeHierarchy(scenefile);

	// now, allocate the CUDA side of the data (in CUDA global memory,
	// in preparation for the textures that will store them...)

	// store vertices in a GPU friendly format using float4
	float* pVerticesData = (float*)malloc(g_verticesNo * 8 * sizeof(float));
	for (unsigned f = 0; f<g_verticesNo; f++) {
		
		// first float4 stores vertex xyz position and precomputed ambient occlusion
		pVerticesData[f * 8 + 0] = g_vertices[f].x;
		pVerticesData[f * 8 + 1] = g_vertices[f].y;
		pVerticesData[f * 8 + 2] = g_vertices[f].z;
		pVerticesData[f * 8 + 3] = g_vertices[f]._ambientOcclusionCoeff;
		// second float4 stores vertex normal xyz
		pVerticesData[f * 8 + 4] = g_vertices[f]._normal.x;
		pVerticesData[f * 8 + 5] = g_vertices[f]._normal.y;
		pVerticesData[f * 8 + 6] = g_vertices[f]._normal.z;
		pVerticesData[f * 8 + 7] = 0.f;
	}

	// copy vertex data to CUDA global memory
	cudaMalloc((void**)&cudaVertices2, g_verticesNo * 8 * sizeof(float));
	cudaMemcpy(cudaVertices2, pVerticesData, g_verticesNo * 8 * sizeof(float), cudaMemcpyHostToDevice);

	// store precomputed triangle intersection data in a GPU friendly format using float4
	float *pTrianglesIntersectionData = (float *)malloc(g_trianglesNo * 20 * sizeof(float));

	for (unsigned e = 0; e<g_trianglesNo; e++) {
		// Texture-wise:
		//
		// first float4, triangle center + two-sided bool
		pTrianglesIntersectionData[20 * e + 0] = g_triangles[e]._center.x;
		pTrianglesIntersectionData[20 * e + 1] = g_triangles[e]._center.y;
		pTrianglesIntersectionData[20 * e + 2] = g_triangles[e]._center.z;
		pTrianglesIntersectionData[20 * e + 3] = g_triangles[e]._twoSided ? 1.0f : 0.0f;
		// second float4, normal
		pTrianglesIntersectionData[20 * e + 4] = g_triangles[e]._normal.x;
		pTrianglesIntersectionData[20 * e + 5] = g_triangles[e]._normal.y;
		pTrianglesIntersectionData[20 * e + 6] = g_triangles[e]._normal.z;
		pTrianglesIntersectionData[20 * e + 7] = g_triangles[e]._d;
		// third float4, precomputed plane normal of triangle edge 1
		pTrianglesIntersectionData[20 * e + 8] = g_triangles[e]._e1.x;  
		pTrianglesIntersectionData[20 * e + 9] = g_triangles[e]._e1.y;
		pTrianglesIntersectionData[20 * e + 10] = g_triangles[e]._e1.z;
		pTrianglesIntersectionData[20 * e + 11] = g_triangles[e]._d1;
		// fourth float4, precomputed plane normal of triangle edge 2
		pTrianglesIntersectionData[20 * e + 12] = g_triangles[e]._e2.x; 
		pTrianglesIntersectionData[20 * e + 13] = g_triangles[e]._e2.y;
		pTrianglesIntersectionData[20 * e + 14] = g_triangles[e]._e2.z;
		pTrianglesIntersectionData[20 * e + 15] = g_triangles[e]._d2;
		// fifth float4, precomputed plane normal of triangle edge 3
		pTrianglesIntersectionData[20 * e + 16] = g_triangles[e]._e3.x;
		pTrianglesIntersectionData[20 * e + 17] = g_triangles[e]._e3.y;
		pTrianglesIntersectionData[20 * e + 18] = g_triangles[e]._e3.z;
		pTrianglesIntersectionData[20 * e + 19] = g_triangles[e]._d3;
	}

	// copy precomputed triangle intersection data to CUDA global memory
	cudaMalloc((void**)&cudaTriangleIntersectionData2, g_trianglesNo * 20 * sizeof(float));
	cudaMemcpy(cudaTriangleIntersectionData2, pTrianglesIntersectionData, g_trianglesNo * 20 * sizeof(float), cudaMemcpyHostToDevice);

	// copy triangle data to CUDA global memory
	cudaMalloc((void**)&cudaTriangles2, g_trianglesNo*sizeof(Triangle));
	cudaMemcpy(cudaTriangles2, g_triangles, g_trianglesNo*sizeof(Triangle), cudaMemcpyHostToDevice);

	// Allocate CUDA-side data (global memory for corresponding textures) for Bounding Volume Hierarchy data
	// See BVH.h for the data we are storing (from CacheFriendlyBVHNode)

	// Leaf nodes triangle lists (indices to global triangle list)
	// copy triangle indices to CUDA global memory
	cudaMalloc((void**)&cudaTriIdxList2, g_triIndexListNo*sizeof(int));
	cudaMemcpy(cudaTriIdxList2, g_triIndexList, g_triIndexListNo*sizeof(int), cudaMemcpyHostToDevice);

	// Bounding box limits need bottom._x, top._x, bottom._y, top._y, bottom._z, top._z...
	// store BVH bounding box limits in a GPU friendly format using float2
	float *pLimits = (float *)malloc(g_pCFBVH_No * 6 * sizeof(float));

	for (unsigned h = 0; h<g_pCFBVH_No; h++) {
		// Texture-wise:
		// First float2
		pLimits[6 * h + 0] = g_pCFBVH[h]._bottom.x;
		pLimits[6 * h + 1] = g_pCFBVH[h]._top.x;
		// Second float2
		pLimits[6 * h + 2] = g_pCFBVH[h]._bottom.y;
		pLimits[6 * h + 3] = g_pCFBVH[h]._top.y;
		// Third float2
		pLimits[6 * h + 4] = g_pCFBVH[h]._bottom.z;
		pLimits[6 * h + 5] = g_pCFBVH[h]._top.z;
	}
	
	// copy BVH limits to CUDA global memory
	cudaMalloc((void**)&cudaBVHlimits2, g_pCFBVH_No * 6 * sizeof(float));
	cudaMemcpy(cudaBVHlimits2, pLimits, g_pCFBVH_No * 6 * sizeof(float), cudaMemcpyHostToDevice);

	// ..and finally, from CacheFriendlyBVHNode, the 4 integer values:
	// store BVH node attributes (triangle count, startindex, left and right child indices) in a GPU friendly format using uint4
	int *pIndexesOrTrilists = (int *)malloc(g_pCFBVH_No * 4 * sizeof(unsigned));

	for (unsigned g = 0; g<g_pCFBVH_No; g++) {
		// Texture-wise:
		// A single uint4
		pIndexesOrTrilists[4 * g + 0] = g_pCFBVH[g].u.leaf._count;  // number of triangles stored in this node if leaf node
		pIndexesOrTrilists[4 * g + 1] = g_pCFBVH[g].u.inner._idxRight; // index to right child if inner node
		pIndexesOrTrilists[4 * g + 2] = g_pCFBVH[g].u.inner._idxLeft;  // index to left node if inner node
		pIndexesOrTrilists[4 * g + 3] = g_pCFBVH[g].u.leaf._startIndexInTriIndexList; // start index in list of triangle indices if leaf node
		// union

	}

	// copy BVH node attributes to CUDA global memory
	cudaMalloc((void**)&cudaBVHindexesOrTrilists2, g_pCFBVH_No * 4 * sizeof(unsigned));
	cudaMemcpy(cudaBVHindexesOrTrilists2, pIndexesOrTrilists, g_pCFBVH_No * 4 * sizeof(unsigned), cudaMemcpyHostToDevice);
	
	// Initialisation Done!
	std::cout << "Rendering data initialised and copied to CUDA global memory\n";
}
*/


void prepCUDAScene2()
{
	/*float* triangles_value;
	int* triangle_indices;
	float* bvh_boxes;
	int* bvh_indices;
	float* vertex_normal;*/

	

	float* triangle_values = (float*)malloc(num_triangles * 16 * sizeof(float));
	int* triangle_indices = (int*)malloc(num_triangles * 4 * sizeof(int));

	for (int i = 0; i < num_triangles; ++i)
	{
		triangle_values[16 * i] = trs[i].p0.x;
		triangle_values[16 * i + 1] = trs[i].p0.y;
		triangle_values[16 * i + 2] = trs[i].p0.z;
		triangle_values[16 * i + 3] = 0.0f;

		triangle_values[16 * i + 4] = trs[i].e1.x;
		triangle_values[16 * i + 5] = trs[i].e1.y;
		triangle_values[16 * i + 6] = trs[i].e1.z;
		triangle_values[16 * i + 7] = 0.0f;

		triangle_values[16 * i + 8] = trs[i].e2.x;
		triangle_values[16 * i + 9] = trs[i].e2.y;
		triangle_values[16 * i + 10] = trs[i].e2.z;
		triangle_values[16 * i + 11] = 0.0f;

		triangle_values[16 * i + 12] = trs[i].n.x;
		triangle_values[16 * i + 13] = trs[i].n.y;
		triangle_values[16 * i + 14] = trs[i].n.z;
		triangle_values[16 * i + 15] = 0.0f;

		triangle_indices[4 * i] = triangle_index[3 * i];
		triangle_indices[4 * i + 1] = triangle_index[3 * i + 1];
		triangle_indices[4 * i + 2] = triangle_index[3 * i + 2];
		triangle_indices[4 * i + 3] = 0;
	}

	/*for (int i = 0; i < num_triangles; ++i)
	{
		for (int j = 0; j < 16; ++j)
			cout << triangle_values[16 * i + j] << " ";
		cout << "\n\n";

		for (int j = 0; j < 4; ++j)
			cout << triangle_indices[4 * i + j] << "\n\n";
	}*/
	// copy precomputed triangle intersection data to CUDA global memory
	//cudaMalloc((void**)&cudaTriangleIntersectionData2, g_trianglesNo * 20 * sizeof(float));
	//cudaMemcpy(cudaTriangleIntersectionData2, pTrianglesIntersectionData, g_trianglesNo * 20 * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&cuda_triangle_values, num_triangles * 16 * sizeof(float));
	cudaMemcpy(cuda_triangle_values, triangle_values, num_triangles * 16 * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&cuda_triangle_indices, num_triangles * 4 * sizeof(int));
	cudaMemcpy(cuda_triangle_indices, triangle_indices, num_triangles * 4 * sizeof(int), cudaMemcpyHostToDevice);

	float* bvh_boxes = (float*)malloc(num_nodes * 6 * sizeof(float));
	int* bvh_indices = (int*)malloc(num_nodes * 4 * sizeof(int));

	for (int i = 0; i < num_nodes; ++i)
	{
		vec3 b0 = flat_nodes[i].box.bbox[0];
		vec3 b1 = flat_nodes[i].box.bbox[1];

		bvh_boxes[6 * i] = b0.x;
		bvh_boxes[6 * i + 1] = b1.x; //b0.y;
		bvh_boxes[6 * i + 2] = b0.y;//b0.z;
		//bvh_boxes[6 * i + 3] = 0.0f;

		bvh_boxes[6 * i + 3] = b1.y;//b1.x;
		bvh_boxes[6 * i + 4] = b0.z;//b1.y;
		bvh_boxes[6 * i + 5] = b1.z;
		//bvh_boxes[8 * i + 7] = 0.0f;

		bvh_indices[4 * i] = flat_nodes[i].start;
		bvh_indices[4 * i + 1] = flat_nodes[i].range;
		bvh_indices[4 * i + 2] = flat_nodes[i].axis;
		bvh_indices[4 * i + 3] = flat_nodes[i].leaf;
	}

	cudaMalloc((void**)&cuda_bvh_boxes, num_nodes * 6 * sizeof(float));
	cudaMemcpy(cuda_bvh_boxes, bvh_boxes, num_nodes * 6 * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&cuda_bvh_indices, num_nodes * 4 * sizeof(int));
	cudaMemcpy(cuda_bvh_indices, bvh_indices, num_nodes * 4 * sizeof(int), cudaMemcpyHostToDevice);
}

int main(int argc, char** argv){

	
	// initialise an interactive camera on the CPU side
	initCamera();
	// create a CPU camera
	hostRendercam = new Camera();
	interactiveCamera->buildRenderCamera(hostRendercam);

	//string file_name = "cube.ply";

	//string file_name = "bunny.ply";

	//string file_name = "dragon_low.ply";

	string file_name = "dragon_high.ply";

	/*load_ply(file_name);

	node* root = NULL;
	int l = 4;

	build_bvh(root, l);*/

	node* root = NULL;
	//int l = 4;

	Read_Scene(root, file_name);

	//cout << "Static: \n";
	//cout << num_triangles << "\n";
	//cout << num_nodes << "\n";

	//total_triangle = num_triangles;
	//total_node = num_nodes;

	//cout << num_triangles << "\n";
	//cout << num_triangles << "\n";
	

	// initialise all data needed to start rendering (BVH data, triangles, vertices)
	prepCUDAScene2();
	
	// allocate GPU memory for accumulation buffer
	cudaMalloc(&accumulatebuffer, width * height * sizeof(vec3));
	// allocate GPU memory for interactive camera
	cudaMalloc((void**)&cudaRendercam2, sizeof(Camera));

	// init glut:
	glutInit(&argc, argv);
	// specify the display mode to be RGB and single buffering:
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	// specify the initial window position:
	glutInitWindowPosition(100, 100);
	// specify the initial window size:
	glutInitWindowSize(width, height);
	// create the window and set title:
	glutCreateWindow("GPU Path Tracing");
	//glutCreateWindow("Basic triangle mesh path tracer in CUDA");
	
	// init opengl:
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0.0, width, 0.0, height);
	fprintf(stderr, "OpenGL initialized \n");

	// register callback function to display graphics:
	glutDisplayFunc(disp);
	
	// functions for user interaction
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(specialkeys);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);

	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 ")) {
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		exit(0);
	}
	fprintf(stderr, "glew initialized  \n");
	// call Timer():
	Timer(0);
	createVBO(&vbo);
	fprintf(stderr, "VBO created  \n");
	// enter the main loop and process events:
	fprintf(stderr, "Entering glutMainLoop...  \n");
	glutMainLoop();


	printf("CUDA initialised.\nStart rendering...\n");

	// free CUDA memory
	cudaFree(finaloutputbuffer);  
	cudaFree(accumulatebuffer);
	cudaFree(cudaRendercam2);
	
	//float* cuda_triangle_value;
	//int* cuda_triangle_indices;
	//float* cuda_bvh_boxes;
	//int* cuda_bvh_indices;

	cudaFree(cuda_triangle_values);
	cudaFree(cuda_triangle_indices);
	cudaFree(cuda_bvh_boxes);
	cudaFree(cuda_bvh_indices);

	//cudaFree(cudaBVHindexesOrTrilists2);
	//cudaFree(cudaBVHlimits2);
	//cudaFree(cudaTriIdxList2);	
	//cudaFree(cudaTriangles2);
	//cudaFree(cudaTriangleIntersectionData2);
	//cudaFree(cudaVertices2);

	system("PAUSE");
}
