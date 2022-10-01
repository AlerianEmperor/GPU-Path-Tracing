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
#include <gl\glew.h>
#include <gl\freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

#include "cuda_pathtracer.h"

#define M_PI 3.1415926535897932384626422832795028841971f
#define TWO_PI 6.2831853071795864769252867665590057683943f
#define NUDGE_FACTOR     1e-3f  // epsilon
#define samps  1 // samples
#define BVH_STACK_SIZE 32
#define SCREEN_DIST (height*2)
#define epsilon 0.0001f

#define eps 1e-4

int texturewidth = 0;
int textureheight = 0;

__device__ int depth = 0;


// Textures for vertices, triangles and BVH data
// (see CudaRender() below, as well as main() to see the data setup process)
//texture<uint1, 1, cudaReadModeElementType> g_triIdxListTexture;
//texture<float2, 1, cudaReadModeElementType> g_pCFBVHlimitsTexture;
//texture<uint4, 1, cudaReadModeElementType> g_pCFBVHindexesOrTrilistsTexture;
//texture<float4, 1, cudaReadModeElementType> g_trianglesTexture;

//Vertex* cudaVertices;
//float* cudaTriangleIntersectionData;
//int* cudaTriIdxList = NULL;
//float* cudaBVHlimits = NULL;
//int* cudaBVHindexesOrTrilists = NULL;
//Triangle* cudaTriangles = NULL;

texture<float4, 1, cudaReadModeElementType> triangle_values_texture;
texture<int, 1, cudaReadModeElementType> triangle_indices_texture;
texture<float2, 1, cudaReadModeElementType> bvh_boxes_texture;
texture<int4, 1, cudaReadModeElementType> bvh_indices_texture;
texture<float4, 1, cudaReadModeElementType> vertex_normals_texture;


Camera* cudaRendercam = NULL;


struct Ray {
	float3 orig;	// ray origin
	float3 dir;		// ray direction	
	__device__ Ray(float3 o_, float3 d_) : orig(o_), dir(d_) {}
};

enum Refl_t { DIFF, METAL, SPEC, REFR, COAT };  // material types

struct Sphere {

	float rad;				// radius 
	float3 pos, emi, col;	// position, emission, color 
	Refl_t refl;			// reflection type (DIFFuse, SPECular, REFRactive)

	/*__device__ float intersect(const Ray &r) const { // returns distance, 0 if nohit 

		// Ray/sphere intersection
		// Quadratic formula required to solve ax^2 + bx + c = 0 
		// Solution x = (-b +- sqrt(b*b - 4ac)) / 2a
		// Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0 

		float3 op = pos - r.orig;  // 
		float t;//, epsilon = 0.01f;
		float b = dot(op, r.dir);
		float disc = b*b - dot(op, op) + rad*rad; // discriminant
		if (disc<0) return 0; else disc = sqrtf(disc);
		return (t = b - disc)>epsilon ? t : ((t = b + disc)>epsilon ? t : 0);
	}*/

	__device__ bool intersect(const Ray& r, double& t)
	{
		float3 oc = r.orig - pos;
		
		double b = dot(oc, r.dir);
		double c = dot(oc, oc) - rad * rad;
		double discriminant = b * b - c;
		if (discriminant < 0.0)
			return false;
		else
		{
			double dis = sqrtf(discriminant);

			double tmin = (-b - dis);// *inv_a;
			if (tmin > eps)
			{
				t = tmin;
				return true;
			}
			double tmax = (-b + dis);// *inv_a;
			if (tmax > eps)
			{
				t = tmax;
				return true;
			}
			return false;
		}
		//return true;
	}

};

__device__ Sphere spheres[] = {

	// sun
	{ 1.6, { 0.0f, 2.8, 0 }, { 6, 4, 2 }, { 0.f, 0.f, 0.f }, DIFF },  // 37, 34, 30  X: links rechts Y: op neer
	//{ 1600, { 3000.0f, 10, 6000 }, { 17, 14, 10 }, { 0.f, 0.f, 0.f }, DIFF },

	// horizon sun2
	//	{ 1560, { 3500.0f, 0, 7000 }, { 50, 25, 2.5 }, { 0.f, 0.f, 0.f }, DIFF },  //  150, 75, 7.5

	// sky
	//{ 10000, { 50.0f, 40.8f, -1060 }, { 0.1, 0.3, 0.55 }, { 0.175f, 0.175f, 0.25f }, DIFF }, // 0.0003, 0.01, 0.15, or brighter: 0.2, 0.3, 0.6
	{ 10000, { 50.0f, 40.8f, -1060 }, { 0.51, 0.51, 0.51 }, { 0.175f, 0.175f, 0.25f }, DIFF },

	// ground
	{ 100000, { 0.0f, -100001.1, 0 }, { 0, 0, 0 }, { 0.5f, 0.0f, 0.0f }, COAT },
	{ 100000, { 0.0f, -100001.2, 0 }, { 0, 0, 0 }, { 0.3f, 0.3f, 0.3f }, DIFF }, // double shell to prevent light leaking

	// horizon brightener
	{ 110000, { 50.0f, -110048.5, 0 }, { 3.6, 2.0, 0.2 }, { 0.f, 0.f, 0.f }, DIFF },
	// mountains
	//{ 4e4, { 50.0f, -4e4 - 30, -3000 }, { 0, 0, 0 }, { 0.2f, 0.2f, 0.2f }, DIFF },
	// white Mirr
	{ 1.1, { 1.6, 0, 1.0 }, { 0, 0.0, 0 }, { 0.9f, .9f, 0.9f }, SPEC }
	// Glass
	//{ 0.3, { 0.0f, -0.4, 4 }, { .0, 0., .0 }, { 0.9f, 0.9f, 0.9f }, DIFF },
	// Glass2
	//{ 22, { 87.0f, 22, 24 }, { 0, 0, 0 }, { 0.9f, 0.9f, 0.9f }, SPEC },
};


// Create OpenGL BGR value for assignment in OpenGL VBO buffer
__device__ int getColor(vec3& p)  // converts vec3 colour to int
{
	return (((unsigned)p.z) << 16) | (((unsigned)p.y) << 8) | (((unsigned)p.x));
}

inline __device__ vec3 minf3(vec3 a, vec3 b){ return vec3(a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y, a.z < b.z ? a.z : b.z); }
inline __device__ vec3 maxf3(vec3 a, vec3 b){ return vec3(a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y, a.z > b.z ? a.z : b.z); }
inline __device__ float3 minf3(float3 a, float3 b){ return make_float3(a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y, a.z < b.z ? a.z : b.z); }
inline __device__ float3 maxf3(float3 a, float3 b){ return make_float3(a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y, a.z > b.z ? a.z : b.z); }
inline __device__ float minf1(float a, float b){ return a < b ? a : b; }
inline __device__ float maxf1(float a, float b){ return a > b ? a : b; }

__device__ bool RayIntersectsBox_2(const float3& originInWorldSpace, const float3& rayInWorldSpace, float& hit_t, int boxIdx)
{
	float3 inv_d = make_float3(1.0f / rayInWorldSpace.x, 1.0f / rayInWorldSpace.y, 1.0f / rayInWorldSpace.z);
	

	float2 limit_x = tex1Dfetch(bvh_boxes_texture, 3 * boxIdx);
	float2 limit_y = tex1Dfetch(bvh_boxes_texture, 3 * boxIdx + 1);
	float2 limit_z = tex1Dfetch(bvh_boxes_texture, 3 * boxIdx + 2);

	float3 box_min = make_float3(limit_x.x, limit_y.x, limit_z.x);
	float3 box_max = make_float3(limit_x.y, limit_y.y, limit_z.y);


	float3 tmin = (box_min - originInWorldSpace) * inv_d;
	float3 tmax = (box_max - originInWorldSpace) * inv_d;

	float3 real_min = minf3(tmin, tmax);
	float3 real_max = maxf3(tmin, tmax);

	float th = minf1(minf1(real_max.x, real_max.y), real_max.z);
	float tl = maxf1(maxf1(real_min.x, real_min.y), real_min.z);

	if(th <= 0.0f)
		return false;

	return tl <= 0.0f ||  tl <= th * 1.00000024f && tl <= hit_t;


	/*if(tl >= hit_t)
		return false;

	if (1.00000024f * th >= tl) 
	{
		return th > epsilon ? true : false;
	}
	else 
		return false;*/
}


//////////////////////////////////////////
//	BVH intersection routine	//
//	using CUDA texture memory	//
//////////////////////////////////////////


__device__ bool hit_triangle(float3& origin, float3& direction, int& triangle_ind, float& mint, float& hit_u, float& hit_v)//, float4& n)
{
	float4 p0 = tex1Dfetch(triangle_values_texture, 3 * triangle_ind);
	float4 e1 = tex1Dfetch(triangle_values_texture, 3 * triangle_ind + 1);
	float4 e2 = tex1Dfetch(triangle_values_texture, 3 * triangle_ind + 2);

	/*float4 pvec = cross(direction, e2);
	float det = dot(pvec, e1);
		if (det > -0.000002f && det < 0.000002f)
			return false;

		float inv_det = 1.0f / det;
		float4 tvec = make_float4(origin.x - p0.x, origin.y - p0.y, origin.z - p0.z, 0.0f);
		
		float4 qvec = cross(tvec, e1);


		float u = dot(tvec, pvec) * inv_det;
		
		if (u <= 0.0f || u >= 1.0f)
			return false;
		
		float v = dot(qvec, direction) * inv_det;
		
		if (v <= 0.0f || u + v >= 1.0f)
			return false;
		
		float t = dot(e2, qvec) * inv_det;
		
		if(t < mint && t >= 0.0f)
		{
			mint = t;
			return true;
		}
		return false;*/

	//--------current use--------
	float4 tvec = make_float4(origin.x - p0.x, origin.y - p0.y, origin.z - p0.z, 0.0f);
	float4 pvec = cross(direction, e2);

	float det = dot(e1, pvec);

	float4 qvec;

	if (det > eps)
	{

		float u = dot(tvec, pvec);

		if (u <= 0.0f || u >= det)
			return false;

		qvec = cross(tvec, e1);

		float v = dot(direction, qvec);

		if (v <= 0.0f || u + v >= det)
			return false;

		hit_u = u;
		hit_v = v;
	}
	else if (det < -eps)
	{
		float u = dot(tvec, pvec);

		if (u >= 0.0f || u <= det)
			return false;

		qvec = cross(tvec, e1);

		float v = dot(direction, qvec);

		if (v >= 0.0 || u + v <= det)
			return false;
		hit_u = u;
		hit_v = v;
	}
	else
		return false;

	float t = dot(e2, qvec) / det;
	if (t >= mint || t < 0.0f)
		return false;

	mint = t;

	return true;

	
	/*float u = dot(tvec, pvec);// * det;

	if(u <= 0.0f || u >= det)
		return false;

	float4 qvec = cross(tvec, e1);

	float v = dot(direction, qvec);// * det;

	if(v <= 0.0f || (u + v) >= det)
		return false;

	float inv_det = 1.0f / det;

	t = dot(e2, qvec) * inv_det;

	//n = tex1Dfetch(triangle_values_texture, 4 * triangle_ind + 3);

	return true;*/

	
	
	
	

	/*float4 tvec = make_float4(origin.x - p0.x, origin.y - p0.y, origin.z - p0.z, 0.0f);
	float4 pvec = cross(direction, e2);
	float det = dot(e1, pvec);

	det = __fdividef(1.0f, det);  // CUDA intrinsic function 

	float u = dot(tvec, pvec) * det;

	if (u <= 0.0f || u >= 1.0f)
		return false;

	float4 qvec = cross(tvec, e1);

	float v = dot(direction, qvec) * det;

	if (v <= 0.0f || (u + v) >= 1.0f)
		return false;

	float t = dot(e2, qvec) * det;

	if(t > mint || t < eps)
		return false;
	
	//if(t < 0.0f)
	//return false;
	mint = t;

	return true;*/
}

__device__ bool BVH_IntersectTriangles_Brute_Force( float3& origin, float3& ray, int& triangle_ind, float& hit_t, vec3& n)
{
	float cache = hit_t;
	                  
	for(int i = 0; i < 12; ++i)
	{
		float t;
		float u, v;		
		hit_triangle(origin, ray, i, t, u, v);//, n);

		if(t < hit_t)
		{		
			triangle_ind = i;
			hit_t = t;
			//return true;
		}
	}

	
	return hit_t < cache;
}

__device__ bool BVH_IntersectTriangles(float3& origin, float3& ray, int& triangle_ind, float& hit_t, float& hit_u, float& hit_v, int avoidSelf)
{
	// in the loop below, maintain the closest triangle and the point where we hit it:
	triangle_ind = -1;

	//float min_t = 1e20;

	int stack[BVH_STACK_SIZE];
	
	int stackIdx = 0;
	stack[stackIdx] = 0; 

	while (stackIdx >= 0) 
	{		
		int boxIdx = stack[stackIdx];
		
		stackIdx--;

		int4 data = tex1Dfetch(bvh_indices_texture, boxIdx);

		if (RayIntersectsBox_2(origin, ray, hit_t, boxIdx)) 
		{
			if (data.w == 0) 
			{   
				
				stack[++stackIdx] = data.x + 1; // right child node index
								
				stack[++stackIdx] = data.x ; // left child node index
			}
			else 
			{ 
				//float4 n;
				for (int i = data.x; i < data.x + data.y; i++) 
				{	
					if(i == avoidSelf)
						continue;
						
					float t, u, v;
				
					
					hit_triangle(origin, ray, i, t, u, v);//, n);

					if(t < hit_t)
					{
						triangle_ind = i;
						hit_t = t;
						hit_u = u;
						hit_v = v;
					}
				}
			}
		}			
	}
	

	/*if(min_t < 1e20)
	{
		hit_t = min_t;
		return true;
	}

	return false;*/

	return triangle_ind != - 1;
}

__device__ float4 intp(float4& a, float4& b, float4& c, float& u, float& v)
{
		return a * (1.0f - u - v) + b * u + c * v;
}

//////////////////////
// PATH TRACING
//////////////////////

//__device__ vec3 path_trace(curandState *randstate, vec3 originInWorldSpace, vec3 rayInWorldSpace, int avoidSelf,
//	Triangle *pTriangles, int* cudaBVHindexesOrTrilists, float* cudaBVHlimits, float* cudaTriangleIntersectionData, int* cudaTriIdxList)

__device__ vec3 path_trace(curandState *randstate, vec3& originInWorldSpace, vec3& rayInWorldSpace, int avoidSelf)
{
	vec3 mask = vec3(1.0f, 1.0f, 1.0f);
	// accumulated colour
	vec3 accucolor = vec3(0.0f, 0.0f, 0.0f);

	for (int bounces = 0; bounces < 5; bounces++){  // iteration up to 4 bounces (instead of recursion in CPU code)

		int sphere_id = -1;
		int triangle_id = -1;
		int pBestTriIdx = -1;
		int geomtype = -1;
				
		
		float triangle_t = 1e20;
		double d = 1e20;
		
		float scene_t = 1e20;
		
		vec3 pointHitInWorldSpace;


		vec3 f = vec3(0, 0, 0);
		vec3 emit = vec3(0, 0, 0);
		vec3 x;
		vec3 n; 
		vec3 nl; 
		vec3 boxnormal = vec3(0, 0, 0);
		vec3 dw; 
		Refl_t refltype;
		                            
		float3 rayorig = make_float3(originInWorldSpace.x, originInWorldSpace.y, originInWorldSpace.z);
		float3 raydir = make_float3(rayInWorldSpace.x, rayInWorldSpace.y, rayInWorldSpace.z);
		
		
		//good brute force test
		//BVH_IntersectTriangles_Brute_Force(rayorig, raydir, pBestTriIdx, triangle_t, n);

		float hit_u, hit_v;

		//float cache = 1e20;
		BVH_IntersectTriangles(rayorig, raydir, pBestTriIdx, triangle_t, hit_u, hit_v, avoidSelf);

		avoidSelf = pBestTriIdx;

		if (triangle_t < scene_t && triangle_t >= 0.0f) // EPSILON
		{
			scene_t = triangle_t;
			triangle_id = pBestTriIdx;
			geomtype = 2;
		}
		
		float numspheres = sizeof(spheres) / sizeof(Sphere);		
		for (int i = 0; i < numspheres; ++i)
		{  
			if(spheres[i].intersect(Ray(rayorig, raydir), d))
			{
				if(d < scene_t)
				{
					scene_t = d; 
					sphere_id = i;
					geomtype = 1;
				}
			}
		}

		
		
		

		
		// set avoidSelf to current triangle index to avoid intersection between this triangle and the next ray, 
		// so that we don't get self-shadow or self-reflection from this triangle...
		
		
		

		if (scene_t == 1e20) return vec3(0.0f, 0.0f, 0.0f);

		// SPHERES:
		if (geomtype == 1){

			Sphere &sphere = spheres[sphere_id]; // hit object with closest intersection
			x = originInWorldSpace + rayInWorldSpace * scene_t;  // intersection point on object
			n = vec3(x.x - sphere.pos.x, x.y - sphere.pos.y, x.z - sphere.pos.z);		// normal
			n.normalize();
			nl = dot(n, rayInWorldSpace) < 0 ? n : n * -1; // correctly oriented normal
			f = vec3(sphere.col.x, sphere.col.y, sphere.col.z);   // object colour
			refltype = sphere.refl;
			emit = vec3(sphere.emi.x, sphere.emi.y, sphere.emi.z);  // object emission
			accucolor += (mask * emit);
		}

		// TRIANGLES:5
		else if (geomtype == 2){

			//pBestTri = &pTriangles[triangle_id];

			//x = pointHitInWorldSpace;  // intersection point
			
			
			//n = pBestTri->_normal;  // normal
			//n = vec3(0,0,1);

			//float4 normal = tex1Dfetch(triangle_values_texture, 4 * pBestTriIdx + 3);

			//n = vec3(normal.x, normal.y, normal.z);

			//n = n.norm();
			
			int ind_v0 = tex1Dfetch(triangle_indices_texture, 4 * pBestTriIdx);
			int ind_v1 = tex1Dfetch(triangle_indices_texture, 4 * pBestTriIdx + 1);
			int ind_v2 = tex1Dfetch(triangle_indices_texture, 4 * pBestTriIdx + 2);


			float4 n0 = tex1Dfetch(vertex_normals_texture, ind_v0);
			float4 n1 = tex1Dfetch(vertex_normals_texture, ind_v0);
			float4 n2 = tex1Dfetch(vertex_normals_texture, ind_v0);


			float4 norm = intp(n0, n1, n2, hit_u, hit_v);

			n = vec3(norm.x, norm.y, norm.z);
			
			nl = dot(n, rayInWorldSpace) < 0 ? n : -n;  // correctly oriented normal

			x = originInWorldSpace + rayInWorldSpace * scene_t + 0.002f * n;

			vec3 colour = vec3(0.9f, 0.3f, 0.0f); // hardcoded triangle colour
			
			refltype = COAT;
			f = colour;
			
		}

		// basic material system, all parameters are hard-coded (such as phong exponent, index of refraction)
		
		// diffuse material, based on smallpt by Kevin Beason 
		if (refltype == DIFF){

			// pick two random numbers
			float phi = 2 * M_PI * curand_uniform(randstate);
			float r2 = curand_uniform(randstate);
			float r2s = sqrtf(r2);

			// compute orthonormal coordinate frame uvw with hitpoint as origin 
			vec3 w = nl; w.normalize();
			vec3 u = cross((fabs(w.x) > .1 ? vec3(0, 1, 0) : vec3(1, 0, 0)), w); u.normalize();
			vec3 v = cross(w, u);

			// compute cosine weighted random ray direction on hemisphere 
			dw = u*cosf(phi)*r2s + v*sinf(phi)*r2s + w*sqrtf(1 - r2); 
			dw.normalize();

			// offset origin next path segment to prevent self intersection
			pointHitInWorldSpace = x + w * 0.01;  // scene size dependent

			// multiply mask with colour of object
			mask *= f;
		}

		// Phong metal material from "Realistic Ray Tracing", P. Shirley
		if (refltype == METAL){

			// compute random perturbation of ideal reflection vector
			// the higher the phong exponent, the closer the perturbed vector is to the ideal reflection direction
			float phi = 2 * M_PI * curand_uniform(randstate);
			float r2 = curand_uniform(randstate);
			float phongexponent = 20;
			float cosTheta = powf(1 - r2, 1.0f / (phongexponent + 1));
			float sinTheta = sqrtf(1 - cosTheta * cosTheta);

			// create orthonormal basis uvw around reflection vector with hitpoint as origin 
			// w is ray direction for ideal reflection
			vec3 w = rayInWorldSpace - n * 2.0f * dot(n, rayInWorldSpace); w.normalize();
			vec3 u = cross((fabs(w.x) > .1 ? vec3(0, 1, 0) : vec3(1, 0, 0)), w); u.normalize();
			vec3 v = cross(w, u); // v is normalised by default

			// compute cosine weighted random ray direction on hemisphere 
			dw = u * cosf(phi) * sinTheta + v * sinf(phi) * sinTheta + w * cosTheta; 
			dw.normalize();

			// offset origin next path segment to prevent self intersection
			pointHitInWorldSpace = x + w * 0.01;  // scene size dependent

			// multiply mask with colour of object
			mask *= f;
		}

		// specular material (perfect mirror)
		if (refltype == SPEC){

			// compute reflected ray direction according to Snell's law
			dw = rayInWorldSpace - n * 2.0f * dot(n, rayInWorldSpace);

			// offset origin next path segment to prevent self intersection
			pointHitInWorldSpace = x + nl * 0.01;   // scene size dependent

			// multiply mask with colour of object
			mask *= f;
		}

		// COAT material based on https://github.com/peterkutz/GPUPathTracer
		// randomly select diffuse or specular reflection
		// looks okay-ish but inaccurate (no Fresnel calculation yet)
		if (refltype == COAT){
		
			float rouletteRandomFloat = curand_uniform(randstate);
			float threshold = 0.05f;
			vec3 specularColor = vec3(1,1,1);  // hard-coded
			bool reflectFromSurface = (rouletteRandomFloat < threshold); //computeFresnel(make_vec3(n.x, n.y, n.z), incident, incidentIOR, transmittedIOR, reflectionDirection, transmissionDirection).reflectionCoefficient);
			
			if (reflectFromSurface) { // calculate perfectly specular reflection
				
				// Ray reflected from the surface. Trace a ray in the reflection direction.
				// TODO: Use Russian roulette instead of simple multipliers! (Selecting between diffuse sample and no sample (absorption) in this case.)
				
				mask *= specularColor;
				dw = rayInWorldSpace - n * 2.0f * dot(n, rayInWorldSpace);

				// offset origin next path segment to prevent self intersection
				pointHitInWorldSpace = x + nl * 0.01; // scene size dependent
			}

			else {  // calculate perfectly diffuse reflection
			
				float r1 = 2 * M_PI * curand_uniform(randstate);
				float r2 = curand_uniform(randstate);
				float r2s = sqrtf(r2);

				// compute orthonormal coordinate frame uvw with hitpoint as origin 
				vec3 w = nl; w.normalize();
				vec3 u = cross((fabs(w.x) > .1 ? vec3(0, 1, 0) : vec3(1, 0, 0)), w); u.normalize();
				vec3 v = cross(w, u);

				// compute cosine weighted random ray direction on hemisphere 
				dw = u*cosf(r1)*r2s + v*sinf(r1)*r2s + w*sqrtf(1 - r2);
				dw.normalize();

				// offset origin next path segment to prevent self intersection
				pointHitInWorldSpace = x + nl * 0.01;  // // scene size dependent

				// multiply mask with colour of object
				mask *= f;
				//mask *= make_vec3(0.15f, 0.15f, 0.15f);  // gold metal
			}	
		} // end COAT

		// perfectly refractive material (glass, water)
		if (refltype == REFR){

			bool into = dot(n, nl) > 0; // is ray entering or leaving refractive material?
			float nc = 1.0f;  // Index of Refraction air
			float nt = 1.5f;  // Index of Refraction glass/water
			float nnt = into ? nc / nt : nt / nc;  // IOR ratio of refractive materials
			float ddn = dot(rayInWorldSpace, nl);
			float cos2t = 1.0f - nnt*nnt * (1.f - ddn*ddn);

			if (cos2t < 0.0f) // total internal reflection 
			{
				dw = rayInWorldSpace;
				dw -= n * 2.0f * dot(n, rayInWorldSpace);

				// offset origin next path segment to prevent self intersection
				pointHitInWorldSpace = x + nl * 0.01; // scene size dependent
			}
			else // cos2t > 0
			{
				// compute direction of transmission ray
				vec3 tdir = rayInWorldSpace * nnt;
				tdir -= n * ((into ? 1 : -1) * (ddn*nnt + sqrtf(cos2t)));
				tdir.normalize();

				float R0 = (nt - nc)*(nt - nc) / (nt + nc)*(nt + nc);
				float c = 1.f - (into ? -ddn : dot(tdir, n));
				float Re = R0 + (1.f - R0) * c * c * c * c * c;
				float Tr = 1 - Re; // Transmission
				float P = .25f + .5f * Re;
				float RP = Re / P;
				float TP = Tr / (1.f - P);

				// randomly choose reflection or transmission ray
				if (curand_uniform(randstate) < 0.25) // reflection ray
				{
					mask *= RP;
					dw = rayInWorldSpace;
					dw -= n * 2.0f * dot(n, rayInWorldSpace);

					pointHitInWorldSpace = x + nl * 0.01; // scene size dependent
				}
				else // transmission ray
				{
					mask *= TP;
					dw = tdir; //r = Ray(x, tdir); 
					pointHitInWorldSpace = x + nl * 0.001f; // epsilon must be small to avoid artefacts
				}
			}
		}

		// set up origin and direction of next path segment
		originInWorldSpace = pointHitInWorldSpace;
		rayInWorldSpace = dw;
	}

	return vec3(accucolor.x, accucolor.y, accucolor.z);
}

union Colour  // 4 bytes = 4 chars = 1 float
{
	float c;
	uchar4 components;
};

// the core path tracing kernel, 
// running in parallel for all pixels
//__global__ void CoreLoopPathTracingKernel(vec3* output, vec3* accumbuffer, Triangle* pTriangles, Camera* cudaRendercam,
//	int* cudaBVHindexesOrTrilists, float* cudaBVHlimits, float* cudaTriangleIntersectionData,
//	int* cudaTriIdxList, unsigned int framenumber, unsigned int hashedframenumber)

__global__ void CoreLoopPathTracingKernel(vec3* output, vec3* accumbuffer, Camera* cudaRendercam, unsigned int framenumber, unsigned int hashedframenumber)

{
	// assign a CUDA thread to every pixel by using the threadIndex
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	// global threadId, see richiesams blogspot
	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	// create random number generator and initialise with hashed frame number, see RichieSams blogspot
	curandState randState; // state of the random number generator, to prevent repetition
	curand_init(hashedframenumber + threadId, 0, 0, &randState);

	vec3 finalcol; // final pixel colour  
	vec3 rendercampos = vec3(cudaRendercam->position.x, cudaRendercam->position.y, cudaRendercam->position.z);

	int i = (height - y - 1)*width + x; // pixel index in buffer
	int pixelx = x; // pixel x-coordinate on screen
	int pixely = height - y - 1; // pixel y-coordintate on screen

	finalcol = vec3(0.0f, 0.0f, 0.0f); // reset colour to zero for every pixel	

	for (int s = 0; s < samps; s++){

		// compute primary ray direction
		// use camera view of current frame (transformed on CPU side) to create local orthonormal basis
		vec3 rendercamview = vec3(cudaRendercam->view.x, cudaRendercam->view.y, cudaRendercam->view.z); rendercamview.normalize(); // view is already supposed to be normalized, but normalize it explicitly just in case.
		vec3 rendercamup = vec3(cudaRendercam->up.x, cudaRendercam->up.y, cudaRendercam->up.z); rendercamup.normalize();
		vec3 horizontalAxis = cross(rendercamview, rendercamup); horizontalAxis.normalize(); // Important to normalize!
		vec3 verticalAxis = cross(horizontalAxis, rendercamview); verticalAxis.normalize(); // verticalAxis is normalized by default, but normalize it explicitly just for good measure.

		vec3 middle = rendercampos + rendercamview;
		vec3 horizontal = horizontalAxis * tanf(cudaRendercam->fov.x * 0.5 * (M_PI / 180)); // Now treating FOV as the full FOV, not half, so I multiplied it by 0.5. I also normzlized A and B, so there's no need to divide by the length of A or B anymore. Also normalized view and removed lengthOfView. Also removed the cast to float.
		vec3 vertical = verticalAxis * tanf(-cudaRendercam->fov.y * 0.5 * (M_PI / 180)); // Now treating FOV as the full FOV, not half, so I multiplied it by 0.5. I also normzlized A and B, so there's no need to divide by the length of A or B anymore. Also normalized view and removed lengthOfView. Also removed the cast to float.

		// anti-aliasing
		// calculate center of current pixel and add random number in X and Y dimension
		// based on https://github.com/peterkutz/GPUPathTracer 
		float jitterValueX = curand_uniform(&randState) - 0.5;
		float jitterValueY = curand_uniform(&randState) - 0.5;
		float sx = (jitterValueX + pixelx) / (cudaRendercam->resolution.x - 1);
		float sy = (jitterValueY + pixely) / (cudaRendercam->resolution.y - 1);

		// compute pixel on screen
		vec3 pointOnPlaneOneUnitAwayFromEye = middle + ( horizontal * ((2 * sx) - 1)) + ( vertical * ((2 * sy) - 1));
		vec3 pointOnImagePlane = rendercampos + ((pointOnPlaneOneUnitAwayFromEye - rendercampos) * cudaRendercam->focalDistance); // Important for depth of field!		

		// calculation of depth of field / camera aperture 
		// based on https://github.com/peterkutz/GPUPathTracer 
		
		vec3 aperturePoint;

		if (cudaRendercam->apertureRadius > 0.00001) { // the small number is an epsilon value.
		
			// generate random numbers for sampling a point on the aperture
			float random1 = curand_uniform(&randState);
			float random2 = curand_uniform(&randState);

			// randomly pick a point on the circular aperture
			float angle = TWO_PI * random1;
			float distance = cudaRendercam->apertureRadius * sqrtf(random2);
			float apertureX = cos(angle) * distance;
			float apertureY = sin(angle) * distance;

			aperturePoint = rendercampos + (horizontalAxis * apertureX) + (verticalAxis * apertureY);
		}
		else { // zero aperture
			aperturePoint = rendercampos;
		}

		// calculate ray direction of next ray in path
		vec3 apertureToImagePlane = pointOnImagePlane - aperturePoint; 
		apertureToImagePlane.normalize(); // ray direction, needs to be normalised
		vec3 rayInWorldSpace = apertureToImagePlane;
		// in theory, this should not be required
		rayInWorldSpace.normalize();

		// origin of next ray in path
		vec3 originInWorldSpace = aperturePoint;

		//finalcol += path_trace(&randState, originInWorldSpace, rayInWorldSpace, -1, pTriangles, 
		//	cudaBVHindexesOrTrilists, cudaBVHlimits, cudaTriangleIntersectionData, cudaTriIdxList) * (1.0f/samps);
	
		finalcol += path_trace(&randState, originInWorldSpace, rayInWorldSpace, -1) * (1.0f / samps);
	}       

	// add pixel colour to accumulation buffer (accumulates all samples) 
	accumbuffer[i] += finalcol;
	// averaged colour: divide colour by the number of calculated frames so far
	vec3 tempcol = accumbuffer[i] / framenumber;

	Colour fcolour;
	vec3 colour = vec3(clamp(tempcol.x, 0.0f, 1.0f), clamp(tempcol.y, 0.0f, 1.0f), clamp(tempcol.z, 0.0f, 1.0f));
	// convert from 96-bit to 24-bit colour + perform gamma correction
	fcolour.components = make_uchar4((unsigned char)(powf(colour.x, 1 / 2.2f) * 255), (unsigned char)(powf(colour.y, 1 / 2.2f) * 255), (unsigned char)(powf(colour.z, 1 / 2.2f) * 255), 1);
	// store pixel coordinates and pixelcolour in OpenGL readable outputbuffer
	output[i] = vec3(x, y, fcolour.c);

}

bool g_bFirstTime = true;

// the gateway to CUDA, called from C++ (in void disp() in main.cpp)
//void cudarender(vec3* dptr, vec3* accumulatebuffer, Triangle* cudaTriangles, int* cudaBVHindexesOrTrilists,
//	float* cudaBVHlimits, float* cudaTriangleIntersectionData, int* cudaTriIdxList, 
//	unsigned framenumber, unsigned hashedframes, Camera* cudaRendercam){

//void cudarender(vec3* dptr, vec3* accumulatebuffer, Triangle* cudaTriangles, int* cudaBVHindexesOrTrilists,
//	float* cudaBVHlimits, float* cudaTriangleIntersectionData, int* cudaTriIdxList, 
//	unsigned framenumber, unsigned hashedframes, Camera* cudaRendercam){

/*
float* cuda_triangle_values;
int* cuda_triangle_indices;
float* cuda_bvh_boxes;
int* cuda_bvh_indices;
*/

//texture<float4, 1, cudaReadModeElementType> triangle_values_texture;
//texture<uint1, 1, cudaReadModeElementType> triangle_indices_texture;
//texture<float2, 1, cudaReadModeElementType> bvh_boxes_texture;
//texture<int4, 1, cudaReadModeElementType> bvh_indices_texture;

void cudarender(vec3* dptr, vec3* accumulatebuffer, float* cuda_triangle_values, int* cuda_triangle_indices, float* cuda_bvh_boxes,
int* cuda_bvh_indices, float* cuda_vertex_normals, unsigned framenumber, unsigned hashedframes, Camera* cudaRendercam)
{

	if (g_bFirstTime) {
		// if this is the first time cudarender() is called,
		// bind the scene data to CUDA textures!
		g_bFirstTime = false;

		
		cudaChannelFormatDesc channel1desc = cudaCreateChannelDesc<float4>();
		cudaBindTexture(NULL, &triangle_values_texture, cuda_triangle_values, &channel1desc, 16 * num_triangles * sizeof(float));

		cudaChannelFormatDesc channel2desc = cudaCreateChannelDesc<int>();
		cudaBindTexture(NULL, &triangle_indices_texture, cuda_triangle_indices, &channel2desc, 4 * num_triangles * sizeof(int));

		cudaChannelFormatDesc channel3desc = cudaCreateChannelDesc<float2>();
		cudaBindTexture(NULL, &bvh_boxes_texture, cuda_bvh_boxes, &channel3desc, 6 * num_nodes * sizeof(float));

		cudaChannelFormatDesc channel4desc = cudaCreateChannelDesc<int4>();
		cudaBindTexture(NULL, &bvh_indices_texture, cuda_bvh_indices, &channel4desc, 4 * num_nodes * sizeof(int));

		cudaChannelFormatDesc channel5desc = cudaCreateChannelDesc<float4>();
		cudaBindTexture(NULL, &vertex_normals_texture, cuda_vertex_normals, &channel5desc, 4 * num_vertices * sizeof(float));
	}

	dim3 block(16, 16, 1);   // dim3 CUDA specific syntax, block and grid are required to schedule CUDA threads over streaming multiprocessors
	dim3 grid(width / block.x, height / block.y, 1);

	// Configure grid and block sizes:
	int threadsPerBlock = 256;
	// Compute the number of blocks required, performing a ceiling operation to make sure there are enough:
	int fullBlocksPerGrid = ((width * height) + threadsPerBlock - 1) / threadsPerBlock;
	// <<<fullBlocksPerGrid, threadsPerBlock>>>
	
	//CoreLoopPathTracingKernel << <grid, block >> >(dptr, accumulatebuffer, cudaTriangles, cudaRendercam, cudaBVHindexesOrTrilists,
	//	cudaBVHlimits, cudaTriangleIntersectionData, cudaTriIdxList, framenumber, hashedframes);

	CoreLoopPathTracingKernel << <grid, block >> >(dptr, accumulatebuffer, cudaRendercam, framenumber, hashedframes);

}
