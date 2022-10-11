#ifndef _vec3_H_
#define _vec3_H_

#include <cuda_runtime.h>
#include <math.h>


#define PI 3.1415926f
#define ipi 1.0f / PI
#define i180 1.0f / 180.0f
#define pi_over_2 PI * 0.5f
#define pi2 2 * PI
#define half_pi PI * 0.5f
#define maxs(x, y) x > y ? x : y
#define mins(x, y) x < y ? x : y

inline float maxf(float& x, float& y)
{
	return x > y ? x : y;
}

inline float minf(float& x, float& y)
{
	return x < y ? x : y;
}

struct vec3
{
	__host__ __device__ vec3() : x(0), y(0), z(0) {}
	__host__ __device__ vec3(float v_) : x(v_), y(v_), z(v_) {}
	__host__ __device__ vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

	union
	{
		struct
		{
			float x, y, z;
		};

		float v[3];
	};

	__host__ __device__ vec3 norm()
	{
		float l = 1.0f / sqrtf(x*x + y*y + z*z); return{ x * l, y * l, z * l }; //return *this * l;
	}

	__host__ __device__ void normalize()
	{
		float l = 1.0f / sqrtf(x*x + y*y + z*z); x *= l, y *= l, z *= l;
	}

	//__host__ __device__ float dot(const vec3& v) const { return x * v.x + y * v.y + z * v.z; }


	__host__ __device__ vec3 cross(const vec3& b) const
	{
		float bx = b.x, by = b.y, bz = b.z;
		return  vec3(y * bz - z * by, z * bx - x * bz, x * by - y * bx);
	}
	__host__ __device__ float length() const { return sqrtf(x * x + y * y + z * z); }
	__host__ __device__ float length2() const { return x * x + y * y + z * z; }

	__host__ __device__ float operator[](const int& i) const { return (&x)[i]; }


	__host__ __device__ friend vec3 operator+(vec3& a, vec3& b)
	{
		return{ a.x + b.x, a.y + b.y, a.z + b.z };
	}
	__host__ __device__ friend vec3 operator-(vec3& a, vec3 b)
	{

		return{ a.x - b.x, a.y - b.y, a.z - b.z };
	}
	__host__ __device__ friend float4 operator-(vec3& a, float4& b)
	{

		return{ a.x - b.x, a.y - b.y, a.z - b.z, 0.0f };
	}
	__host__ __device__ friend vec3 operator*(vec3& a, vec3& b)
	{

		return{ a.x * b.x, a.y * b.y, a.z * b.z };
	}
	__host__ __device__ friend vec3 operator*(vec3& a, float v)
	{

		return{ a.x * v, a.y * v, a.z * v };
	}
	__host__ __device__ friend vec3 operator*(float v, vec3& a)
	{

		return{ a.x * v, a.y * v, a.z * v };
	}
	//__host__ __device__ friend vec3 operator*=(vec3& a, float& v)
	//{
	//	return{ a.x * v, a.y * v, a.z * v };
	//}

	//__host__ __device__ vec3 operator*(float v) { return{ x * v, y * v, z * v }; }

	//__host__ __device__ friend vec3  operator*(vec3& a, float& v) { return{ a.x * v, a.y * v, a.z * v }; }
	//__host__ __device__ friend vec3  operator*(float& v, vec3& a) { return{ a.x * v, a.y * v, a.z * v }; }
	__host__ __device__ vec3 operator+=(vec3& v) { x += v.x; y += v.y; z += v.z;  return *this; }
	__host__ __device__ vec3 operator-=(vec3& v) { x -= v.x; y -= v.y; z -= v.z;  return *this; }
	__host__ __device__ vec3 operator*=(const float& value) { x *= value; y *= value; z *= value; return *this; }
	__host__ __device__ vec3 operator*=(const vec3& value) { x *= value.x; y *= value.y; z *= value.z; return *this; }

	__host__ __device__ float maxc() const { float d = maxs(x, y); return maxs(d, z); }//{ return max(max(x, y), z); }//
	__host__ __device__ float minc() const { float d = mins(x, y); return mins(d, z); }//{ return min(min(x, y), z); }//

	__host__ __device__ float average()
	{
		return (x + y + z) / 3.0f;
	}

	__host__ __device__ friend vec3 operator/(const vec3& a, const vec3& b) { return{ a.x / b.x, a.y / b.y, a.z / b.z }; }
	__host__ __device__ friend vec3 operator/=(const vec3& a, const float& v) { return{ a.x / v, a.y / v, a.z / v }; }
	__host__ __device__ friend vec3 operator-(const vec3& a) { return{ -a.x, -a.y, -a.z }; }

	__host__ __device__ friend vec3 operator/(const vec3& a, const float& v) { return{ a.x / v, a.y / v, a.z / v }; }


	__host__ __device__ bool all_zero()
	{
		return x == y == z == 0.0f;
	}
};

inline __host__ __device__ vec3 min3(const vec3& v1, const vec3& v2) { return vec3(v1.x < v2.x ? v1.x : v2.x, v1.y < v2.y ? v1.y : v2.y, v1.z < v2.z ? v1.z : v2.z); }
inline __host__ __device__ vec3 max3(const vec3& v1, const vec3& v2) { return vec3(v1.x > v2.x ? v1.x : v2.x, v1.y > v2.y ? v1.y : v2.y, v1.z > v2.z ? v1.z : v2.z); }


inline __host__ __device__ float dot(const vec3& a, const vec3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float dot(const vec3& a, const float4& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float dot(const float3& a, const float4& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float dot(const float4& a, const vec3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float dot(const float4& a, const float3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

//inline __host__ __device__ float dot(const float4& a, const float4& b)
//{
//	return a.x * b.x + a.y * b.y + a.z * b.z;
//}

inline __host__ __device__ vec3 cross(vec3& a, vec3& b)
{
	return{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}

inline __host__ __device__ float4 cross(float4& a, float4& b)
{
	return{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x, 0.0f };
}

inline __host__ __device__ float4 cross(vec3& a, float4& b)
{
	return{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x, 0.0f };
}

inline __host__ __device__ float4 cross(float3& a, float4& b)
{
	return{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x, 0.0f };
}

inline __host__ __device__ float4 cross(float4& a, vec3& b)
{
	return{ a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x, 0.0f };
}

inline __host__ __device__ float distancesq(vec3& a, vec3& b)
{
	return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z);
}



#endif // !_vec3_H_

