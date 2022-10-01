#ifndef _BBOX_H_
#define _BBOX_H_

#include "vec3.h"


static float maxf(const float& a, const float& b)
{
	return a > b ? a : b;
}

static float minf(const float& a, const float& b)
{
	return a < b ? a : b;
}


struct BBox
{
	vec3 bbox[2];

	BBox() { bbox[0] = { 1e10f, 1e10f, 1e10f }, bbox[1] = { -1e10f, -1e10f, -1e10f }; }
	BBox(vec3 bmin, vec3 bmax) { bbox[0] = bmin; bbox[1] = bmax; }


	vec3 c() { return (bbox[0] + bbox[1]) * 0.5f; }


	void expand(const BBox& box)
	{
		bbox[0] = vec3(minf(bbox[0].x, box.bbox[0].x), minf(bbox[0].y, box.bbox[0].y), minf(bbox[0].z, box.bbox[0].z));
		bbox[1] = vec3(maxf(bbox[1].x, box.bbox[1].x), maxf(bbox[1].y, box.bbox[1].y), maxf(bbox[1].z, box.bbox[1].z));
	}

	BBox expand_box(const BBox& box)
	{
		vec3 v1(minf(bbox[0].x, box.bbox[0].x), minf(bbox[0].y, box.bbox[0].y), minf(bbox[0].z, box.bbox[0].z));
		vec3 v2(maxf(bbox[1].x, box.bbox[1].x), maxf(bbox[1].y, box.bbox[1].y), maxf(bbox[1].z, box.bbox[1].z));

		return BBox(v1, v2);
	}

	void expand(const vec3& p)
	{

		bbox[0] = vec3(minf(bbox[0].x, p.x), minf(bbox[0].y, p.y), minf(bbox[0].z, p.z));
		bbox[1] = vec3(maxf(bbox[1].x, p.x), maxf(bbox[1].y, p.y), maxf(bbox[1].z, p.z));
	}

	int maxDim()
	{
		vec3 extend(bbox[1] - bbox[0]);
		if (extend.x > extend.y && extend.x > extend.z) return 0;
		else if (extend.y > extend.z) return 1;
		return 2;

	}
	int minDim()
	{
		vec3 extend(bbox[1] - bbox[0]);
		if (extend.x < extend.y && extend.x < extend.z) return 0;
		else if (extend.y < extend.z) return 1;
		return 2;
	}

	float area()
	{
		vec3 extend(bbox[1] - bbox[0]);
		return (extend.x * extend.y + extend.y * extend.z + extend.z * extend.x);
	}
};

#endif // ! _BBOX_H_

