#ifndef __CAMERA_H__
#define __CAMERA_H__

//#include "geometry.h"
#include "vec3.h"
#include <cuda_runtime.h>

#define M_PI 3.14156265
#define PI_OVER_TWO 1.5707963267948966192313216916397514420985

// Camera struct, used to store interactive camera data, copied to the GPU and used by CUDA for each frame
struct Camera {
	float2 resolution;
	vec3 position;
	vec3 view;
	vec3 up;
	float2 fov;
	float apertureRadius;
	float focalDistance;
};

// class for interactive camera object, updated on the CPU for each frame and copied into Camera struct
class InteractiveCamera
{
private:

	vec3 centerPosition;
	vec3 viewDirection;
	float yaw;
	float pitch;
	float radius;
	float apertureRadius;
	float focalDistance;

	void fixYaw();
	void fixPitch();
	void fixRadius();
	void fixApertureRadius();
	void fixFocalDistance();

public:
	InteractiveCamera();
	virtual ~InteractiveCamera();
   	void changeYaw(float m);
	void changePitch(float m);
	void changeRadius(float m);
	void changeAltitude(float m);
	void changeFocalDistance(float m);
	void strafe(float m);
	void goForward(float m);
	void rotateRight(float m);
	void changeApertureDiameter(float m);
	void setResolution(float x, float y);
	void setFOVX(float fovx);

	void buildRenderCamera(Camera* renderCamera);

	float2 resolution;
	float2 fov;
};



#endif
