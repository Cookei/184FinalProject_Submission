#include <ai.h>
#include <string.h>
#define PI 3.14159265358979323846

AI_CAMERA_NODE_EXPORT_METHODS(MyCameraMethods)

enum
{
	p_fov
};

struct MyCameraData
{
	float tan_fov;
	float fov;
	float focalDistance;
	float apertureSize;
	float sph;
	float cyl;
	float axis;
	float xres;
	float yres;
	float aspectRatio;
	bool paperSampling;
	AtMatrix c2w;
};

node_parameters
{
	AiParameterFlt("fov", 60.0f);
	AiParameterFlt("focus_distance", 86.7139206f);
	AiParameterFlt("aperture_size", 2.00f);
	AiParameterFlt("sph", 1.5f);
	AiParameterFlt("cyl", -2.5f);
	AiParameterFlt("axis", 180.0f);
	AiParameterFlt("xres", 1920.0);
	AiParameterFlt("yres", 1080.0);
	AiParameterBool("paper_sampling", false);
}

node_initialize
{
   AiCameraInitialize(node);
   AiNodeSetLocalData(node, new MyCameraData());
}

node_update
{
	MyCameraData * data = (MyCameraData*)AiNodeGetLocalData(node);
   data->tan_fov = tanf(AiNodeGetFlt(node, AtString("fov")) * AI_DTOR / 2);
   data->focalDistance = AiNodeGetFlt(node, AtString("focus_distance"));
   data->apertureSize = AiNodeGetFlt(node, AtString("aperture_size"));
   data->fov = AiNodeGetFlt(node, AtString("fov"));
   data->sph = AiNodeGetFlt(node, AtString("sph"));
   data->cyl = AiNodeGetFlt(node, AtString("cyl"));
   data->axis = AiNodeGetFlt(node, AtString("axis"));
   data->xres = AiNodeGetFlt(node, AtString("xres"));
   data->yres = AiNodeGetFlt(node, AtString("yres"));
   data->paperSampling = AiNodeGetBool(node, AtString("paper_sampling"));
   data->aspectRatio = data->xres / data->yres;
   AiCameraToWorldMatrix(node, 1, data->c2w);
   AiCameraUpdate(node, false);
}

node_finish
{
   MyCameraData * data = (MyCameraData*)AiNodeGetLocalData(node);
   delete data;
}

camera_create_ray
{
   const MyCameraData * data = (MyCameraData*)AiNodeGetLocalData(node);
//const AtVector p(input.sx* data->tan_fov, input.sy* data->tan_fov, 1);
//// warp ray origin with a noise vector
//AtVector noise_point(input.sx, input.sy, 0.5f);
//noise_point *= 5;
//AtVector noise_vector = AiVNoise3(noise_point, 1, 0.f, 1.92f);
//output.origin = noise_vector * 0.04f;
//output.dir = AiV3Normalize(p - output.origin);

//// vignetting
//const float dist2 = input.sx * input.sx + input.sy * input.sy;
//output.weight = 1 - dist2;

//// now looking down -Z
//output.dir.z *= -1

//sampling
double r, phi;
if (data->paperSampling) {
	double DistantIndex = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	double AngleFactor = 20;
	double AngleIndex = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * AngleFactor;


	r = DistantIndex;
	phi = 2 * PI / AngleFactor * (AngleIndex + r);
}
else {
	r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	r = sqrt(r);
	phi = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * 2 * PI;
}

double hFov = data->fov;
double vFov = data->fov;
//generate ray
double x_sensor_cam = input.sx * tan(hFov * AI_DTOR * 0.5);
double y_sensor_cam = input.sy * tan(vFov * AI_DTOR * 0.5);
AtVector pSensor_cam = AtVector(x_sensor_cam, y_sensor_cam, -1.0); // location of sensor plane
AtVector pLens_cam = AtVector(data->apertureSize * r * cos(phi), data->apertureSize * r * sin(phi), 0.0);


//double angle = PI / 2.0;  // Ping-pong peak
//double period = PI;  // Full cycle: 0 -> PI/2 -> 0
//double t = fmod(phi + data->axis * AI_DTOR, period);  // Wrap around full cycle
//
//double phase;
//if (t < angle) {
//	phase = t / angle;  // Rising edge
//}
//else {
//	phase = (period - t) / (period - angle);  // Falling edge (reflection)
//}
//double eyePower = data->sph + data->cyl * phase;
double eyePower = data->sph + data->cyl * pow(sin(phi + data->axis * AI_DTOR), 2);
//double f = 1 / (1 / data->focalDistance + eyePower);
double f = eyePower != 0 ? data->focalDistance + 1 / (eyePower) : data->focalDistance;

AtVector pFocus_cam = pSensor_cam * f;

AtVector dir = AiV3Normalize(pFocus_cam - pLens_cam);

output.origin = pLens_cam;
output.dir = dir;
}

camera_reverse_ray
{
   const MyCameraData * data = (MyCameraData*)AiNodeGetLocalData(node);

// Note: we ignore distortion to compute the screen projection
// compute projection factor: avoid divide by zero and flips when crossing the camera plane
float coeff = 1 / AiMax(fabsf(Po.z * data->tan_fov), 1e-3f);
Ps.x = Po.x * coeff;
Ps.y = Po.y * coeff;
return true;
}

node_loader
{
   if (i != 0) return false;
   node->methods = MyCameraMethods;
   node->output_type = AI_TYPE_UNDEFINED;
   node->name = "mycamera";
   node->node_type = AI_NODE_CAMERA;
   strcpy(node->version, AI_VERSION);
   return true;
}