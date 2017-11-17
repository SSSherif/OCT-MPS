#ifndef _GPUOCTMPS_KERNEL_H_
#define _GPUOCTMPS_KERNEL_H_

#include "octmps.h"

//SEED1  3858638025 - 1400969702
#define SEED0 985456376
#define SEED1 3858638025
#define SEED2 2658951225

/*
 * CUDA Double Precision Divisions:
 * __device__ double 	__ddiv_rd (double x, double y)
 *	Divide two floating point values in round-down mode.
 * __device__ double 	__ddiv_rn (double x, double y)
 *	Divide two floating point values in round-to-nearest-even mode.
 * __device__ double 	__ddiv_ru (double x, double y)
 *	Divide two floating point values in round-up mode.
 * __device__ double 	__ddiv_rz (double x, double y)
 *	Divide two floating point values in round-towards-zero mode.
 *
 * CUDA Single Precision Divisions:
 * __DEVICE_FUNCTIONS_DECL__ float __fdiv_rd ( float  x, float  y )
 *  Divide two floating point values in round-down mode.
 * __DEVICE_FUNCTIONS_DECL__ float __fdiv_rn ( float  x, float  y )
 *  Divide two floating point values in round-to-nearest-even mode.
 * __DEVICE_FUNCTIONS_DECL__ float __fdiv_ru ( float  x, float  y )
 * Divide two floating point values in round-up mode.
 * __DEVICE_FUNCTIONS_DECL__ float __fdiv_rz ( float  x, float  y )
 * Divide two floating point values in round-towards-zero mode.
 * __DEVICE_FUNCTIONS_DECL__ float __fdividef ( float  x, float  y )
 * Calculate the fast approximate division of the input arguments.
 *
 *
 * Read more at: http://docs.nvidia.com/cuda/cuda-math-api/index.html
 */

#define FAST_DIV(x, y) __ddiv_rd(x,y) // __fdividef(x,y) // ((x)/(y))
#define ACCURATE_DIV(x, y) exp(log(x)-log(y))
#define SQRT(x) sqrt(x) // sqrtf(x)
#define RSQRT(x) rsqrt(x) // rsqrtf(x)
#define LOG(x) log(x) // logf(x)
#define SINCOS(x, sptr, cptr) sincos(x, sptr, cptr) // __sincosf(x, sptr, cptr)
#define SQ(x) ( (x)*(x) )
#define CUBE(x) ( (x)*(x)*(x) )
#define FAST_MIN(x, y) fmin(x, y)
#define FAST_MAX(x, y) fmax(x, y)
#define PARTIAL_REFLECTION 1

/*
** Fused fused multiply-add
** __host__ __device__ double fma ( double  x, double  y, double  z )
** Compute x × y + z
*/

/*  Number of simulation steps performed by each thread in one kernel call */
#define NUM_STEPS 50000  //Use 5000 for faster response time

#define MAX_GPU_COUNT 16 //mrei 4x Teslas C1060

#define __CUDA_ARCH__ 350 //mrei
// Make sure __CUDA_ARCH__ is always defined by the user.
#ifdef _WIN32
#define __CUDA_ARCH__ 120
#endif

#ifndef __CUDA_ARCH__
#error "__CUDA_ARCH__ undefined!"
#endif

/***************************
** Compute Capability 3.5
***************************/
#if __CUDA_ARCH__ == 350
#define NUM_THREADS_PER_BLOCK 600
// #define EMULATED_ATOMIC
#define USE_TRUE_CACHE
#define FMA(x, y, z) fma(x,y,z)
/***************************
** Compute Capability 2.0
***************************/
#elif __CUDA_ARCH__ == 200
#define NUM_THREADS_PER_BLOCK 512
// #define EMULATED_ATOMIC
#define USE_TRUE_CACHE
#define FMA(x,y,z) ((x*y)+z)
/******************************************
**  Compute Capability 1.2 or 1.3  30x256
******************************************/
#elif (__CUDA_ARCH__ == 120) || (__CUDA_ARCH__ == 130)
#define NUM_THREADS_PER_BLOCK 128
#define EMULATED_ATOMIC
#define FMA(x,y,z) ((x*y)+z)
/***************************
**  Compute Capability 1.1
***************************/
#elif (__CUDA_ARCH__ == 110)
#define NUM_THREADS_PER_BLOCK 192
#define EMULATED_ATOMIC
#define FMA(x,y,z) ((x*y)+z)
/************************************
**  Unsupported Compute Capability
************************************/
#else
#error "The compute compatibility is not supported!"

#endif

// The max number of regions supported (MAX_REGIONS including 1 ambient region)
#define MAX_REGIONS 100

typedef struct __align__(16){

    FLOAT Rspecular;          // Specular Reflectance
    FLOAT TargetDepthMin;
    FLOAT TargetDepthMax;
    FLOAT BackwardBiasCoefficient;
    FLOAT rndStepSizeInTissue;
    FLOAT MaxCollectingAngleDeg;
    FLOAT MaxCollectingRadius;
    FLOAT ProbabilityAdditionalBias;
    FLOAT OpticalDepthShift;
    FLOAT CoherenceLengthSource;
    UINT64 NumOpticalDepthLengthSteps;
    UINT32 num_regions;        // number of regions
    UINT32 rootIdx;            // Root Tetrahedron Index
    short int TypeBias;

}
SimParamGPU;

typedef struct __align__(16) {

    FLOAT n;                  // refractive index of a region
    FLOAT muas;               // mua + mus
    FLOAT rmuas;              // 1/(mua+mus) = mutr = 1 / mua+mus
    FLOAT mua_muas;           // mua/(mua+mus)
    FLOAT g;                  // anisotropy
}
RegionStructGPU;

__constant__ SimParamGPU d_simparam;
__constant__ RegionStructGPU d_regionspecs[MAX_REGIONS];

/*************************************************************************
** Thread-private states that live across batches of kernel invocations
** Each field is an array of length NUM_THREADS.
**
** We use a struct of arrays as opposed to an array of structs to enable
** global memory coalescing.
*************************************************************************/
typedef struct {
    int *NextTetrahedron;
    int *NextTetrahedron_cont;

    UINT32 *is_active; // is this thread active?

    // From TMCv2
    UINT32 *dead;
    UINT32 *hit;

    // index to region where the photon resides
    UINT32 *photon_region;
    UINT32 *rootIdx;
    UINT32 *FstBackReflectionFlag;
    UINT32 *NumBackwardsSpecularReflections;

    UINT32 *dead_cont;
    UINT32 *rootIdx_cont;

    UINT32 *FstBackReflectionFlag_cont;
    UINT32 *NumBackwardsSpecularReflections_cont;

    // Cartesian coordinates of the photon [cm]
    FLOAT *photon_x;
    FLOAT *photon_y;
    FLOAT *photon_z;

    FLOAT *photon_x_cont;
    FLOAT *photon_y_cont;
    FLOAT *photon_z_cont;

    // directional cosines of the photon
    FLOAT *photon_ux;
    FLOAT *photon_uy;
    FLOAT *photon_uz;

    FLOAT *photon_ux_cont;
    FLOAT *photon_uy_cont;
    FLOAT *photon_uz_cont;

    FLOAT *photon_w;          // photon weight
    FLOAT *photon_s;          // photon step size
    FLOAT *photon_sleft;      // leftover step size [cm]

    FLOAT *photon_w_cont;     // photon weight
    FLOAT *photon_s_cont;     // photon step size
    FLOAT *photon_sleft_cont; // leftover step size [cm]

    // More from TMCv2
    FLOAT *MinCos;
    FLOAT *OpticalPath;
    FLOAT *MaxDepth;
    FLOAT *LikelihoodRatio;
    FLOAT *LocationFstBias;

    FLOAT *MinCos_cont;
    FLOAT *OpticalPath_cont;
    FLOAT *MaxDepth_cont;
    FLOAT *LikelihoodRatio_cont;
    FLOAT *LocationFstBias_cont;

    FLOAT *LikelihoodRatioAfterFstBias;

    TetrahedronStructGPU *tetrahedron;
    TriangleFaces *faces;
} GPUThreadStates;

typedef struct {
    int NextTetrahedron;

    // flag to indicate if photon hits a boundary
    UINT32 hit;

    UINT32 dead;
    UINT32 FstBackReflectionFlag;

    UINT32 rootIdx;
    UINT32 NumBackwardsSpecularReflections;

    // Cartesian coordinates of the photon [cm]
    FLOAT x;
    FLOAT y;
    FLOAT z;

    // directional cosines of the photon
    FLOAT ux;
    FLOAT uy;
    FLOAT uz;

    FLOAT w;            // photon weight

    FLOAT s;            // step size [cm]
    FLOAT sleft;        // leftover step size [cm]

    FLOAT MinCos;
    FLOAT LocationFstBias;
    FLOAT OpticalPath;
    FLOAT MaxDepth;
    FLOAT LikelihoodRatio;
    FLOAT LikelihoodRatioAfterFstBias;

    TetrahedronStructGPU *tetrahedron;
    TriangleFaces *faces;
} PhotonStructGPU;

#endif // _GPUOCTMPS_KERNEL_H_
