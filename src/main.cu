#include <float.h> // for FLT_MAX
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_timer.h>
#include "octmps.h"
#include "octmps_kernel.cuh"
#include <pthread.h>
#include "multithreading.h"
#include <time.h>

volatile int running_threads = 0;
volatile int wait_copy = 1;
pthread_mutex_t running_mutex = PTHREAD_MUTEX_INITIALIZER;

/**************************************************************
**	Compute the specular reflection.
**
**	If the first region is a turbid medium, use the Fresnel
**	reflection from the boundary of the first region as the
**	specular reflectance.
**
**	If the first region is glass, multiple reflections in
**	the first region is considered to get the specular
**	reflectance.
**
**	The subroutine assumes the Regionspecs array is correctly
**	initialized.
***************************************************************/
FLOAT Rspecular(FLOAT ni,
                FLOAT nt) {
    FLOAT r1;
    /* direct reflections from the 1st and 2nd regions. */
    FLOAT temp;
    temp = (ni - nt) / (ni + nt);
    r1 = temp * temp;
    return (r1);
}

/**************************************************************
**   Supports multiple GPUs
**   Calls RunGPU with HostThreadState parameters
***************************************************************/
static CUT_THREADPROC RunGPUi(HostThreadState *hstate) {

    checkCudaErrors(cudaSetDevice(hstate->dev_id));
    cudaError_t cudastat;

    SimState *HostMem = &(hstate->host_sim_state);

    SimState *DeviceMem;
    GPUThreadStates *tstates;

    TetrahedronStructGPU *h_root = hstate->root;
    TriangleFaces *h_faces = hstate->faces;
    UINT32 * h_rootIdx = &(hstate->rootIdx);

    int region = hstate->root[*h_rootIdx].region;

    hstate->sim->Rspecular = Rspecular(hstate->sim->regions[0].n, hstate->sim->regions[region].n);

    hstate->sim->rootIdx = *h_rootIdx;
    hstate->sim->NumFilteredPhotons = 0;
    hstate->sim->NumFilteredPhotonsClassI = 0;
    hstate->sim->NumFilteredPhotonsClassII = 0;

    FLOAT * d_probe_x, *d_probe_y, *d_probe_z;

    UINT32 * d_n_photons_left, *d_a, *d_aR, *d_aS;
    UINT64 * d_x, *d_xR, *d_xS, *d_NumClassI_PhotonsFilteredInRange, *d_NumClassII_PhotonsFilteredInRange;
    FLOAT * d_ReflectanceClassI_Sum, *d_ReflectanceClassI_Max, *d_ReflectanceClassI_SumSq,
            *d_ReflectanceClassII_Sum, *d_ReflectanceClassII_Max, *d_ReflectanceClassII_SumSq;

    int *d_NextTetrahedron, *d_NextTetrahedron_cont;

    UINT32 * d_is_active, *d_dead, *d_hit, *d_photon_region, *d_rootIdx, *d_FstBackReflectionFlag,
            *d_NumBackwardsSpecularReflections, *d_dead_cont, *d_rootIdx_cont, *d_FstBackReflectionFlag_cont;

    FLOAT * d_photon_x, *d_photon_y, *d_photon_z,
            *d_photon_x_cont, *d_photon_y_cont, *d_photon_z_cont,
            *d_photon_ux, *d_photon_uy, *d_photon_uz,
            *d_photon_ux_cont, *d_photon_uy_cont, *d_photon_uz_cont,
            *d_photon_w, *d_photon_w_cont,
            *d_photon_s, *d_photon_sleft,
            *d_photon_s_cont, *d_photon_sleft_cont,
            *d_MinCos, *d_OpticalPath, *d_MaxDepth, *d_LikelihoodRatio,
            *d_LocationFstBias, *d_MinCos_cont, *d_OpticalPath_cont, *d_MaxDepth_cont,
            *d_LikelihoodRatio_cont, *d_LocationFstBias_cont, *d_LikelihoodRatioAfterFstBias;

    UINT32 * d_NumBackwardsSpecularReflections_cont;
    TetrahedronStructGPU *d_Tetrahedron; // intermediary pointer

    // total number of threads in the grid
    UINT32 n_threads = hstate->n_tblks * NUM_THREADS_PER_BLOCK;

    UINT32 n_tetras = hstate->n_tetras;
    UINT32 n_faces = hstate->n_faces;

    unsigned int size;

    // Allocate DeviceMem pointer on deviced_
    size = n_threads * sizeof(SimState);
    checkCudaErrors(cudaMalloc((void **) &(DeviceMem), size));

    size = sizeof(FLOAT);
    checkCudaErrors(cudaMalloc((void **) &d_probe_x, size));
    checkCudaErrors(cudaMemcpy(&(DeviceMem->probe_x), &hstate->sim->probe_x, sizeof(FLOAT), cudaMemcpyHostToDevice));

    size = sizeof(FLOAT);
    checkCudaErrors(cudaMalloc((void **) &d_probe_y, size));
    checkCudaErrors(cudaMemcpy(&(DeviceMem->probe_y), &hstate->sim->probe_y, sizeof(FLOAT), cudaMemcpyHostToDevice));

    size = sizeof(FLOAT);
    checkCudaErrors(cudaMalloc((void **) &d_probe_z, size));
    checkCudaErrors(cudaMemcpy(&(DeviceMem->probe_z), &hstate->sim->probe_z, sizeof(FLOAT), cudaMemcpyHostToDevice));


    size = sizeof(UINT32);
    checkCudaErrors(cudaMalloc((void **) &d_n_photons_left, size));
    checkCudaErrors(
            cudaMemcpy(&(DeviceMem->n_photons_left), &d_n_photons_left, sizeof(UINT32 *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_n_photons_left, HostMem->n_photons_left, size, cudaMemcpyHostToDevice));

    // random number generation (on device only)
    size = n_threads * sizeof(UINT32);
    checkCudaErrors(cudaMalloc((void **) &d_a, size));
    checkCudaErrors(cudaMemcpy(&(DeviceMem->a), &d_a, sizeof(UINT32 *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_a, HostMem->a, size, cudaMemcpyHostToDevice));


    checkCudaErrors(cudaMalloc((void **) &d_aR, size));
    checkCudaErrors(cudaMemcpy(&(DeviceMem->aR), &d_aR, sizeof(UINT32 *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_aR, HostMem->aR, size, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **) &d_aS, size));
    checkCudaErrors(cudaMemcpy(&(DeviceMem->aS), &d_aS, sizeof(UINT32 *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_aS, HostMem->aS, size, cudaMemcpyHostToDevice));

    size = n_threads * sizeof(UINT64);
    checkCudaErrors(cudaMalloc((void **) &d_x, size));
    checkCudaErrors(cudaMemcpy(&(DeviceMem->x), &d_x, sizeof(UINT64 *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_x, HostMem->x, size, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **) &d_xR, size));
    checkCudaErrors(cudaMemcpy(&(DeviceMem->xR), &d_xR, sizeof(UINT64 *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_xR, HostMem->xR, size, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **) &d_xS, size));
    checkCudaErrors(cudaMemcpy(&(DeviceMem->xS), &d_xS, sizeof(UINT64 *),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_xS, HostMem->xS, size, cudaMemcpyHostToDevice));

    /********************************
    **  Values for recording matrix
    ********************************/
    HostMem->NumFilteredPhotons = 0;
    checkCudaErrors(cudaMemset(&(DeviceMem->NumFilteredPhotons), 0, sizeof(UINT64)));
    HostMem->NumFilteredPhotonsClassI = 0;
    checkCudaErrors(cudaMemset(&(DeviceMem->NumFilteredPhotonsClassI), 0, sizeof(UINT64)));
    HostMem->NumFilteredPhotonsClassII = 0;
    checkCudaErrors(cudaMemset(&(DeviceMem->NumFilteredPhotonsClassII), 0, sizeof(UINT64)));

    short STEPS = hstate->sim->NumOpticalDepthLengthSteps;
    STEPS += 1; // Following serial implementation
    size = STEPS * sizeof(UINT64);
    HostMem->NumClassI_PhotonsFilteredInRange = (UINT64 *) malloc(size);
    if (HostMem->NumClassI_PhotonsFilteredInRange == NULL) {
        printf("Error allocating HostMem->NumClassI_PhotonsFilteredInRange");
        exit(1);
    }

    checkCudaErrors(cudaMalloc((void **) &d_NumClassI_PhotonsFilteredInRange, size));
    checkCudaErrors(cudaMemcpy(&(DeviceMem->NumClassI_PhotonsFilteredInRange),
                               &d_NumClassI_PhotonsFilteredInRange,
                               sizeof(UINT64 *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_NumClassI_PhotonsFilteredInRange, 0, size));

    HostMem->NumClassII_PhotonsFilteredInRange = (UINT64 *) malloc(size);
    if (HostMem->NumClassII_PhotonsFilteredInRange == NULL) {
        printf("Error allocating HostMem->NumClassII_PhotonsFilteredInRange");
        exit(1);
    }

    checkCudaErrors(cudaMalloc((void **) &d_NumClassII_PhotonsFilteredInRange,
                               size));
    checkCudaErrors(cudaMemcpy(&(DeviceMem->NumClassII_PhotonsFilteredInRange),
                               &d_NumClassII_PhotonsFilteredInRange,
                               sizeof(UINT64 *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_NumClassII_PhotonsFilteredInRange, 0, size));

    for (UINT64 jj = 0; jj < STEPS; jj++) {
        HostMem->NumClassI_PhotonsFilteredInRange[jj] = 0;
        HostMem->NumClassII_PhotonsFilteredInRange[jj] = 0;
    }

    /********************************
    **  Recording values - Class I
    *********************************/
    STEPS += 1; // Following serial implementation
    size = STEPS * sizeof(FLOAT);
    HostMem->ReflectanceClassI_Sum = (FLOAT *) malloc(size);
    if (HostMem->ReflectanceClassI_Sum == NULL) {
        printf("Error allocating HostMem->ReflectanceClassI_Sum");
        exit(1);
    }
    for (int i = 0; i < STEPS; i++) { HostMem->ReflectanceClassI_Sum[i] = 0.0; }
    checkCudaErrors(cudaMalloc((void **) &d_ReflectanceClassI_Sum, size));
    checkCudaErrors(cudaMemcpy(&(DeviceMem->ReflectanceClassI_Sum),
                               &d_ReflectanceClassI_Sum,
                               sizeof(FLOAT *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_ReflectanceClassI_Sum, 0, size));

    HostMem->ReflectanceClassI_Max = (FLOAT *) malloc(size);
    if (HostMem->ReflectanceClassI_Max == NULL) {
        printf("Error allocating HostMem->ReflectanceClassI_Max");
        exit(1);
    }
    for (int i = 0; i < STEPS; i++) { HostMem->ReflectanceClassI_Max[i] = 0.0; }
    checkCudaErrors(cudaMalloc((void **) &d_ReflectanceClassI_Max, size));
    checkCudaErrors(cudaMemcpy(&(DeviceMem->ReflectanceClassI_Max),
                               &d_ReflectanceClassI_Max,
                               sizeof(FLOAT *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_ReflectanceClassI_Max, 0, size));

    HostMem->ReflectanceClassI_SumSq = (FLOAT *) malloc(size);
    if (HostMem->ReflectanceClassI_SumSq == NULL) {
        printf("Error allocating HostMem->ReflectanceClassI_SumSq");
        exit(1);
    }
    for (int i = 0; i < STEPS; i++) { HostMem->ReflectanceClassI_SumSq[i] = 0.0; }
    checkCudaErrors(cudaMalloc((void **) &d_ReflectanceClassI_SumSq, size));
    checkCudaErrors(cudaMemcpy(&(DeviceMem->ReflectanceClassI_SumSq),
                               &d_ReflectanceClassI_SumSq,
                               sizeof(FLOAT *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_ReflectanceClassI_SumSq, 0, size));

    /********************************
    **  Recording values - Class II
    ********************************/
    HostMem->ReflectanceClassII_Sum = (FLOAT *) malloc(size);
    if (HostMem->ReflectanceClassII_Sum == NULL) {
        printf("Error allocating HostMem->ReflectanceClassII_Sum");
        exit(1);
    }
    for (int i = 0; i < STEPS; i++) { HostMem->ReflectanceClassI_Sum[i] = 0.0; }
    checkCudaErrors(cudaMalloc((void **) &d_ReflectanceClassII_Sum, size));
    checkCudaErrors(cudaMemcpy(&(DeviceMem->ReflectanceClassII_Sum),
                               &d_ReflectanceClassII_Sum,
                               sizeof(FLOAT *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_ReflectanceClassII_Sum, 0, size));

    HostMem->ReflectanceClassII_Max = (FLOAT *) malloc(size);
    if (HostMem->ReflectanceClassII_Max == NULL) {
        printf("Error allocating HostMem->ReflectanceClassII_Max");
        exit(1);
    }
    for (int i = 0; i < STEPS; i++) { HostMem->ReflectanceClassII_Max[i] = 0.0; }
    checkCudaErrors(cudaMalloc((void **) &d_ReflectanceClassII_Max, size));
    checkCudaErrors(cudaMemcpy(&(DeviceMem->ReflectanceClassII_Max),
                               &d_ReflectanceClassII_Max,
                               sizeof(FLOAT *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_ReflectanceClassII_Max, 0, size));

    HostMem->ReflectanceClassII_SumSq = (FLOAT *) malloc(size);
    if (HostMem->ReflectanceClassII_SumSq == NULL) {
        printf("Error allocating HostMem->ReflectanceClassII_SumSq");
        exit(1);
    }
    for (int i = 0; i < STEPS; i++) { HostMem->ReflectanceClassII_SumSq[i] = 0.0; }
    checkCudaErrors(cudaMalloc((void **) &d_ReflectanceClassII_SumSq, size));
    checkCudaErrors(cudaMemcpy(&(DeviceMem->ReflectanceClassII_SumSq),
                               &d_ReflectanceClassII_SumSq,
                               sizeof(FLOAT *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_ReflectanceClassII_SumSq, 0, size));


    // Allocate tstates (photon structure) pointer on device
    size = n_threads * sizeof(GPUThreadStates);
    checkCudaErrors(cudaMalloc((void **) &(tstates), size));

    size = n_threads * sizeof(int);
    checkCudaErrors(cudaMalloc((void **) &d_NextTetrahedron, size));
    checkCudaErrors(cudaMemcpy(&(tstates->NextTetrahedron), &d_NextTetrahedron,
                               sizeof(int *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &d_NextTetrahedron_cont, size));
    checkCudaErrors(cudaMemcpy(&(tstates->NextTetrahedron_cont),
                               &d_NextTetrahedron_cont, sizeof(int *),
                               cudaMemcpyHostToDevice));

    size = n_threads * sizeof(UINT32);
    checkCudaErrors(cudaMalloc((void **) &d_is_active, size));
    checkCudaErrors(cudaMemcpy(&(tstates->is_active), &d_is_active,
                               sizeof(UINT32 *), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **) &d_dead, size));
    checkCudaErrors(cudaMemcpy(&(tstates->dead), &d_dead, sizeof(UINT32 *),
                               cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **) &d_hit, size));
    checkCudaErrors(cudaMemcpy(&(tstates->hit), &d_hit, sizeof(UINT32 *),
                               cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **) &d_photon_region, size));
    checkCudaErrors(cudaMemcpy(&(tstates->photon_region), &d_photon_region,
                               sizeof(UINT32 *), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **) &d_rootIdx, size));
    checkCudaErrors(cudaMemcpy(&(tstates->rootIdx), &d_rootIdx, sizeof(UINT32 *),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_rootIdx, h_rootIdx, size,
                               cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **) &d_FstBackReflectionFlag, size));
    checkCudaErrors(cudaMemcpy(&(tstates->FstBackReflectionFlag),
                               &d_FstBackReflectionFlag, sizeof(UINT32 *),
                               cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **) &d_NumBackwardsSpecularReflections, size));
    checkCudaErrors(cudaMemcpy(&(tstates->NumBackwardsSpecularReflections),
                               &d_NumBackwardsSpecularReflections,
                               sizeof(UINT32 *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &d_dead_cont, size));
    checkCudaErrors(cudaMemcpy(&(tstates->dead_cont), &d_dead_cont,
                               sizeof(UINT32 *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &d_rootIdx_cont, size));
    checkCudaErrors(cudaMemcpy(&(tstates->rootIdx_cont), &d_rootIdx_cont,
                               sizeof(UINT32 *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_rootIdx_cont, h_rootIdx, size,
                               cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **) &d_FstBackReflectionFlag_cont, size));
    checkCudaErrors(cudaMemcpy(&(tstates->FstBackReflectionFlag_cont),
                               &d_FstBackReflectionFlag_cont,
                               sizeof(UINT32 *), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **) &d_NumBackwardsSpecularReflections_cont,
                               size));
    checkCudaErrors(cudaMemcpy(&(tstates->NumBackwardsSpecularReflections_cont),
                               &d_NumBackwardsSpecularReflections_cont,
                               sizeof(UINT32 *), cudaMemcpyHostToDevice));

    // photon structure
    size = n_threads * sizeof(FLOAT);
    checkCudaErrors(cudaMalloc((void **) &d_photon_x, size));
    checkCudaErrors(cudaMemcpy(&(tstates->photon_x), &d_photon_x, sizeof(FLOAT *),
                               cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **) &d_photon_y, size));
    checkCudaErrors(cudaMemcpy(&(tstates->photon_y), &d_photon_y, sizeof(FLOAT *),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &d_photon_z, size));
    checkCudaErrors(cudaMemcpy(&(tstates->photon_z), &d_photon_z, sizeof(FLOAT *),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &d_photon_x_cont, size));
    checkCudaErrors(cudaMemcpy(&(tstates->photon_x_cont), &d_photon_x_cont,
                               sizeof(FLOAT *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &d_photon_y_cont, size));
    checkCudaErrors(cudaMemcpy(&(tstates->photon_y_cont), &d_photon_y_cont,
                               sizeof(FLOAT *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &d_photon_z_cont, size));
    checkCudaErrors(cudaMemcpy(&(tstates->photon_z_cont), &d_photon_z_cont,
                               sizeof(FLOAT *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &d_photon_ux, size));
    checkCudaErrors(cudaMemcpy(&(tstates->photon_ux), &d_photon_ux,
                               sizeof(FLOAT *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &d_photon_uy, size));
    checkCudaErrors(cudaMemcpy(&(tstates->photon_uy), &d_photon_uy,
                               sizeof(FLOAT *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &d_photon_uz, size));
    checkCudaErrors(cudaMemcpy(&(tstates->photon_uz), &d_photon_uz,
                               sizeof(FLOAT *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &d_photon_ux_cont, size));
    checkCudaErrors(cudaMemcpy(&(tstates->photon_ux_cont), &d_photon_ux_cont,
                               sizeof(FLOAT *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &d_photon_uy_cont, size));
    checkCudaErrors(cudaMemcpy(&(tstates->photon_uy_cont), &d_photon_uy_cont,
                               sizeof(FLOAT *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &d_photon_uz_cont, size));
    checkCudaErrors(cudaMemcpy(&(tstates->photon_uz_cont), &d_photon_uz_cont,
                               sizeof(FLOAT *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &d_photon_w, size));
    checkCudaErrors(cudaMemcpy(&(tstates->photon_w), &d_photon_w, sizeof(FLOAT *),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &d_photon_w_cont, size));
    checkCudaErrors(cudaMemcpy(&(tstates->photon_w_cont), &d_photon_w_cont,
                               sizeof(FLOAT *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &d_photon_s, size));
    checkCudaErrors(cudaMemcpy(&(tstates->photon_s), &d_photon_s, sizeof(FLOAT *),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &d_photon_sleft, size));
    checkCudaErrors(cudaMemcpy(&(tstates->photon_sleft), &d_photon_sleft,
                               sizeof(FLOAT *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &d_photon_s_cont, size));
    checkCudaErrors(cudaMemcpy(&(tstates->photon_s_cont), &d_photon_s_cont,
                               sizeof(FLOAT *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &d_photon_sleft_cont, size));
    checkCudaErrors(cudaMemcpy(&(tstates->photon_sleft_cont),
                               &d_photon_sleft_cont, sizeof(FLOAT *),
                               cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **) &d_MinCos, size));
    checkCudaErrors(cudaMemcpy(&(tstates->MinCos), &d_MinCos, sizeof(FLOAT *),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &d_OpticalPath, size));
    checkCudaErrors(cudaMemcpy(&(tstates->OpticalPath), &d_OpticalPath,
                               sizeof(FLOAT *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &d_MaxDepth, size));
    checkCudaErrors(cudaMemcpy(&(tstates->MaxDepth), &d_MaxDepth, sizeof(FLOAT *),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &d_LikelihoodRatio, size));
    checkCudaErrors(cudaMemcpy(&(tstates->LikelihoodRatio), &d_LikelihoodRatio,
                               sizeof(FLOAT *), cudaMemcpyHostToDevice));

    size = n_threads * sizeof(FLOAT);
    checkCudaErrors(cudaMalloc((void **) &d_LocationFstBias, size));
    checkCudaErrors(cudaMemcpy(&(tstates->LocationFstBias), &d_LocationFstBias,
                               sizeof(FLOAT *), cudaMemcpyHostToDevice));

    size = n_threads * sizeof(FLOAT);
    checkCudaErrors(cudaMalloc((void **) &d_MinCos_cont, size));
    checkCudaErrors(cudaMemcpy(&(tstates->MinCos_cont), &d_MinCos_cont,
                               sizeof(FLOAT *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &d_OpticalPath_cont, size));
    checkCudaErrors(cudaMemcpy(&(tstates->OpticalPath_cont), &d_OpticalPath_cont,
                               sizeof(FLOAT *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &d_MaxDepth_cont, size));
    checkCudaErrors(cudaMemcpy(&(tstates->MaxDepth_cont), &d_MaxDepth_cont,
                               sizeof(FLOAT *), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &d_LikelihoodRatio_cont, size));
    checkCudaErrors(cudaMemcpy(&(tstates->LikelihoodRatio_cont),
                               &d_LikelihoodRatio_cont, sizeof(FLOAT *),
                               cudaMemcpyHostToDevice));

    size = n_threads * sizeof(FLOAT);
    checkCudaErrors(cudaMalloc((void **) &d_LocationFstBias_cont, size));
    checkCudaErrors(cudaMemcpy(&(tstates->LocationFstBias_cont),
                               &d_LocationFstBias_cont, sizeof(FLOAT *),
                               cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **) &d_LikelihoodRatioAfterFstBias, size));
    checkCudaErrors(cudaMemcpy(&(tstates->LikelihoodRatioAfterFstBias),
                               &d_LikelihoodRatioAfterFstBias,
                               sizeof(FLOAT *), cudaMemcpyHostToDevice));

    /******************************************
    ** Copy tetrahedrons to photon structure
    ******************************************/
    size = n_tetras * sizeof(TetrahedronStructGPU);
    checkCudaErrors(cudaMalloc((void **) &d_Tetrahedron, size));
    checkCudaErrors(cudaMemcpy(&(tstates->tetrahedron), &d_Tetrahedron,
                               sizeof(TetrahedronStructGPU *),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Tetrahedron, h_root, size,
                               cudaMemcpyHostToDevice));
    /***********************************
    **  Copy face to photon structure
    ***********************************/
    size = n_faces * sizeof(TriangleFaces);
    TriangleFaces *d_TriangleFaces; // intermediary pointer
    // tstates->faces = (TriangleFaces*)malloc(size);
    checkCudaErrors(cudaMalloc((void **) &d_TriangleFaces, size));
    checkCudaErrors(cudaMemcpy(&(tstates->faces), &d_TriangleFaces,
                               sizeof(TriangleFaces *),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_TriangleFaces, h_faces, size,
                               cudaMemcpyHostToDevice));
    /******************************************************
    **  Wait for all threads to finish copying above data
    ******************************************************/
    checkCudaErrors(cudaThreadSynchronize());
    cudastat = cudaGetLastError(); // Check if there was an error
    if (cudastat) {
        fprintf(stderr, "[GPU %u] failure in InitSimStates (%i): %s\n",
                hstate->dev_id, cudastat, cudaGetErrorString(cudastat));
        FreeHostSimState(HostMem);
        FreeDeviceSimStates(DeviceMem, d_n_photons_left, d_x, d_a, d_xR, d_aR,
                            d_xS, d_aS, tstates,
                            d_NextTetrahedron, d_NextTetrahedron_cont,
                            d_is_active, d_dead, d_hit, d_photon_region, d_rootIdx,
                            d_FstBackReflectionFlag,
                            d_NumBackwardsSpecularReflections, d_dead_cont,
                            d_rootIdx_cont,
                            d_FstBackReflectionFlag_cont,
                            d_NumBackwardsSpecularReflections_cont,
                            d_photon_x, d_photon_y, d_photon_z, d_photon_ux,
                            d_photon_uy, d_photon_uz, d_photon_x_cont,
                            d_photon_y_cont, d_photon_z_cont, d_photon_ux_cont,
                            d_photon_uy_cont, d_photon_uz_cont, d_photon_w,
                            d_photon_s, d_photon_sleft, d_photon_w_cont,
                            d_photon_s_cont, d_photon_sleft_cont, d_MinCos,
                            d_OpticalPath, d_MaxDepth, d_LikelihoodRatio,
                            d_LocationFstBias,
                            d_MinCos_cont, d_OpticalPath_cont, d_MaxDepth_cont,
                            d_LikelihoodRatio_cont,
                            d_LocationFstBias_cont,
                            d_LikelihoodRatioAfterFstBias, d_Tetrahedron);
        exit(1);
    }

    InitDeviceConstantMemory(hstate->sim);
    checkCudaErrors(cudaThreadSynchronize()); // Wait for all threads to finish
    cudastat = cudaGetLastError(); // Check if there was an error
    if (cudastat) {
        fprintf(stderr, "[GPU %u] failure in InitDCMem (%i): %s\n",
                hstate->dev_id, cudastat, cudaGetErrorString(cudastat));
        FreeHostSimState(HostMem);
        FreeDeviceSimStates(DeviceMem, d_n_photons_left, d_x, d_a, d_xR, d_aR,
                            d_xS, d_aS, tstates,
                            d_NextTetrahedron, d_NextTetrahedron_cont,
                            d_is_active, d_dead, d_hit, d_photon_region, d_rootIdx,
                            d_FstBackReflectionFlag,
                            d_NumBackwardsSpecularReflections, d_dead_cont,
                            d_rootIdx_cont,
                            d_FstBackReflectionFlag_cont,
                            d_NumBackwardsSpecularReflections_cont,
                            d_photon_x, d_photon_y, d_photon_z, d_photon_ux,
                            d_photon_uy, d_photon_uz, d_photon_x_cont,
                            d_photon_y_cont, d_photon_z_cont, d_photon_ux_cont,
                            d_photon_uy_cont, d_photon_uz_cont, d_photon_w,
                            d_photon_s, d_photon_sleft, d_photon_w_cont,
                            d_photon_s_cont, d_photon_sleft_cont, d_MinCos,
                            d_OpticalPath, d_MaxDepth, d_LikelihoodRatio,
                            d_LocationFstBias,
                            d_MinCos_cont, d_OpticalPath_cont, d_MaxDepth_cont,
                            d_LikelihoodRatio_cont,
                            d_LocationFstBias_cont,
                            d_LikelihoodRatioAfterFstBias, d_Tetrahedron);
        exit(1);
    }

    dim3 dimBlock(NUM_THREADS_PER_BLOCK);
    dim3 dimGrid(hstate->n_tblks);

    // Initialize the remaining thread states
    InitThreadState << < dimGrid, dimBlock >> > (tstates, HostMem->probe_x, HostMem->probe_y, HostMem->probe_z);
    checkCudaErrors(cudaThreadSynchronize()); // Wait for all threads to finish
    cudastat = cudaGetLastError(); // Check if there was an error
    if (cudastat) {
        fprintf(stderr, "[GPU %u] failure in InitThreadState (%i): %s\n",
                hstate->dev_id, cudastat, cudaGetErrorString(cudastat));
        FreeHostSimState(HostMem);
        FreeDeviceSimStates(DeviceMem, d_n_photons_left, d_x, d_a, d_xR, d_aR,
                            d_xS, d_aS, tstates,
                            d_NextTetrahedron, d_NextTetrahedron_cont,
                            d_is_active, d_dead, d_hit, d_photon_region, d_rootIdx,
                            d_FstBackReflectionFlag,
                            d_NumBackwardsSpecularReflections, d_dead_cont,
                            d_rootIdx_cont,
                            d_FstBackReflectionFlag_cont,
                            d_NumBackwardsSpecularReflections_cont,
                            d_photon_x, d_photon_y, d_photon_z, d_photon_ux,
                            d_photon_uy, d_photon_uz, d_photon_x_cont,
                            d_photon_y_cont, d_photon_z_cont, d_photon_ux_cont,
                            d_photon_uy_cont, d_photon_uz_cont, d_photon_w,
                            d_photon_s, d_photon_sleft, d_photon_w_cont,
                            d_photon_s_cont, d_photon_sleft_cont, d_MinCos,
                            d_OpticalPath, d_MaxDepth, d_LikelihoodRatio,
                            d_LocationFstBias,
                            d_MinCos_cont, d_OpticalPath_cont, d_MaxDepth_cont,
                            d_LikelihoodRatio_cont,
                            d_LocationFstBias_cont,
                            d_LikelihoodRatioAfterFstBias, d_Tetrahedron);
        exit(1);
    }

    // Configure the L1 cache for Fermi
#ifdef USE_TRUE_CACHE

    cudaFuncSetCacheConfig(OCTMPSKernel, cudaFuncCachePreferL1);

#endif

    for (int i = 1; *HostMem->n_photons_left > 0; ++i) {
        // Run the kernel.
        OCTMPSKernel << < dimGrid, dimBlock >> > (DeviceMem, tstates);

        // Wait for all threads to finish.
        checkCudaErrors(cudaThreadSynchronize());

        // Check if there was an error
        cudastat = cudaGetLastError();
        if (cudastat) {
            fprintf(stderr, "[GPU %u] failure in OCTMPSKernel (%i): %s.\n",
                    hstate->dev_id, cudastat, cudaGetErrorString(cudastat));
            FreeHostSimState(HostMem);

            FreeDeviceSimStates(DeviceMem, d_n_photons_left, d_x, d_a, d_xR, d_aR,
                                d_xS, d_aS, tstates,
                                d_NextTetrahedron, d_NextTetrahedron_cont,
                                d_is_active, d_dead, d_hit, d_photon_region, d_rootIdx,
                                d_FstBackReflectionFlag,
                                d_NumBackwardsSpecularReflections, d_dead_cont,
                                d_rootIdx_cont,
                                d_FstBackReflectionFlag_cont,
                                d_NumBackwardsSpecularReflections_cont,
                                d_photon_x, d_photon_y, d_photon_z, d_photon_ux,
                                d_photon_uy, d_photon_uz, d_photon_x_cont,
                                d_photon_y_cont, d_photon_z_cont, d_photon_ux_cont,
                                d_photon_uy_cont, d_photon_uz_cont, d_photon_w,
                                d_photon_s, d_photon_sleft, d_photon_w_cont,
                                d_photon_s_cont, d_photon_sleft_cont, d_MinCos,
                                d_OpticalPath, d_MaxDepth, d_LikelihoodRatio,
                                d_LocationFstBias,
                                d_MinCos_cont, d_OpticalPath_cont, d_MaxDepth_cont,
                                d_LikelihoodRatio_cont,
                                d_LocationFstBias_cont,
                                d_LikelihoodRatioAfterFstBias, d_Tetrahedron);
            exit(1);
        }

        // Copy the number of photons left from device to host
        checkCudaErrors(cudaMemcpy(HostMem->n_photons_left, d_n_photons_left,
                                   sizeof(UINT32), cudaMemcpyDeviceToHost));

        printf("[GPU %u] batch %5d, number of photons left %10u\n",
               hstate->dev_id, i, *(HostMem->n_photons_left));

        if (*HostMem->n_photons_left == 0) {
            pthread_mutex_lock(&running_mutex);
            running_threads--;
            pthread_mutex_unlock(&running_mutex);
            //printf("Running threads %u",running_threads);
        }
    }

    printf("[GPU %u] simulation done!\n", hstate->dev_id);

    wait_copy =
            CopyDeviceToHostMem(HostMem, d_x, d_xR, d_xS,
                                hstate->sim, DeviceMem,
                                d_NumClassI_PhotonsFilteredInRange,
                                d_NumClassII_PhotonsFilteredInRange,
                                d_ReflectanceClassI_Sum, d_ReflectanceClassI_Max,
                                d_ReflectanceClassI_SumSq,
                                d_ReflectanceClassII_Sum, d_ReflectanceClassII_Max,
                                d_ReflectanceClassII_SumSq,
                                hstate->sim->NumOpticalDepthLengthSteps);

    FreeDeviceSimStates(DeviceMem, d_n_photons_left, d_x, d_a, d_xR, d_aR,
                        d_xS, d_aS, tstates,
                        d_NextTetrahedron, d_NextTetrahedron_cont,
                        d_is_active, d_dead, d_hit, d_photon_region, d_rootIdx,
                        d_FstBackReflectionFlag,
                        d_NumBackwardsSpecularReflections, d_dead_cont,
                        d_rootIdx_cont,
                        d_FstBackReflectionFlag_cont,
                        d_NumBackwardsSpecularReflections_cont,
                        d_photon_x, d_photon_y, d_photon_z, d_photon_ux,
                        d_photon_uy, d_photon_uz, d_photon_x_cont,
                        d_photon_y_cont, d_photon_z_cont, d_photon_ux_cont,
                        d_photon_uy_cont, d_photon_uz_cont, d_photon_w,
                        d_photon_s, d_photon_sleft, d_photon_w_cont,
                        d_photon_s_cont, d_photon_sleft_cont, d_MinCos,
                        d_OpticalPath, d_MaxDepth, d_LikelihoodRatio,
                        d_LocationFstBias,
                        d_MinCos_cont, d_OpticalPath_cont, d_MaxDepth_cont,
                        d_LikelihoodRatio_cont,
                        d_LocationFstBias_cont,
                        d_LikelihoodRatioAfterFstBias, d_Tetrahedron);
    // We still need the host-side structure.
}

/*************************************************************************
** Perform OCTMPS simulation for one run out of N runs (in the input file)
*************************************************************************/
static void DoOneSimulation(int sim_id,
                            SimulationStruct *simulation,
                            TetrahedronStructGPU *d_root,
                            TriangleFaces *d_faces,
                            int n_tetras,
                            int rootIdx,
                            HostThreadState *hstates[],
                            UINT32 num_GPUs,
                            UINT64 *x,
                            UINT32 *a,
                            UINT64 *xR,
                            UINT32 *aR,
                            UINT64 *xS,
                            UINT32 *aS) {

    printf("\n------------------------------------------------------------\n");
    printf("        Simulation #%d\n", sim_id);
    printf("        - number_of_photons = %u\n", simulation->number_of_photons);
    printf("------------------------------------------------------------\n\n");

    if (simulation->TypeSimulation == 1) {
        simulation->NumFilteredPhotons = 0;
        simulation->NumFilteredPhotonsClassI = 0;
        simulation->NumFilteredPhotonsClassII = 0;
    }

    // Distribute all photons among GPUs
    UINT32 n_photons_per_GPU = simulation->number_of_photons / num_GPUs;

    cudaDeviceProp props;
    int cc[num_GPUs];
    int num_pthreads = 0;
    for (int i = 0; i < num_GPUs; i++) {
        checkCudaErrors(cudaGetDeviceProperties(&props, hstates[i]->dev_id));
        cc[i] = ((props.major * 10 + props.minor) * 10);
        if (cc[i] != 110) num_pthreads++;
    }

    // For each GPU, init the host-side structure
    for (UINT32 i = 0; i < num_GPUs; ++i) {
        if (cc[i] != 110) {
            hstates[i]->sim = simulation;
            hstates[i]->n_tetras = n_tetras;
            hstates[i]->n_faces = 4 * n_tetras;
            hstates[i]->root = d_root;
            hstates[i]->faces = d_faces;
            hstates[i]->rootIdx = rootIdx;
            SimState *hss = &(hstates[i]->host_sim_state);
            // number of photons responsible
            hss->n_photons_left = (UINT32 *) malloc(sizeof(UINT32));
            // The last GPU may be responsible for more photons if the
            // distribution is uneven
            *(hss->n_photons_left) = (i == num_GPUs - 1) ?
                                     simulation->number_of_photons - (num_GPUs - 1) * n_photons_per_GPU :
                                     n_photons_per_GPU;
        }
    }
    // Start simulation kernel exec timer
    StopWatchInterface *execTimer = NULL;
    sdkCreateTimer(&execTimer);
    sdkStartTimer(&execTimer);

    // Launch simulation
    int failed = 0;
    // Launch a dedicated host thread for each GPU.
    pthread_t hthreads[MAX_GPU_COUNT];
    for (UINT32 i = 0; i < num_GPUs; ++i) {
        if (cc[i] != 110) {
            pthread_mutex_lock(&running_mutex);
            running_threads++;
            pthread_mutex_unlock(&running_mutex);
            failed = pthread_create(&hthreads[i], NULL, (CUT_THREADROUTINE) RunGPUi, hstates[i]);
        }
    }
    // Wait for all host threads to finish
    timespec sleepValue = {0};
    sleepValue.tv_sec = 1;
    while (running_threads > 0) {
        nanosleep(&sleepValue, NULL);
    }
    for (UINT32 i = 0; i < num_GPUs; ++i) {
        if (cc[i] != 110) {
            pthread_join(hthreads[i], NULL);
        }
    }
    // Check any of the threads failed.
    for (UINT32 i = 0; i < num_GPUs && !failed; ++i) {
        if (cc[i] != 110) {
            if (hstates[i]->host_sim_state.n_photons_left == NULL) failed = 1;
        }
    }

    // End the timer
    printf("\nSimulation completed for all GPUs, stopping timer...\n");
    int timer = sdkStopTimer(&execTimer);
    printf("Timer stopped: %u - <1:true, 0:false>\n", timer);

    float elapsedTime = sdkGetTimerValue(&execTimer);
    printf("\n\n>>>>>>Simulation time: %f (ms)\n", elapsedTime);

    if (!failed) {
        // Sum the results to hstates[0]
        SimState *hss0 = &(hstates[0]->host_sim_state);
        short STEPS = hstates[0]->sim->NumOpticalDepthLengthSteps;
        for (UINT32 i = 1; i < num_GPUs; i++) {
            SimState *hssi = &(hstates[i]->host_sim_state);
            hss0->NumFilteredPhotons += hssi->NumFilteredPhotons;
            hss0->NumFilteredPhotonsClassI += hssi->NumFilteredPhotonsClassI;
            hss0->NumFilteredPhotonsClassII += hssi->NumFilteredPhotonsClassII;

            for (int j = 0; j < STEPS; j++)
                hss0->NumClassI_PhotonsFilteredInRange[j] +=
                        hssi->NumClassI_PhotonsFilteredInRange[j];
            for (int j = 0; j < STEPS; j++)
                hss0->NumClassII_PhotonsFilteredInRange[j] +=
                        hssi->NumClassII_PhotonsFilteredInRange[j];
            for (int j = 0; j < STEPS; j++)
                hss0->ReflectanceClassI_Sum[j] += hssi->ReflectanceClassI_Sum[j];
            for (int j = 0; j < STEPS; j++)
                hss0->ReflectanceClassI_Max[j] += hssi->ReflectanceClassI_Max[j];
            for (int j = 0; j < STEPS; j++)
                hss0->ReflectanceClassI_SumSq[j] += hssi->ReflectanceClassI_SumSq[j];
            for (int j = 0; j < STEPS; j++)
                hss0->ReflectanceClassII_Sum[j] += hssi->ReflectanceClassII_Sum[j];
            for (int j = 0; j < STEPS; j++)
                hss0->ReflectanceClassII_Max[j] += hssi->ReflectanceClassII_Max[j];
            for (int j = 0; j < STEPS; j++)
                hss0->ReflectanceClassII_SumSq[j] += hssi->ReflectanceClassII_SumSq[j];
        }
        Write_Bias_Parameters(hss0, simulation, elapsedTime);

        int size = STEPS * sizeof(FLOAT);

        hss0->ReflectanceClassI = (FLOAT *) malloc(size);
        if (hss0->ReflectanceClassI == NULL) {
            printf("Error allocating hss0->ReflectanceClassI");
            exit(1);
        }
        hss0->MeanReflectanceClassI_Sum = (FLOAT *) malloc(size);
        if (hss0->MeanReflectanceClassI_Sum == NULL) {
            printf("Error allocating hss0->MeanReflectanceClassI_Sum");
            exit(1);
        }
        hss0->MeanReflectanceClassI_SumSq = (FLOAT *) malloc(size);
        if (hss0->MeanReflectanceClassI_SumSq == NULL) {
            printf("Error allocating hss0->MeanReflectanceClassI_SumSq");
            exit(1);
        }
        hss0->ReflectanceClassII = (FLOAT *) malloc(size);
        if (hss0->ReflectanceClassII == NULL) {
            printf("Error allocating hss0->ReflectanceClassII");
            exit(1);
        }
        hss0->MeanReflectanceClassII_Sum = (FLOAT *) malloc(size);
        if (hss0->MeanReflectanceClassII_Sum == NULL) {
            printf("Error allocating hss0->MeanReflectanceClassII_Sum");
            exit(1);
        }
        hss0->MeanReflectanceClassII_SumSq = (FLOAT *) malloc(size);
        if (hss0->MeanReflectanceClassII_SumSq == NULL) {
            printf("Error allocating hss0->MeanReflectanceClassII_SumSq");
            exit(1);
        }
        for (UINT64 jj = 0; jj < STEPS; jj++) {
            hss0->ReflectanceClassI[jj] = 0;
            hss0->MeanReflectanceClassI_Sum[jj] = 0;
            hss0->MeanReflectanceClassI_SumSq[jj] = 0;
            hss0->ReflectanceClassII[jj] = 0;
            hss0->MeanReflectanceClassII_Sum[jj] = 0;
            hss0->MeanReflectanceClassII_SumSq[jj] = 0;
        }
        WriteResult(simulation, hss0, &elapsedTime);
    }
    sdkDeleteTimer(&execTimer);

    // Free SimState structs
    for (UINT32 i = 0; i < num_GPUs; ++i) {
        if (cc[i] != 110) {
            FreeHostSimState(&(hstates[i]->host_sim_state));
        }
    }
//  FreeHostSimState(hss);
//  free(hstates);
}

int main(int argc, char *argv[]) {

    ShowVersion("Version 1, 2016");
    char *filename_opt = NULL;
    char *filename_mesh = NULL;
    char *filename_bias = NULL;

    unsigned long long seed = SEED0; //(unsigned long long) time(NULL); //

    UINT32 num_GPUs = 1;

    SimulationStruct *simulations;
    int n_simulations;

    int number_of_vertices = 0;
    int number_of_faces = 0;
    int number_of_tetrahedrons = 0;
    int mesh_param; // check vertices and no. of tetrahedrons
    int bias_flag;  // to check for errors in bias reading

    // Parse command-line arguments.
    if (interpret_arg(argc, argv, &filename_opt, &filename_mesh, &filename_bias, &seed, &num_GPUs)) {
        usage(argv[0]);
        return 1;
    }

    // Determine the number of GPUs available
    int dev_count;
    checkCudaErrors(cudaGetDeviceCount(&dev_count));
    if (dev_count <= 0) {
        fprintf(stderr, "No GPU available. Quit.\n");
        return 1;
    }

    // Make sure we do not use more than what we have.
    if (num_GPUs > dev_count) {
        printf("The number of GPUs available is (%d).\n", dev_count);
        num_GPUs = (UINT32) dev_count;
    }

    // Output the execution configuration
    printf("\n====================================\n");
    printf("EXECUTION MODE:\n");
    printf("  seed#1 Step + Roulette:  %llu\n", seed);
    printf("  seed#2 ReflectTransmit:  %lu\n", SEED1);
    printf("  seed#3 Spin:             %lu\n", SEED2);
    printf("  # of GPUs:               %u\n", num_GPUs);
    printf("====================================\n\n");

    /*********************************
    ** Read the simulation inputs.
    *********************************/
    n_simulations = read_simulation_data(filename_opt, &simulations);
    if (n_simulations == 0) {
        printf("Something wrong with read_simulation_data!\n");
        return 1;
    }
    printf("Read %d simulations\n", n_simulations);

    /*********************************
    ** Read the simulation inputs.
    *********************************/

    printf("Reading bias parameters.\n\n");
    bias_flag = Bias_Read(filename_bias, &simulations);
    if (bias_flag == 0) {
        printf("Something wrong with the bias reading, check file!\n");
        return 1;
    }

    mesh_param = read_mesh_param(filename_mesh, &number_of_vertices,
                                 &number_of_tetrahedrons);
    if (mesh_param == 0) {
        printf("Something wrong with the meshing parameters!\n");
        return 1;
    }

    printf("\n====================================\n");
    printf("Mesh Parameters:\n");
    printf("No. of nodes=%d\n", number_of_vertices);
    printf("No. of tetrahedrons=%d\n", number_of_tetrahedrons);
    number_of_faces = 4 * number_of_tetrahedrons;
    printf("No. of faces=%d\n", number_of_faces);
    printf("====================================\n");

    TetrahedronStruct *Graph;
    TetrahedronStruct *Root;
    Vertex *d_vertices = new Vertex[number_of_vertices];
    TriangleFaces *d_faces = new TriangleFaces[number_of_faces];
    TetrahedronStructGPU *d_root = new TetrahedronStructGPU[number_of_tetrahedrons];;

    printf("Constructing mesh graph. Please wait...\n");
    clock_t start = clock(), diff;
    Graph = MeshGraph(filename_mesh, &simulations);
    diff = clock() - start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("Mesh graph is created in %d seconds %d milliseconds.\n", msec / 1000, msec % 1000);

    for (simulations->probe.current_scanning_x = simulations->probe.start_x;
         simulations->probe.current_scanning_x < simulations->probe.end_x;
         simulations->probe.current_scanning_x += simulations->probe.distance_Ascans) {

        // probe locations
        simulations->probe_x = simulations->probe.current_scanning_x;
        simulations->probe_y = 0;
        simulations->probe_z = 0;

        printf("Finding the root of the graph...\n");

        Root = FindRootofGraph(Graph, &simulations);
        UINT32 rootIdx = Root[0].index;

        printf("Serializing the graph...\n");
        SerializeGraph(Root, number_of_vertices, number_of_faces, number_of_tetrahedrons, d_vertices, d_faces, d_root);

        // Allocate one host thread state for each GPU
        HostThreadState *hstates[MAX_GPU_COUNT];
        cudaDeviceProp props;
        int n_threads = 0;    // total number of threads for all GPUs
        int cc[num_GPUs];
        for (int i = 0; i < num_GPUs; ++i) {
            hstates[i] = (HostThreadState *) malloc(sizeof(HostThreadState));
            // Set the GPU ID
            hstates[i]->dev_id = i;
            // Get the GPU properties
            checkCudaErrors(cudaGetDeviceProperties(&props, hstates[i]->dev_id));
            // Validate the GPU compute capability
            cc[i] = (props.major * 10 + props.minor) * 10;
            if (cc[i] >= __CUDA_ARCH__) {
                printf("[GPU %u] \"%s\" with Compute Capability %d.%d (%d SMs)\n",
                       i, props.name, props.major, props.minor,
                       props.multiProcessorCount);
                // We launch one thread block for each SM on this GPU
                hstates[i]->n_tblks = props.multiProcessorCount;

                n_threads += hstates[i]->n_tblks * NUM_THREADS_PER_BLOCK;
            } else {
                fprintf(stderr, "\n[GPU %u] \"%s\" with Compute Capability %d.%d,"
                                "\ndoes not meet the minimum requirement (1.3) for this program! "
                                "\nExcluding [GPU %d].\n\n", i, props.name, props.major,
                        props.minor, i);
                //exit(1);
            }
        }

        // Allocate and initialize RNG seeds (General, ReflectT, Spin)
        UINT64 * x = (UINT64 *) malloc(n_threads * sizeof(UINT64));
        UINT32 * a = (UINT32 *) malloc(n_threads * sizeof(UINT32));

        UINT64 * xR = (UINT64 *) malloc(n_threads * sizeof(UINT64));
        UINT32 * aR = (UINT32 *) malloc(n_threads * sizeof(UINT32));

        UINT64 * xS = (UINT64 *) malloc(n_threads * sizeof(UINT64));
        UINT32 * aS = (UINT32 *) malloc(n_threads * sizeof(UINT32));

#ifdef _WIN32
        if (init_RNG(x,a,xR,aR,xS,aS,n_threads,"safeprimes_base32.txt",seed))
    return 1;
#else
        if (init_RNG(x, a, xR, aR, xS, aS, n_threads,
                     "safeprimes_base32.txt", seed))
            return 1;
#endif

        printf("\nUsing MWC random number generator ...\n");

        // Assign these seeds to each host thread state
        int ofst = 0;
        for (int i = 0; i < num_GPUs; ++i) {
            if (cc[i] != 110) {
                SimState *hss = &(hstates[i]->host_sim_state);
                hss->x = &x[ofst];
                hss->a = &a[ofst];
                hss->xR = &xR[ofst];
                hss->aR = &aR[ofst];
                hss->xS = &xS[ofst];
                hss->aS = &aS[ofst];
                ofst += hstates[i]->n_tblks * NUM_THREADS_PER_BLOCK;
            }
        }

        //perform all the simulations
        for (int i = 0; i < n_simulations; i++) {
            // Run a simulation
            DoOneSimulation(i, &simulations[i], d_root, d_faces,
                            number_of_tetrahedrons, rootIdx, hstates, num_GPUs,
                            x, a, xR, aR, xS, aS);
        }

        // Free host thread states.
        for (int i = 0; i < num_GPUs; ++i) {
            free(hstates[i]);
        }

        // Free the random number seed arrays
        free(x);
        free(a);
        free(xR);
        free(aR);
        free(xS);
        free(aS);
    }

    FreeSimulationStruct(simulations, n_simulations);
    cudaDeviceReset();
    return 0;
}
