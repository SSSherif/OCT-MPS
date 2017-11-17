#ifndef _OCTMPS_KERNEL_CUH_
#define _OCTMPS_KERNEL_CUH_

#include "octmps_kernel.h"
#include "octmps.h"

/****************************************************
 **  Generates a random number between 0 and 1 [0,1)
 ****************************************************/
__device__ FLOAT rand_MWC_co(UINT64* x, UINT32* a) {
	//#define SEED0 985456376
	//#define SEED1 3858638025
	//#define SEED2 2658951225
	//

	*x = (*x & 0xffffffffull) * (*a) + (*x >> 32);
	return FAST_DIV(__uint2float_rz((UINT32 )(*x)), (FLOAT )0x100000000);
	// The typecast will truncate the x so that it is 0<=x<(2^32-1),
	// __uint2FLOAT_rz ensures a round towards zero since 32-bit FLOATing
	// point cannot represent all integers that large.
	// Dividing by 2^32 will hence yield [0,1)
}

/****************************************************
 **  Generates a random number between 0 and 1 (0,1]
 ****************************************************/
__device__ FLOAT rand_MWC_oc(UINT64* x, UINT32* a) {
	return FP_ONE - rand_MWC_co(x, a);
}

__device__ FLOAT AtomicAddD(FLOAT* address, FLOAT value) {
	unsigned long long int* address_as_ull = (unsigned long long int*) address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
				__double_as_longlong(value + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

__device__ void AtomicAddULL(UINT64* address, UINT32 add) {
#ifdef EMULATED_ATOMIC
	if (atomicAdd((UINT32*)address, add) + add < add) {
		atomicAdd(((UINT32*)address) + 1, 1U);
	}
#else
	atomicAdd(address, (UINT64) add);
#endif
}

int init_RNG(UINT64* x, UINT32* a, UINT64* xR, UINT32* aR, UINT64* xS,
		UINT32* aS, const UINT32 n_rng, const char* safeprimes_file,
		UINT64 xinit) {
	// xinit is the SEED (given or taken from clock)
	FILE* fp;
	UINT32 begin = 0u, beginR = 0u, beginS = 0u;
	UINT32 fora, forR, forS, tmp1, tmp2;
	UINT64 xinitR = SEED1, xinitS = SEED2;  // xinit=SEED0;

	if (strlen(safeprimes_file) == 0) {
		// Try to find it in the local directory
		safeprimes_file = "safeprimes_base32.txt";
	}

	fp = fopen(safeprimes_file, "r");

	if (fp == NULL) {
		printf("Could not find the file of safeprimes (%s)! Terminating!\n",
				safeprimes_file);
		return 1;
	}

	fscanf(fp, "%u %u %u", &begin, &tmp1, &tmp2);   // first safe prime
	fscanf(fp, "%u %u %u", &beginR, &tmp1, &tmp2);  // safe prime for ReflectT
	fscanf(fp, "%u %u %u", &beginS, &tmp1, &tmp2);  // safe prime for Spin

	// Here we set up a loop, using the first multiplier in the file to
	// generate x's and c's
	// There are some restrictions to these two numbers:
	// 0<=c<a and 0<=x<b, where a is the multiplier and b is the base (2^32)
	// also [x,c]=[0,0] and [b-1,a-1] are not allowed.

	// Make sure xinit is a valid seed (using the above mentioned
	// restrictions) HINT: xinit is the SEED (given or taken from clock)
	if ((xinit == 0ull) | (((UINT32) (xinit >> 32)) >= (begin - 1))
			| (((UINT32) xinit) >= 0xfffffffful)) {
		// xinit (probably) not a valid seed!
		//(we have excluded a few unlikely exceptions)
		printf("%llu not a valid seed! Terminating!\n", xinit);
		return 1;
	}

	for (UINT32 i = 0; i < n_rng; i++) {
		fscanf(fp, "%u %u %u", &fora, &tmp1, &tmp2);
		a[i] = fora;
		x[i] = 0;

		// seed for ReflectTransmit
		fscanf(fp, "%u %u %u", &forR, &tmp1, &tmp2);
		aR[i] = forR;
		xR[i] = 0;

		// seed for Spin
		fscanf(fp, "%u %u %u", &forS, &tmp1, &tmp2);
		aS[i] = forS;
		xS[i] = 0;

		while ((x[i] == 0) | (((UINT32) (x[i] >> 32)) >= (fora - 1))
				| (((UINT32) x[i]) >= 0xfffffffful)) {
			// generate a random number
			// HINT: xinit is the SEED (given or taken from clock) and begin is
			// the first safe prime in the list
			xinit = (xinit & 0xffffffffull) * (begin) + (xinit >> 32);

			// calculate c and store in the upper 32 bits of x[i]
			x[i] = (UINT32) floor(
					(((double) ((UINT32) xinit)) / (double) 0x100000000)
							* fora);  // Make sure 0<=c<a
			x[i] = x[i] << 32;

			// generate a random number and store in the lower 32 bits of x[i]
			//(as the initial x of the generator)
			// x will be 0<=x<b, where b is the base 2^32
			xinit = (xinit & 0xffffffffull) * (begin) + (xinit >> 32);
			x[i] += (UINT32) xinit;
		}  // End while x[i]

		while ((xR[i] == 0) | (((UINT32) (xR[i] >> 32)) >= (forR - 1))
				| (((UINT32) xR[i]) >= 0xfffffffful)) {
			// generate a random number
			// HINT: xinit is the SEED (given or taken from clock) and begin is
			// the first safe prime in the list
			xinitR = (xinitR & 0xffffffffull) * (beginR) + (xinitR >> 32);

			// calculate c and store in the upper 32 bits of x[i]
			xR[i] = (UINT32) floor(
					(((double) ((UINT32) xinitR)) / (double) 0x100000000)
							* forR);  // Make sure 0<=c<a
			xR[i] = xR[i] << 32;

			// generate a random number and store in the lower 32 bits of x[i]
			//(as the initial x of the generator)
			// x will be 0<=x<b, where b is the base 2^32
			xinitR = (xinitR & 0xffffffffull) * (beginR) + (xinitR >> 32);
			xR[i] += (UINT32) xinitR;
		}  // End while x[i]

		while ((xS[i] == 0) | (((UINT32) (xS[i] >> 32)) >= (forS - 1))
				| (((UINT32) xS[i]) >= 0xfffffffful)) {
			// generate a random number
			// HINT: xinit is the SEED (given or taken from clock) and begin is
			// the first safe prime in the list
			xinitS = (xinitS & 0xffffffffull) * (beginS) + (xinitS >> 32);

			// calculate c and store in the upper 32 bits of x[i]
			xS[i] = (UINT32) floor(
					(((double) ((UINT32) xinitS)) / (double) 0x100000000)
							* forS);  // Make sure 0<=c<a
			xS[i] = xS[i] << 32;

			// generate a random number and store in the lower 32 bits of x[i]
			//(as the initial x of the generator)
			// x will be 0<=x<b, where b is the base 2^32
			xinitS = (xinitS & 0xffffffffull) * (beginS) + (xinitS >> 32);
			xS[i] += (UINT32) xinitS;
		}
	}
	fclose(fp);

	return 0;
}

/**********************************************************
 **  Initialize Device Constant Memory with read-only data
 **********************************************************/
int InitDeviceConstantMemory(SimulationStruct* sim) {
	// Make sure that the number of regions is within the limit
	UINT32 n_regions = sim->n_regions;
	if (n_regions > MAX_REGIONS)
		return 1;

	SimParamGPU h_simparam;

	h_simparam.num_regions = sim->n_regions;  // not plus 2 here
	h_simparam.Rspecular = sim->Rspecular;
	h_simparam.rootIdx = sim->rootIdx;
	h_simparam.TypeBias = sim->TypeBias;
	h_simparam.TargetDepthMin = sim->TargetDepthMin;
	h_simparam.TargetDepthMax = sim->TargetDepthMax;
	h_simparam.BackwardBiasCoefficient = sim->BackwardBiasCoefficient;
	h_simparam.rndStepSizeInTissue = sim->rndStepSizeInTissue;
	h_simparam.MaxCollectingAngleDeg = sim->MaxCollectingAngleDeg;
	h_simparam.MaxCollectingRadius = sim->MaxCollectingRadius;
	h_simparam.ProbabilityAdditionalBias = sim->ProbabilityAdditionalBias;
	h_simparam.OpticalDepthShift = sim->OpticalDepthShift;
	h_simparam.CoherenceLengthSource = sim->CoherenceLengthSource;
	h_simparam.NumOpticalDepthLengthSteps = sim->NumOpticalDepthLengthSteps;

	checkCudaErrors(cudaMemcpyToSymbol(d_simparam, &h_simparam, sizeof(SimParamGPU)));

	RegionStructGPU h_regionspecs[MAX_REGIONS];

	for (UINT32 i = 0; i < n_regions; ++i) {

		h_regionspecs[i].n = sim->regions[i].n;

		FLOAT rmuas = sim->regions[i].mutr;
		h_regionspecs[i].muas = FP_ONE / rmuas;
		h_regionspecs[i].rmuas = rmuas;
		h_regionspecs[i].mua_muas = sim->regions[i].mua * rmuas;

		h_regionspecs[i].g = sim->regions[i].g;
	}

	// Copy region data to constant device memory
	checkCudaErrors(cudaMemcpyToSymbol(d_regionspecs, &h_regionspecs, n_regions * sizeof(RegionStructGPU)));

	return 0;
}

/**************************************************************
 **  Transfer data from Device to Host memory after simulation
 **************************************************************/
int CopyDeviceToHostMem(SimState* HostMem, UINT64* d_x, UINT64* d_xR,
		UINT64* d_xS, SimulationStruct* sim, SimState* DeviceMem,
		UINT64* d_NumClassI_PhotonsFilteredInRange,
		UINT64* d_NumClassII_PhotonsFilteredInRange,
		FLOAT* d_ReflectanceClassI_Sum, FLOAT* d_ReflectanceClassI_Max,
		FLOAT* d_ReflectanceClassI_SumSq,
		FLOAT* d_ReflectanceClassII_Sum,
		FLOAT* d_ReflectanceClassII_Max, FLOAT* d_ReflectanceClassII_SumSq,
		short STEPS) {


	// Probe Locations
	checkCudaErrors(
		cudaMemcpy(&HostMem->probe_x, &DeviceMem->probe_x, sizeof(FLOAT),
				cudaMemcpyDeviceToHost));

	checkCudaErrors(
			cudaMemcpy(&HostMem->probe_y, &DeviceMem->probe_y, sizeof(FLOAT),
					cudaMemcpyDeviceToHost));

	checkCudaErrors(
			cudaMemcpy(&HostMem->probe_z, &DeviceMem->probe_z, sizeof(FLOAT),
					cudaMemcpyDeviceToHost));

	// Copy Recordings matrix
	checkCudaErrors(
			cudaMemcpy(&HostMem->NumFilteredPhotons,
					&DeviceMem->NumFilteredPhotons, sizeof(UINT64),
					cudaMemcpyDeviceToHost));
	checkCudaErrors(
			cudaMemcpy(&HostMem->NumFilteredPhotonsClassI,
					&DeviceMem->NumFilteredPhotonsClassI, sizeof(UINT64),
					cudaMemcpyDeviceToHost));
	checkCudaErrors(
			cudaMemcpy(&HostMem->NumFilteredPhotonsClassII,
					&DeviceMem->NumFilteredPhotonsClassII, sizeof(UINT64),
					cudaMemcpyDeviceToHost));

	STEPS += 1;
	checkCudaErrors(
			cudaMemcpy(HostMem->NumClassI_PhotonsFilteredInRange,
					d_NumClassI_PhotonsFilteredInRange, STEPS * sizeof(UINT64),
					cudaMemcpyDeviceToHost));
	checkCudaErrors(
			cudaMemcpy(HostMem->NumClassII_PhotonsFilteredInRange,
					d_NumClassII_PhotonsFilteredInRange, STEPS * sizeof(UINT64),
					cudaMemcpyDeviceToHost));

	// Copy Recording values
	STEPS += 1;
	checkCudaErrors(
			cudaMemcpy(HostMem->ReflectanceClassI_Sum, d_ReflectanceClassI_Sum,
					STEPS * sizeof(FLOAT), cudaMemcpyDeviceToHost));
	checkCudaErrors(
			cudaMemcpy(HostMem->ReflectanceClassI_Max, d_ReflectanceClassI_Max,
					STEPS * sizeof(FLOAT), cudaMemcpyDeviceToHost));
	checkCudaErrors(
			cudaMemcpy(HostMem->ReflectanceClassI_SumSq,
					d_ReflectanceClassI_SumSq, STEPS * sizeof(FLOAT),
					cudaMemcpyDeviceToHost));

	checkCudaErrors(
			cudaMemcpy(HostMem->ReflectanceClassII_Sum,
					d_ReflectanceClassII_Sum, STEPS * sizeof(FLOAT),
					cudaMemcpyDeviceToHost));
	checkCudaErrors(
			cudaMemcpy(HostMem->ReflectanceClassII_Max,
					d_ReflectanceClassII_Max, STEPS * sizeof(FLOAT),
					cudaMemcpyDeviceToHost));
	checkCudaErrors(
			cudaMemcpy(HostMem->ReflectanceClassII_SumSq,
					d_ReflectanceClassII_SumSq, STEPS * sizeof(FLOAT),
					cudaMemcpyDeviceToHost));

	return 0;
}

/*********************
 **  Free Host Memory
 *********************/
void FreeHostSimState(SimState* hstate) {
	if (hstate->n_photons_left != NULL) {
		free(hstate->n_photons_left);  // hstate->n_photons_left = NULL;
	}
}

/********************
 **  Free GPU Memory
 ********************/
void FreeDeviceSimStates(SimState* dstate, UINT32* d_n_photons_left,
		UINT64* d_x, UINT32* d_a, UINT64* d_xR, UINT32* d_aR, UINT64* d_xS,
		UINT32* d_aS, GPUThreadStates* tstates, int* d_NextTetrahedron,
		int* d_NextTetrahedron_cont, UINT32* d_is_active, UINT32* d_dead,
		UINT32* d_hit, UINT32* d_photon_region, UINT32* d_rootIdx,
		UINT32* d_FstBackReflectionFlag,
		UINT32* d_NumBackwardsSpecularReflections, UINT32* d_dead_cont,
		UINT32* d_rootIdx_cont,
		UINT32* d_FstBackReflectionFlag_cont,
		UINT32* d_NumBackwardsSpecularReflections_cont, FLOAT* d_photon_x,
		FLOAT* d_photon_y, FLOAT* d_photon_z, FLOAT* d_photon_ux,
		FLOAT* d_photon_uy, FLOAT* d_photon_uz, FLOAT* d_photon_x_cont,
		FLOAT* d_photon_y_cont, FLOAT* d_photon_z_cont, FLOAT* d_photon_ux_cont,
		FLOAT* d_photon_uy_cont, FLOAT* d_photon_uz_cont, FLOAT* d_photon_w,
		FLOAT* d_photon_s, FLOAT* d_photon_sleft, FLOAT* d_photon_w_cont,
		FLOAT* d_photon_s_cont, FLOAT* d_photon_sleft_cont, FLOAT* d_MinCos,
		FLOAT* d_OpticalPath, FLOAT* d_MaxDepth, FLOAT* d_LikelihoodRatio,
		FLOAT* d_LocationFstBias,
		FLOAT* d_MinCos_cont, FLOAT* d_OpticalPath_cont, FLOAT* d_MaxDepth_cont,
		FLOAT* d_LikelihoodRatio_cont,
		FLOAT* d_LocationFstBias_cont,
		FLOAT* d_LikelihoodRatioAfterFstBias,
		TetrahedronStructGPU* d_Tetrahedron) {
	cudaFree(dstate);
	cudaFree(d_n_photons_left);
	d_n_photons_left = NULL;

	cudaFree(d_x);
	d_x = NULL;
	cudaFree(d_a);
	d_a = NULL;
	cudaFree(d_xR);
	d_xR = NULL;
	cudaFree(d_aR);
	d_aR = NULL;
	cudaFree(d_xS);
	d_xS = NULL;
	cudaFree(d_aS);
	d_aS = NULL;

	cudaFree(tstates);
	cudaFree(d_NextTetrahedron);
	d_NextTetrahedron = NULL;
	cudaFree(d_NextTetrahedron_cont);
	d_NextTetrahedron_cont = NULL;
	cudaFree(d_is_active);
	d_is_active = NULL;
	cudaFree(d_dead);
	d_dead = NULL;
	cudaFree(d_hit);
	d_hit = NULL;
	cudaFree(d_photon_region);
	d_photon_region = NULL;
	cudaFree(d_rootIdx);
	d_rootIdx = NULL;
	cudaFree(d_FstBackReflectionFlag);
	d_FstBackReflectionFlag = NULL;
	cudaFree(d_NumBackwardsSpecularReflections);
	d_NumBackwardsSpecularReflections = NULL;
	cudaFree(d_dead_cont);
	d_dead_cont = NULL;
	cudaFree(d_rootIdx_cont);
	d_rootIdx_cont = NULL;
	cudaFree(d_FstBackReflectionFlag_cont);
	d_FstBackReflectionFlag_cont = NULL;
	cudaFree(d_NumBackwardsSpecularReflections_cont);
	d_NumBackwardsSpecularReflections_cont = NULL;

	cudaFree(d_photon_x);
	d_photon_x = NULL;
	cudaFree(d_photon_y);
	d_photon_y = NULL;
	cudaFree(d_photon_z);
	d_photon_z = NULL;
	cudaFree(d_photon_ux);
	d_photon_ux = NULL;
	cudaFree(d_photon_uy);
	d_photon_uy = NULL;
	cudaFree(d_photon_uz);
	d_photon_uz = NULL;
	cudaFree(d_photon_x_cont);
	d_photon_x = NULL;
	cudaFree(d_photon_y_cont);
	d_photon_y = NULL;
	cudaFree(d_photon_z_cont);
	d_photon_z = NULL;
	cudaFree(d_photon_ux_cont);
	d_photon_ux = NULL;
	cudaFree(d_photon_uy_cont);
	d_photon_uy = NULL;
	cudaFree(d_photon_uz_cont);
	d_photon_uz = NULL;

	cudaFree(d_photon_w);
	d_photon_w = NULL;
	cudaFree(d_photon_s);
	d_photon_s = NULL;
	cudaFree(d_photon_sleft);
	d_photon_sleft = NULL;
	cudaFree(d_photon_w_cont);
	d_photon_w_cont = NULL;
	cudaFree(d_photon_s_cont);
	d_photon_s_cont = NULL;
	cudaFree(d_photon_sleft_cont);
	d_photon_sleft_cont = NULL;

	cudaFree(d_MinCos);
	d_MinCos = NULL;
	cudaFree(d_OpticalPath);
	d_OpticalPath = NULL;
	cudaFree(d_MaxDepth);
	d_MaxDepth = NULL;
	cudaFree(d_LikelihoodRatio);
	d_LikelihoodRatio = NULL;
	cudaFree(d_LocationFstBias);
	d_LocationFstBias = NULL;
	cudaFree(d_MinCos_cont);
	d_MinCos_cont = NULL;
	cudaFree(d_OpticalPath_cont);
	d_OpticalPath_cont = NULL;
	cudaFree(d_MaxDepth_cont);
	d_MaxDepth_cont = NULL;
	cudaFree(d_LikelihoodRatio_cont);
	d_LikelihoodRatio_cont = NULL;
	cudaFree(d_LocationFstBias_cont);
	d_LocationFstBias_cont = NULL;

	cudaFree(d_LikelihoodRatioAfterFstBias);
	d_LikelihoodRatioAfterFstBias = NULL;

	cudaFree(d_Tetrahedron);
	d_Tetrahedron = NULL;
}

__device__ void CopyPhotonStruct(PhotonStructGPU* OrigPhoton,
		PhotonStructGPU* DestPhoton) {
	DestPhoton->x = OrigPhoton->x;
	DestPhoton->y = OrigPhoton->y;
	DestPhoton->z = OrigPhoton->z;
	DestPhoton->ux = OrigPhoton->ux;
	DestPhoton->uy = OrigPhoton->uy;
	DestPhoton->uz = OrigPhoton->uz;
	DestPhoton->w = OrigPhoton->w;
	DestPhoton->dead = OrigPhoton->dead;

	DestPhoton->MinCos = OrigPhoton->MinCos;
	DestPhoton->rootIdx = OrigPhoton->rootIdx;
	DestPhoton->NextTetrahedron = OrigPhoton->NextTetrahedron;

	DestPhoton->s = OrigPhoton->s;
	DestPhoton->sleft = OrigPhoton->sleft;

	DestPhoton->OpticalPath = OrigPhoton->OpticalPath;
	DestPhoton->MaxDepth = OrigPhoton->MaxDepth;
	DestPhoton->LikelihoodRatio = OrigPhoton->LikelihoodRatio;

	DestPhoton->FstBackReflectionFlag = OrigPhoton->FstBackReflectionFlag;
	DestPhoton->LocationFstBias = OrigPhoton->LocationFstBias;
	DestPhoton->NumBackwardsSpecularReflections = OrigPhoton->NumBackwardsSpecularReflections;
}

__device__ void LaunchPhoton(PhotonStructGPU* photon, FLOAT probe_x, FLOAT probe_y, FLOAT probe_z) {

	photon->x = probe_x;
	photon->y = probe_y;
	photon->z = probe_z;

	photon->ux = photon->uy = ZERO_FP;
	photon->uz = FP_ONE;
	photon->w = FP_ONE - d_simparam.Rspecular;

	photon->dead = 0;
	photon->hit = 0;
	photon->MinCos = ZERO_FP;
	photon->s = ZERO_FP;
	photon->sleft = ZERO_FP;

	photon->OpticalPath = ZERO_FP;
	photon->MaxDepth = ZERO_FP;
	photon->LikelihoodRatio = FP_ONE;
	photon->LikelihoodRatioAfterFstBias = FP_ONE;

	photon->FstBackReflectionFlag = 0;
	photon->LocationFstBias = -FP_ONE;
	photon->NumBackwardsSpecularReflections = 0;

	photon->NextTetrahedron = -1;

	photon->rootIdx = d_simparam.rootIdx;
}

__device__ void ComputeStepSize(PhotonStructGPU* photon, UINT64* rnd_x,
		UINT32* rnd_a) {
	// Make a new step if no leftover.
	if (photon->sleft == ZERO_FP) {
		FLOAT rand = rand_MWC_oc(rnd_x, rnd_a);
		photon->s = -log(rand) * d_regionspecs[photon->tetrahedron[photon->rootIdx].region].rmuas;
	} else {
		photon->s = photon->sleft * d_regionspecs[photon->tetrahedron[photon->rootIdx].region].rmuas;
		photon->sleft = ZERO_FP;
	}
}

__device__ UINT32 HitBoundary(PhotonStructGPU* photon) {
	/* step size to boundary */
	UINT32 tetrahedron_index = photon->rootIdx;

	FLOAT min_distance = 1E10;
	int index_of_tetrahedron_with_min_distance = -1;

	FLOAT cos_normals_and_photon_direction[4];
	FLOAT distance_from_face_in_photon_direction;
	FLOAT perpendicular_distance;

	for (int i = 0; i < 4; i++) {
		int face_idx = photon->tetrahedron[tetrahedron_index].faces[i];
		int sign = photon->tetrahedron[tetrahedron_index].signs[i];
		cos_normals_and_photon_direction[i] = (photon->faces[face_idx].Nx * sign) * photon->ux
				+ (photon->faces[face_idx].Ny * sign) * photon->uy
				+ (photon->faces[face_idx].Nz * sign) * photon->uz;
	}

	for (int i = 0; i < 4; i++) {
		if (cos_normals_and_photon_direction[i] < ZERO_FP) {
			// Photon is going toward the face
			int face_idx = photon->tetrahedron[tetrahedron_index].faces[i];
			int sign = photon->tetrahedron[tetrahedron_index].signs[i];
			perpendicular_distance = ((photon->faces[face_idx].Nx * sign) * photon->x
					+ (photon->faces[face_idx].Ny * sign) * photon->y
					+ (photon->faces[face_idx].Nz * sign) * photon->z
					+ photon->faces[face_idx].d * sign);

			distance_from_face_in_photon_direction = (FLOAT) FAST_DIV(-perpendicular_distance, cos_normals_and_photon_direction[i]);

			if (distance_from_face_in_photon_direction < min_distance) {
				min_distance = distance_from_face_in_photon_direction;
				index_of_tetrahedron_with_min_distance = i;
				photon->MinCos = cos_normals_and_photon_direction[i];
			}
		}
	}

	UINT32 hit_boundary;
	if (photon->s > min_distance) {

		FLOAT mut = d_regionspecs[photon->tetrahedron[tetrahedron_index].region].muas;
		photon->sleft = (photon->s - min_distance) * mut;
		photon->s = min_distance;
		hit_boundary = 1;
		photon->NextTetrahedron = index_of_tetrahedron_with_min_distance;

	} else {

		hit_boundary = 0;
		photon->NextTetrahedron = -1;

	}
	return hit_boundary;
}

__device__ void Hop(PhotonStructGPU* photon) {
	photon->x += photon->s * photon->ux;
	photon->y += photon->s * photon->uy;
	photon->z += photon->s * photon->uz;

	photon->OpticalPath += photon->s;
	if (photon->MaxDepth < photon->z)
		photon->MaxDepth = photon->z;
}

__device__ void Drop(PhotonStructGPU* photon) {
	UINT32 region = photon->tetrahedron[photon->rootIdx].region;
	FLOAT dwa = photon->w * d_regionspecs[region].mua_muas;
	photon->w -= dwa;
}

__device__ void FastReflectTransmit(PhotonStructGPU* photon, SimState* d_state,
		UINT64* rnd_xR, UINT32* rnd_aR, FLOAT probe_x, FLOAT probe_y, FLOAT probe_z) {

	UINT32 rootIdx = photon->rootIdx;
	UINT32 ONE = 1;

	// cosines of transmission alpha
	FLOAT ux_reflected = 0, uy_reflected = 0, uz_reflected = 0;
	FLOAT ux_refracted = 0, uy_refracted = 0, uz_refracted = 0;

	// cosine of the incident angle (0 to 90 deg)
	FLOAT rFresnel;
	FLOAT ca1 = -(photon->MinCos);

	FLOAT ni = d_regionspecs[photon->tetrahedron[rootIdx].region].n;
	FLOAT nt;
	int next = photon->NextTetrahedron;
	int next_tetrahedron_index = photon->tetrahedron[rootIdx].adjTetras[next];
	if (next_tetrahedron_index == -1)
		nt = d_regionspecs[0].n;  // Ambient medium's n
	else {
		int next_region = photon->tetrahedron[next_tetrahedron_index].region;
		nt = d_regionspecs[next_region].n;
	}

	FLOAT ni_nt = (FLOAT) FAST_DIV(ni, nt);

	FLOAT sa1 = SQRT(FP_ONE - ca1 * ca1);
	if (ca1 > COSZERO)
		sa1 = ZERO_FP;
	FLOAT sa2 = FAST_MIN(ni_nt * sa1, FP_ONE);
	FLOAT ca2 = SQRT(FP_ONE - sa2 * sa2);

	FLOAT ca1ca2 = ca1 * ca2;
	FLOAT sa1sa2 = sa1 * sa2;
	FLOAT sa1ca2 = sa1 * ca2;
	FLOAT ca1sa2 = ca1 * sa2;

	// normal incidence: [(1-ni_nt)/(1+ni_nt)]^2
	// We ensure that ca1ca2 = 1, sa1sa2 = 0, sa1ca2 = 1, ca1sa2 = ni_nt
	if (ca1 > COSZERO) {
		sa1ca2 = FP_ONE;
		ca1sa2 = ni_nt;
	}

	FLOAT cam = ca1ca2 + sa1sa2; /* c- = cc + ss. */
	FLOAT sap = sa1ca2 + ca1sa2; /* s+ = sc + cs. */
	FLOAT sam = sa1ca2 - ca1sa2; /* s- = sc - cs. */

	rFresnel = (FLOAT) FAST_DIV(sam, sap * cam);
	rFresnel *= rFresnel;
	rFresnel *= (ca1ca2 * ca1ca2 + sa1sa2 * sa1sa2);

	// In this case, we do not care if "uz1" is exactly 0.
	if (ca1 < COSNINETYDEG || sa2 == FP_ONE)
		rFresnel = FP_ONE;

	UINT32 nxtFaceIdx =
			photon->tetrahedron[rootIdx].faces[photon->NextTetrahedron];
	int sign = photon->tetrahedron[rootIdx].signs[photon->NextTetrahedron];

	ux_refracted = photon->ux;
	uy_refracted = photon->uy;
	uz_refracted = photon->uz;

	ux_reflected = 2 * ca1 * photon->faces[nxtFaceIdx].Nx * sign + photon->ux;
	uy_reflected = 2 * ca1 * photon->faces[nxtFaceIdx].Ny * sign + photon->uy;
	uz_reflected = 2 * ca1 * photon->faces[nxtFaceIdx].Nz * sign + photon->uz;

	FLOAT rand = rand_MWC_co(rnd_xR, rnd_aR);

#if PARTIAL_REFLECTION
	if (next_tetrahedron_index == -1 && rFresnel < FP_ONE) {  // -1 = NULL, no adjacent tetra
		photon->ux = ux_refracted;
		photon->uy = uy_refracted;
		photon->uz = uz_refracted;

		if (photon->uz < 0) {
			short SpatialFilterFlag = 0;
			if (SQRT(SQ(photon->x-probe_x) + SQ(photon->y-probe_y))
					< d_simparam.MaxCollectingRadius
					&& acos(-photon->uz)
							< d_simparam.MaxCollectingAngleDeg * PI_const
									/ 180) {
				if (d_simparam.TypeBias == 0 || d_simparam.TypeBias == 3)
					SpatialFilterFlag = 1;

                if ((photon->FstBackReflectionFlag && d_simparam.TypeBias == 37
						|| (photon->LocationFstBias == photon->MaxDepth
								&& d_simparam.TypeBias != 37)
						|| photon->NumBackwardsSpecularReflections > 0))
					SpatialFilterFlag = 1;
			}

			if (SpatialFilterFlag) {
				AtomicAddULL(&d_state->NumFilteredPhotons, 1);
				FLOAT FilterOpticalPath = photon->OpticalPath;
				FLOAT FilterOpticalDepth = (FLOAT) FAST_DIV(FilterOpticalPath, FP_TWO);

				unsigned int VectorPosition = (unsigned int) (FAST_DIV(
								(FAST_DIV(FilterOpticalPath, FP_TWO) - d_simparam.OpticalDepthShift),
								FAST_DIV(d_simparam.CoherenceLengthSource, NUM_SUBSTEPS_RESOLUTION)));
				int ii;
				for (ii = VectorPosition - FAST_DIV(NUM_SUBSTEPS_RESOLUTION, 2);
						ii < VectorPosition + FAST_DIV(NUM_SUBSTEPS_RESOLUTION, 2);
						ii++) {
					if (ii >= 0 && ii < d_simparam.NumOpticalDepthLengthSteps) {
						short FilterClassI_Flag =
								photon->MaxDepth > (FilterOpticalDepth - FAST_DIV(d_simparam.CoherenceLengthSource, FP_TWO));

						FLOAT tmpPhotonContribution = photon->w * (FP_ONE - rFresnel) * photon->LikelihoodRatio;

						if (FilterClassI_Flag) {
							if (ii == VectorPosition)
								AtomicAddULL(&d_state->NumFilteredPhotonsClassI,
										ONE);

							AtomicAddD((&d_state->ReflectanceClassI_Sum[ii]),
									tmpPhotonContribution);

							if (d_state->ReflectanceClassI_Max[ii]
									< tmpPhotonContribution)
								d_state->ReflectanceClassI_Max[ii] =
										tmpPhotonContribution;

							AtomicAddD((&d_state->ReflectanceClassI_SumSq[ii]),
									SQ(tmpPhotonContribution));

							AtomicAddULL(
									(&d_state->NumClassI_PhotonsFilteredInRange[ii]),
									ONE);

						} else {
							if (ii == VectorPosition)
								AtomicAddULL(
										&d_state->NumFilteredPhotonsClassII,
										ONE);

							if (d_state->ReflectanceClassII_Max[ii]
									< tmpPhotonContribution)
								d_state->ReflectanceClassII_Max[ii] =
										tmpPhotonContribution;

							AtomicAddD((&d_state->ReflectanceClassII_Sum[ii]),
									tmpPhotonContribution);
							AtomicAddD((&d_state->ReflectanceClassII_SumSq[ii]),
									SQ(tmpPhotonContribution));

							AtomicAddULL(
									(&d_state->NumClassII_PhotonsFilteredInRange[ii]),
									ONE);
						}
					}
				}
			}
		} else
			photon->NumBackwardsSpecularReflections++;

		// The rest of the photon will be reflected
		photon->w *= rFresnel;
		photon->ux = ux_reflected;
		photon->uy = uy_reflected;
		photon->uz = uz_reflected;

	} else if (rand > rFresnel) {

		photon->rootIdx = photon->tetrahedron[next_tetrahedron_index].idx;

		photon->ux = ux_refracted;
		photon->uy = uy_refracted;
		photon->uz = uz_refracted;

		photon->NextTetrahedron = -1;

	} else {

		photon->NextTetrahedron = -1;

		if (photon->uz > ZERO_FP)
			photon->NumBackwardsSpecularReflections++;

		photon->ux = ux_reflected;
		photon->uy = uy_reflected;
		photon->uz = uz_reflected;

	}
#else  // Statistical splitting
	if (rFresnel < rand) {  // transmitted
								// The move is to transmit.

		if (nxtAdjIdx == -1) { // There is no adjacent tetrahedron, -1 = NULL

			photon->ux = ux_refracted;
			photon->uy = uy_refracted;
			photon->uz = uz_refracted;

			if (photon->uz < ZERO_FP) photon->dead = 1;
		} else {

			// Refraction direction cosines
			photon->ux = ux_refracted;
			photon->uy = uy_refracted;
			photon->uz = uz_refracted;

			photon->rootIdx = photon->tetrahedron[next_tetrahedron_index].idx;
			photon->NextTetrahedron = -1;
		}
	} else {
		// Reflected
		photon->ux = ux_reflected;
		photon->uy = uy_reflected;
		photon->uz = uz_reflected;
		photon->NextTetrahedron = -1;
	}
#endif
}

__device__ FLOAT SpinTheta(FLOAT g, UINT64* rnd_x, UINT32* rnd_a)
{
    FLOAT cost;

    if(g == ZERO_FP)
        cost = FP_TWO*rand_MWC_oc(rnd_x, rnd_a)-FP_ONE;
    else {
        double temp = (FLOAT) FAST_DIV((FP_ONE-SQ(g)),(FP_ONE - g + FP_TWO * g * rand_MWC_oc(rnd_x, rnd_a)));
        cost = (FLOAT) FAST_DIV((FP_ONE + SQ(g) - SQ(temp)),(FP_TWO * g));
        if(cost < -FP_ONE) cost = -FP_ONE;
        else if(cost > FP_ONE) cost = FP_ONE;
    }
    return(cost);
}

__device__ void Spin(FLOAT g, PhotonStructGPU* photon, UINT64* rnd_xS,
		UINT32* rnd_aS) {
	FLOAT cost, sint;  // cosine and sine of the polar deflection angle theta
	FLOAT cosp, sinp;  // cosine and sine of the azimuthal angle psi
	FLOAT psi;
	FLOAT SIGN;
	FLOAT temp;
	FLOAT last_ux, last_uy, last_uz;

	/**************************************************************
	 **  Choose (sample) a new theta angle for photon propagation
	 **	according to the anisotropy.
	 **
	 **	If anisotropy g is 0, then
	 **		cos(theta) = 2*rand-1.
	 **	otherwise
	 **		sample according to the Henyey-Greenstein function.
	 **
	 **	Returns the cosine of the polar deflection angle theta.
	 ***************************************************************/

	cost = SpinTheta(g, rnd_xS, rnd_aS);
	sint = SQRT(FP_ONE - SQ(cost));

	psi = FP_TWO * PI_const * rand_MWC_co(rnd_xS, rnd_aS);;
	sincos(psi, &sinp, &cosp);

	FLOAT stcp = sint * cosp;
	FLOAT stsp = sint * sinp;

	last_ux = photon->ux;
	last_uy = photon->uy;
	last_uz = photon->uz;

	if (fabs(last_uz) > COSZERO) {  // Normal incident
		photon->ux = stcp;
		photon->uy = stsp;
		SIGN = ((last_uz) >= ZERO_FP ? FP_ONE : -FP_ONE);
		photon->uz = cost * SIGN;
	} else {  // Regular incident
		temp = RSQRT(FP_ONE - last_uz * last_uz);
		photon->ux = (stcp * last_ux * last_uz - stsp * last_uy) * temp
				+ last_ux * cost;
		photon->uy = (stcp * last_uy * last_uz + stsp * last_ux) * temp
				+ last_uy * cost;
		photon->uz = (FLOAT) FAST_DIV(-stcp, temp) + last_uz * cost;
	}

// Normalize unit vector to ensure its magnitude is 1 (unity)
// only required in 32-bit floating point version
#ifdef SINGLE_PRECISION
	temp = RSQRT(photon->ux * photon->ux + photon->uy * photon->uy +
			photon->uz * photon->uz);
	photon->ux = photon->ux * temp;
	photon->uy = photon->uy * temp;
	photon->uz = photon->uz * temp;
#endif
}

__device__ FLOAT SpinThetaForwardFstBias(FLOAT g, UINT64* rnd_x, UINT32* rnd_a){
  FLOAT cost;

  if(g == ZERO_FP)
    cost = rand_MWC_oc(rnd_x, rnd_a);
  else {
    FLOAT randTmp = rand_MWC_oc(rnd_x, rnd_a);
    FLOAT temp = (FLOAT) FAST_DIV(randTmp, (FP_ONE-g)) + FAST_DIV((FP_ONE-randTmp), SQRT(SQ(g)+FP_ONE));
    cost = FAST_DIV(SQ(g) + FP_ONE - FAST_DIV(FP_ONE, SQ(temp) ),(FP_TWO*g));
	if(cost < -FP_ONE)
		cost = -FP_ONE;
	else if(cost > FP_ONE)
		cost = FP_ONE;
  }
  return(cost);
}

__device__ void SpinBias(FLOAT g, UINT32 region, PhotonStructGPU* photon, PhotonStructGPU* photon_cont, UINT64* rnd_x, UINT32* rnd_a, FLOAT probe_x, FLOAT probe_y, FLOAT probe_z) {

	FLOAT g_squared = SQ(g);

	FLOAT cost, sint;  // cosine and sine of the polar deflection angle theta
	FLOAT cosp, sinp;  // cosine and sine of the azimuthal angle psi
	FLOAT costg, costg1, costg2;
	FLOAT psi;
	FLOAT temp;
	FLOAT ux = photon->ux;
	FLOAT uy = photon->uy;
	FLOAT uz = photon->uz;
	FLOAT ux_Orig = ux;
	FLOAT uy_Orig = uy;
	FLOAT uz_Orig = uz;
	FLOAT rand;

	FLOAT BackwardBiasCoefficient = d_simparam.BackwardBiasCoefficient;
	FLOAT BiasCoefficientTmp = ZERO_FP;

	short int ReachedTargetOpticalDepthAndGoingBackwarFlag = 0;
	short int ThisIsFirstBackwardBiasFlag = 0;

	if (photon->z > d_simparam.TargetDepthMin
        && photon->z < d_simparam.TargetDepthMax
        && photon->uz > ZERO_FP
        && !photon->FstBackReflectionFlag) {
		// This bias backwards will be applied only if the photon is going forward
		// The status of the photon prior to the bias will be saved

		CopyPhotonStruct(photon, photon_cont);
		ReachedTargetOpticalDepthAndGoingBackwarFlag = 1;
		photon->FstBackReflectionFlag = 1;  // Bias backwards only once
		photon->LocationFstBias = photon->z;
		BiasCoefficientTmp = BackwardBiasCoefficient;
		ThisIsFirstBackwardBiasFlag = 1;
	}

	/**********************************
	 ** Biased Direction towards probe
	 **********************************/
	FLOAT vx = probe_x-photon->x;
	FLOAT vy = probe_y-photon->y;
	FLOAT vz = probe_z-photon->z;
	FLOAT LengthVector = SQRT(SQ(vx) + SQ(vy) + SQ(vz));
	vx = (FLOAT) FAST_DIV(vx, LengthVector);
	vy = (FLOAT) FAST_DIV(vy, LengthVector);
	vz = (FLOAT) FAST_DIV(vz, LengthVector);
	/*********************************/

	if ((photon->FstBackReflectionFlag
         || photon->NumBackwardsSpecularReflections > 0)
			&& !ReachedTargetOpticalDepthAndGoingBackwarFlag) {

		// It was biased at least once before and is moving backwards
		ReachedTargetOpticalDepthAndGoingBackwarFlag = 2;

		FLOAT mut = d_regionspecs[photon->tetrahedron[photon->rootIdx].region].muas;
		FLOAT NextStepSize = (FLOAT) FAST_DIV(
				-log(d_simparam.rndStepSizeInTissue), mut);
		FLOAT CurrentDistanceToOrigin = SQRT(
				SQ(photon->x-probe_x) + SQ(photon->y-probe_y) + SQ(photon->z-probe_z));

        if (NextStepSize >= CurrentDistanceToOrigin
				&& acos(-vz) <= (d_simparam.MaxCollectingAngleDeg * PI_const / 180)) {
			ReachedTargetOpticalDepthAndGoingBackwarFlag = 1;
		}

        BiasCoefficientTmp = BackwardBiasCoefficient;
	}

	int BiasFunctionRandomlySelected = 0;
	if (ReachedTargetOpticalDepthAndGoingBackwarFlag) {
		// Photon reached target optical region it may undergo additional biased
		// scattering or unbiased scattering

		// BiasFunctionRandomlySelected=1 means use biased scattering and 2 means unbiased scattering
		rand_MWC_co(rnd_x, rnd_a) <= d_simparam.ProbabilityAdditionalBias ? BiasFunctionRandomlySelected = 1 : BiasFunctionRandomlySelected = 2;

		if (ReachedTargetOpticalDepthAndGoingBackwarFlag == 1
            || BiasFunctionRandomlySelected == 1) {
			/*************************************************************************
			** The photon is within the target depth and going forward
			** The additional biased scattering is randomly chosen
			** So the scattering is biased Henyey-Greenstein scattering
			*************************************************************************/
			cost = SpinThetaForwardFstBias(BiasCoefficientTmp, rnd_x, rnd_a);
			ux = vx;
			uy = vy;
			uz = vz;
		} else {
			/**************************************************************************************
			** The photon is within the target depth but the scattering is randomly selected is
			** unbiased scattering
			** or the photon is already going backward or it is out of target depth
			**************************************************************************************/
			cost = SpinTheta(g, rnd_x, rnd_a);
		}
	} else {
		/**************************************************************************
		**  The photon is not within the target depth or it is not going forward
		**  so do unbiased scattering
		**************************************************************************/
		cost = SpinTheta(g, rnd_x, rnd_a);
	}
	cost = FAST_MAX(cost, -FP_ONE);
	cost = FAST_MIN(cost, FP_ONE);

	sint = SQRT(FP_ONE - cost * cost);

	/* spin psi 0-2pi. */
	rand = rand_MWC_co(rnd_x, rnd_a);

	psi = FP_TWO * PI_const * rand;
	sincos(psi, &sinp, &cosp);

	FLOAT stcp = sint * cosp;
	FLOAT stsp = sint * sinp;

	if (fabs(uz) > COSZERO) {  // Normal incident
		photon->ux = stcp;
		photon->uy = stsp;
		photon->uz = copysign(cost, uz * cost);
	} else {  // Regular incident
		temp = RSQRT(FP_ONE - uz * uz);
		photon->ux = (stcp * ux * uz - stsp * uy) * temp + ux * cost;
		photon->uy = (stcp * uy * uz + stsp * ux) * temp + uy * cost;
		photon->uz = FAST_DIV(-stcp, temp) + uz * cost;
	}

	costg = ux_Orig * photon->ux
            + uy_Orig * photon->uy
            + uz_Orig * photon->uz;
	costg2 = costg;
	costg1 = vx * photon->ux
             + vy * photon->uy
             + vz * photon->uz;

	if (BiasCoefficientTmp) {
		FLOAT one_plus_a_squared = 1 + SQ(BiasCoefficientTmp);
		FLOAT sqrt_one_plus_a_squared = SQRT(one_plus_a_squared);
		FLOAT LikelihoodRatioIncreaseFactor;
		if (ReachedTargetOpticalDepthAndGoingBackwarFlag == 1)
			/****************************************************************************************
			 ** Likelihood for the first Bias scattering. Equation (8) of the paper:
			 ** Malektaji, Siavash, Ivan T. Lima, and Sherif S. Sherif. "Monte Carlo simulation of
			 ** optical coherence tomography for turbid media with arbitrary spatial distributions."
			 ** Journal of biomedical optics 19.4 (2014): 046001-046001.
			 ****************************************************************************************/
			LikelihoodRatioIncreaseFactor = FAST_DIV(
					(1 - g_squared) * (sqrt_one_plus_a_squared - FP_ONE + BiasCoefficientTmp) * SQRT( CUBE( one_plus_a_squared - FP_TWO * BiasCoefficientTmp * cost)),
					FP_TWO*BiasCoefficientTmp * (1-BiasCoefficientTmp) * sqrt_one_plus_a_squared * SQRT( CUBE( FP_ONE+ g_squared - FP_TWO * g * costg))
					);

        else {
			FLOAT cost1, cost2;
			if (BiasFunctionRandomlySelected == 1){
				   cost1 = cost;
				   cost2 = costg2;
			}
			else{
				   cost1 = costg1;
				   cost2 = cost;
			}
			/*******************************************************************************************************
			 **  The likelihood ratio of additional biased scatterings, whether the biased or the unbiased
			 **  probability density function is randomly selected, is calculated according to the equation (9)
			 **  of the paper:
			 **  Malektaji, Siavash, Ivan T. Lima, and Sherif S. Sherif. "Monte Carlo simulation of
			 **  optical coherence tomography for turbid media with arbitrary spatial distributions."
			 **  Journal of biomedical optics 19.4 (2014): 046001-046001.
			 *******************************************************************************************************/

      FLOAT pdf1 = (FLOAT) FAST_DIV(sqrt_one_plus_a_squared * BiasCoefficientTmp * (FP_ONE - BiasCoefficientTmp),
    		  	  	  	  	  	  	  (sqrt_one_plus_a_squared - FP_ONE + BiasCoefficientTmp)*SQRT(CUBE(one_plus_a_squared - FP_TWO * BiasCoefficientTmp * cost1)));

      FLOAT pdf2 = (FLOAT) FAST_DIV( FP_ONE - g_squared,
    		  	  	  	  	  	  	 (FP_TWO * CUBE(sqrt(FP_ONE + g_squared - FP_TWO * g * cost2))) );

      LikelihoodRatioIncreaseFactor = (FLOAT) FAST_DIV( pdf2,
    		  (d_simparam.ProbabilityAdditionalBias * pdf1 + (FP_ONE - d_simparam.ProbabilityAdditionalBias)* pdf2) );
    }
    photon->LikelihoodRatio *= LikelihoodRatioIncreaseFactor;
    if (ThisIsFirstBackwardBiasFlag == 1)
      // In case there was a sure backward bias and that was the very first one
      photon->LikelihoodRatioAfterFstBias = photon->LikelihoodRatio;

  }
}

__global__ void InitThreadState(GPUThreadStates* tstates, FLOAT probe_x, FLOAT probe_y, FLOAT probe_z) {
	PhotonStructGPU photon_temp;

	// This is the unique ID for each thread (or thread ID = tid)
	UINT32 tid = blockIdx.x * NUM_THREADS_PER_BLOCK + threadIdx.x;

	// Initialize the photon and copy into photon_<parameter x>
	LaunchPhoton(&photon_temp, probe_x, probe_y, probe_z);

	tstates->photon_x[tid] = photon_temp.x;
	tstates->photon_y[tid] = photon_temp.y;
	tstates->photon_z[tid] = photon_temp.z;
	tstates->photon_ux[tid] = photon_temp.ux;
	tstates->photon_uy[tid] = photon_temp.uy;
	tstates->photon_uz[tid] = photon_temp.uz;
	tstates->photon_w[tid] = photon_temp.w;
	tstates->dead[tid] = photon_temp.dead;
	tstates->hit[tid] = photon_temp.hit;
	tstates->MinCos[tid] = photon_temp.MinCos;

	tstates->photon_x_cont[tid] = photon_temp.x;
	tstates->photon_y_cont[tid] = photon_temp.y;
	tstates->photon_z_cont[tid] = photon_temp.z;
	tstates->photon_ux_cont[tid] = photon_temp.ux;
	tstates->photon_uy_cont[tid] = photon_temp.uy;
	tstates->photon_uz_cont[tid] = photon_temp.uz;
	tstates->photon_w_cont[tid] = photon_temp.w;
	tstates->dead_cont[tid] = photon_temp.dead;
	tstates->MinCos_cont[tid] = photon_temp.MinCos;

	tstates->NextTetrahedron[tid] = photon_temp.NextTetrahedron;
	tstates->rootIdx[tid] = photon_temp.rootIdx;

	tstates->NextTetrahedron_cont[tid] = photon_temp.NextTetrahedron;
	tstates->rootIdx_cont[tid] = photon_temp.rootIdx;

	tstates->photon_s[tid] = photon_temp.s;
	tstates->photon_sleft[tid] = photon_temp.sleft;
	tstates->OpticalPath[tid] = photon_temp.OpticalPath;
	tstates->MaxDepth[tid] = photon_temp.MaxDepth;
	tstates->LikelihoodRatio[tid] = photon_temp.LikelihoodRatio;

	tstates->photon_s_cont[tid] = photon_temp.s;
	tstates->photon_sleft_cont[tid] = photon_temp.sleft;
	tstates->OpticalPath_cont[tid] = photon_temp.OpticalPath;
	tstates->MaxDepth_cont[tid] = photon_temp.MaxDepth;
	tstates->LikelihoodRatio_cont[tid] = photon_temp.LikelihoodRatio;


	tstates->LikelihoodRatioAfterFstBias[tid] =
			photon_temp.LikelihoodRatioAfterFstBias;

	tstates->FstBackReflectionFlag[tid] = photon_temp.FstBackReflectionFlag;
	tstates->LocationFstBias[tid] = photon_temp.LocationFstBias;
	tstates->NumBackwardsSpecularReflections[tid] =
			photon_temp.NumBackwardsSpecularReflections;

	tstates->FstBackReflectionFlag_cont[tid] =
			photon_temp.FstBackReflectionFlag;
	tstates->LocationFstBias_cont[tid] = photon_temp.LocationFstBias;
	tstates->NumBackwardsSpecularReflections_cont[tid] =
			photon_temp.NumBackwardsSpecularReflections;

	tstates->is_active[tid] = 1;
}

__device__ void SaveThreadState(SimState* d_state, GPUThreadStates* tstates,
		PhotonStructGPU* photon, PhotonStructGPU* photon_cont, UINT64 rnd_x,
		UINT32 rnd_a, UINT64 rnd_xR, UINT32 rnd_aR, UINT64 rnd_xS,
		UINT32 rnd_aS, UINT32 is_active) {
	UINT32 tid = blockIdx.x * NUM_THREADS_PER_BLOCK + threadIdx.x;

	d_state->x[tid] = rnd_x;
	d_state->a[tid] = rnd_a;
	d_state->xR[tid] = rnd_xR;
	d_state->aR[tid] = rnd_aR;
	d_state->xS[tid] = rnd_xS;
	d_state->aS[tid] = rnd_aS;

	tstates->photon_x[tid] = photon->x;
	tstates->photon_y[tid] = photon->y;
	tstates->photon_z[tid] = photon->z;
	tstates->photon_ux[tid] = photon->ux;
	tstates->photon_uy[tid] = photon->uy;
	tstates->photon_uz[tid] = photon->uz;
	tstates->photon_w[tid] = photon->w;
	tstates->dead[tid] = photon->dead;
	tstates->hit[tid] = photon->hit;
	tstates->MinCos[tid] = photon->MinCos;

	tstates->photon_x_cont[tid] = photon_cont->x;
	tstates->photon_y_cont[tid] = photon_cont->y;
	tstates->photon_z_cont[tid] = photon_cont->z;
	tstates->photon_ux_cont[tid] = photon_cont->ux;
	tstates->photon_uy_cont[tid] = photon_cont->uy;
	tstates->photon_uz_cont[tid] = photon_cont->uz;
	tstates->photon_w_cont[tid] = photon_cont->w;
	tstates->dead_cont[tid] = photon_cont->dead;
	tstates->MinCos_cont[tid] = photon_cont->MinCos;

	tstates->rootIdx[tid] = photon->rootIdx;
	tstates->NextTetrahedron[tid] = photon->NextTetrahedron;

	tstates->rootIdx_cont[tid] = photon_cont->rootIdx;
	tstates->NextTetrahedron_cont[tid] = photon_cont->NextTetrahedron;

	tstates->tetrahedron = photon->tetrahedron;
	tstates->faces = photon->faces;

	tstates->photon_s[tid] = photon->s;
	tstates->photon_sleft[tid] = photon->sleft;
	tstates->OpticalPath[tid] = photon->OpticalPath;
	tstates->MaxDepth[tid] = photon->MaxDepth;
	tstates->LikelihoodRatio[tid] = photon->LikelihoodRatio;

	tstates->photon_s_cont[tid] = photon_cont->s;
	tstates->photon_sleft_cont[tid] = photon_cont->sleft;
	tstates->OpticalPath_cont[tid] = photon_cont->OpticalPath;
	tstates->MaxDepth_cont[tid] = photon_cont->MaxDepth;
	tstates->LikelihoodRatio_cont[tid] = photon_cont->LikelihoodRatio;

	tstates->LikelihoodRatioAfterFstBias[tid] =
			photon->LikelihoodRatioAfterFstBias;

	tstates->FstBackReflectionFlag[tid] = photon->FstBackReflectionFlag;
	tstates->LocationFstBias[tid] = photon->LocationFstBias;
	tstates->NumBackwardsSpecularReflections[tid] =
			photon->NumBackwardsSpecularReflections;

	tstates->FstBackReflectionFlag_cont[tid] =
			photon_cont->FstBackReflectionFlag;
	tstates->LocationFstBias_cont[tid] = photon_cont->LocationFstBias;
	tstates->NumBackwardsSpecularReflections_cont[tid] =
			photon_cont->NumBackwardsSpecularReflections;

	tstates->is_active[tid] = is_active;
}

__device__ void CopytstatesToPhoton(GPUThreadStates* tstates,
		PhotonStructGPU* photon, PhotonStructGPU* photon_cont,
		UINT32* is_active, UINT32 tid) {

	/************
	 *  Photon
	 ***********/

	photon->x = tstates->photon_x[tid];
	photon->y = tstates->photon_y[tid];
	photon->z = tstates->photon_z[tid];

	photon->ux = tstates->photon_ux[tid];
	photon->uy = tstates->photon_uy[tid];
	photon->uz = tstates->photon_uz[tid];

	photon->w = tstates->photon_w[tid];
	photon->dead = tstates->dead[tid];
	photon->hit = tstates->hit[tid];

	photon->MinCos = tstates->MinCos[tid];

	photon->rootIdx = tstates->rootIdx[tid];
	photon->NextTetrahedron = tstates->NextTetrahedron[tid];

	photon->tetrahedron = tstates->tetrahedron;
	photon->faces = tstates->faces;

	photon->s = tstates->photon_s[tid];
	photon->sleft = tstates->photon_sleft[tid];
	photon->OpticalPath = tstates->OpticalPath[tid];
	photon->MaxDepth = tstates->MaxDepth[tid];
	photon->LikelihoodRatio = tstates->LikelihoodRatio[tid];

	photon->LikelihoodRatioAfterFstBias =
				tstates->LikelihoodRatioAfterFstBias[tid];

	photon->FstBackReflectionFlag = tstates->FstBackReflectionFlag[tid];
	photon->LocationFstBias = tstates->LocationFstBias[tid];

	photon->NumBackwardsSpecularReflections =
				tstates->NumBackwardsSpecularReflections[tid];

	/***************
	 *  Photon_Cont
	 ***************/

	photon_cont->x = tstates->photon_x_cont[tid];
	photon_cont->y = tstates->photon_y_cont[tid];
	photon_cont->z = tstates->photon_z_cont[tid];

	photon_cont->ux = tstates->photon_ux_cont[tid];
	photon_cont->uy = tstates->photon_uy_cont[tid];
	photon_cont->uz = tstates->photon_uz_cont[tid];

	photon_cont->w = tstates->photon_w_cont[tid];
	photon_cont->dead = tstates->dead_cont[tid];
	photon_cont->hit = tstates->hit[tid];  // Create tstates hit_cont

	photon_cont->MinCos = tstates->MinCos_cont[tid];

	photon_cont->rootIdx = tstates->rootIdx_cont[tid];
	photon_cont->NextTetrahedron = tstates->NextTetrahedron_cont[tid];

	photon_cont->tetrahedron = tstates->tetrahedron;
	photon_cont->faces = tstates->faces;

	photon_cont->s = tstates->photon_s_cont[tid];
	photon_cont->sleft = tstates->photon_sleft_cont[tid];
	photon_cont->OpticalPath = tstates->OpticalPath_cont[tid];
	photon_cont->MaxDepth = tstates->MaxDepth_cont[tid];
	photon_cont->LikelihoodRatio = tstates->LikelihoodRatio_cont[tid];

	photon_cont->FstBackReflectionFlag =
			tstates->FstBackReflectionFlag_cont[tid];
	photon_cont->LocationFstBias = tstates->LocationFstBias_cont[tid];

	photon_cont->NumBackwardsSpecularReflections =
			tstates->NumBackwardsSpecularReflections_cont[tid];

	*is_active = tstates->is_active[tid];
}

__device__ void RestoreThreadState(SimState* d_state, GPUThreadStates* tstates,
		PhotonStructGPU* photon, PhotonStructGPU* photon_cont, UINT64* rnd_x,
		UINT32* rnd_a, UINT64* rnd_xR, UINT32* rnd_aR, UINT64* rnd_xS,
		UINT32* rnd_aS, UINT32* is_active, FLOAT* probe_x, FLOAT* probe_y, FLOAT* probe_z) {
	UINT32 tid = blockIdx.x * blockDim.x + threadIdx.x; // NUM_THREADS_PER_BLOCK

	*rnd_x = d_state->x[tid];
	*rnd_a = d_state->a[tid];
	*rnd_xR = d_state->xR[tid];
	*rnd_aR = d_state->aR[tid];
	*rnd_xS = d_state->xS[tid];
	*rnd_aS = d_state->aS[tid];

	*probe_x = d_state->probe_x;
	*probe_y = d_state->probe_y;
	*probe_z = d_state->probe_z;

	CopytstatesToPhoton(tstates, photon, photon_cont, is_active, tid);
}

__global__ void OCTMPSKernel(SimState* d_state, GPUThreadStates* tstates) {
	// photon structure stored in registers
	PhotonStructGPU photon;
	PhotonStructGPU photon_cont;

	// random number seeds
	UINT64 rnd_x, rnd_xR, rnd_xS;
	UINT32 rnd_a, rnd_aR, rnd_aS;

	// probe locations
	FLOAT probe_x, probe_y, probe_z;

	// Flag to indicate if this thread is active
	UINT32 is_active;

	// Restore the thread state from global memory
	RestoreThreadState(d_state, tstates, &photon, &photon_cont, &rnd_x, &rnd_a,
			&rnd_xR, &rnd_aR, &rnd_xS, &rnd_aS, &is_active, &probe_x, &probe_y, &probe_z);

	for (int iIndex = 0; iIndex < NUM_STEPS; ++iIndex) {
		if (is_active) {

			ComputeStepSize(&photon, &rnd_x, &rnd_a);

			photon.hit = HitBoundary(&photon);

			Hop(&photon);

			if (photon.hit)
				FastReflectTransmit(&photon, d_state, &rnd_xR, &rnd_aR, probe_x, probe_y, probe_z);
			else {
				Drop(&photon);
				switch (d_simparam.TypeBias) {
				case 0:
					Spin(d_regionspecs[photon.tetrahedron[photon.rootIdx].region].g, &photon, &rnd_xS, &rnd_aS);
					break;
				case 37:
					SpinBias(d_regionspecs[photon.tetrahedron[photon.rootIdx].region].g, photon.tetrahedron[photon.rootIdx].region, &photon, &photon_cont, &rnd_xS, &rnd_aS, probe_x, probe_y, probe_z);
					break;
				}
			}

			/***********************************************************
			 *  Roulette()
			 *  If the photon weight is small, the photon packet tries
			 *  to survive a roulette
			 ***********************************************************/
			if (photon.w < WEIGHT) {
				FLOAT rand = rand_MWC_co(&rnd_x, &rnd_a);
				if (photon.w != ZERO_FP && rand < CHANCE) {
					photon.w *= (FLOAT) FAST_DIV(FP_ONE, CHANCE);
				} else if (photon.FstBackReflectionFlag && d_simparam.TypeBias != 3) {
					FLOAT LikelihoodRatioTmp = photon.LikelihoodRatioAfterFstBias;
					CopyPhotonStruct(&photon_cont, &photon);
					Spin(d_regionspecs[photon.tetrahedron[photon.rootIdx].region].g, &photon, &rnd_xS, &rnd_aS);
					if (LikelihoodRatioTmp < 1)
						photon.LikelihoodRatio = 1 - LikelihoodRatioTmp;
					else
						photon.LikelihoodRatio = 1;
				} else if (atomicSub(d_state->n_photons_left, 1) > gridDim.x * blockDim.x) {
					LaunchPhoton(&photon, probe_x, probe_y, probe_z);
				} else
					is_active = 0;
			}
		}
	}
	__syncthreads();

	/**********************************************
	 ** Save the thread state to the global memory
	 **********************************************/
	SaveThreadState(d_state, tstates, &photon, &photon_cont, rnd_x, rnd_a,
			rnd_xR, rnd_aR, rnd_xS, rnd_aS, is_active);
}
#endif  // _OCTMPS_KERNEL_CUH_
