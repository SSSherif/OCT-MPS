#ifndef _OCTMPS_H_
#define _OCTMPS_H_

#include <math.h>
#include <stdio.h>

//#define SINGLE_PRECISION

// Various data types
typedef unsigned long long UINT64;
typedef unsigned int UINT32;
typedef unsigned char BOOLEAN;

// OCTMPS constants
typedef double FLOAT;

#define ZERO_FP 0.0
#define FP_ONE  1.0
#define FP_TWO  2.0

#define PI_const 3.14159265359
#define WEIGHT 1E-4
#define CHANCE 0.1

#define SQ(x) ( (x)*(x) )

//NOTE: Single Precision
#define COSNINETYDEG 1.0E-6
#define COSZERO (1.0F - 1.0E-14)


#define STR_LEN 256

#define NUM_SUBSTEPS_RESOLUTION 6

// Data structure for specifying each region
typedef struct {
    FLOAT mutr;        // Reciprocal mu_total [cm]
    FLOAT mua;        // Absorption coefficient [1/cm]
    FLOAT mus;        // Scattering coefficient [1/cm]
    FLOAT g;            // Anisotropy factor [-]
    FLOAT n;            // Refractive index [-]
} RegionStruct;

// Vertices coordinates
typedef struct {
    int idx;
    double x, y, z;
} Vertex;

// Faces formed with triangles
typedef struct {
    int idx;
    double Nx, Ny, Nz;
    double d;
    Vertex V[3];
} TriangleFaces;

// Tetrahedrons with adjacency information to be used in simulation
struct TetrahedronStruct {
    int index;
    int region;
    TriangleFaces Faces[4];
    struct TetrahedronStruct *adjTetrahedrons[4];
    Vertex Vertices[4];
};

// Probe Position Data
typedef struct {
    double current_scanning_x;
    double start_x;
    double end_x;
    double distance_Ascans;
} ProbeData;

// Simulation input parameters
typedef struct {

    FLOAT probe_x, probe_y, probe_z;

    char outp_filename[STR_LEN];
    char inp_filename[STR_LEN];

    // the starting and ending offset (in the input file) for this simulation
    long begin, end;
    // ASCII or binary output
    char AorB;

    UINT32 number_of_photons;
    UINT32 rootIdx;

    FLOAT Rspecular;

    // Tetrahedrons variables
    short num_tetrahedrons;
    short number_of_regions;
    ProbeData probe;

    // Bias variables
    short int TypeSimulation;
    short int TypeBias;
    FLOAT BackwardBiasCoefficient;
    FLOAT TargetOpticalDepth;
    FLOAT CoherenceLengthSource;

    FLOAT TargetDepthMin;
    FLOAT TargetDepthMax;

    short RecordPhotonsFlag;
    FLOAT MaxCollectingRadius;
    FLOAT MaxCollectingAngleDeg;

    FLOAT ProbabilityAdditionalBias;
    FLOAT MaxRelativeContributionToBinPerPostProcessedSample;

    FILE *output_file_ptr; // check what's used in simulation from this file
    long int NumFilteredPhotons;
    long int NumFilteredPhotonsClassI;
    long int NumFilteredPhotonsClassII;

    FLOAT MaxFilteredOpticalDepth;
    short NumOpticalDepthLengthSteps;
    FLOAT OpticalDepthShift;
    FLOAT rndStepSizeInTissue;

    UINT32 n_regions; // Number of regions including 1 region (with index 0) for ambient medium
    RegionStruct *regions;
} SimulationStruct;

// Per-GPU simulation states
// One instance of this struct exists in the host memory, while the other
// in the global memory.
typedef struct {

    FLOAT probe_x, probe_y, probe_z;

    // points to a scalar that stores the number of photons that are not
    // completed (i.e. either on the fly or not yet started)
    UINT32 *n_photons_left;

    // per-thread seeds for random number generation
    // arrays of length NUM_THREADS
    // We put these arrays here as opposed to in GPUThreadStates because
    // they live across different simulation runs and must be copied back
    // to the host.

    UINT32 *a;  // General
    UINT32 *aR; // ReflectT
    UINT32 *aS; // Spin

    UINT64 *x;  // General
    UINT64 *xR; // ReflectT
    UINT64 *xS; // Spin

    UINT64 NumFilteredPhotons;
    UINT64 NumFilteredPhotonsClassI;
    UINT64 NumFilteredPhotonsClassII;

    UINT64 *NumClassI_PhotonsFilteredInRange;
    UINT64 *NumClassII_PhotonsFilteredInRange;

    FLOAT *ReflectanceClassI;
    FLOAT *ReflectanceClassI_Sum;
    FLOAT *ReflectanceClassI_Max;
    FLOAT *ReflectanceClassI_SumSq;

    FLOAT *ReflectanceClassII;
    FLOAT *ReflectanceClassII_Sum;
    FLOAT *ReflectanceClassII_Max;
    FLOAT *ReflectanceClassII_SumSq;

    FLOAT *MeanReflectanceClassI_Sum;
    FLOAT *MeanReflectanceClassI_SumSq;
    FLOAT *MeanReflectanceClassII_Sum;
    FLOAT *MeanReflectanceClassII_SumSq;
} SimState;

typedef struct {
    FLOAT *ReflectanceClassI;
    FLOAT *ReflectanceClassI_Max;
    FLOAT *ReflectanceClassI_Sum;
    FLOAT *ReflectanceClassI_SumSq;
    unsigned long int *NumClassI_PhotonsFilteredInRange;

    FLOAT *ReflectanceClassII;
    FLOAT *ReflectanceClassII_Max;
    FLOAT *ReflectanceClassII_Sum;
    FLOAT *ReflectanceClassII_SumSq;
    unsigned long int *NumClassII_PhotonsFilteredInRange;

} OutStruct;

typedef struct {
    int idx, region;
    int faces[4], signs[4], adjTetras[4];
} TetrahedronStructGPU;

// Everything a host thread needs to know in order to run simulation on
// one GPU (host-side only)
typedef struct {
    // GPU identifier
    unsigned int dev_id;

    // those states that will be updated
    SimState host_sim_state;

    // simulation input parameters
    SimulationStruct *sim;

    // number of thread blocks launched
    UINT32 n_tblks;

    // root tetrahedron and faces
    TetrahedronStructGPU *root;
    TriangleFaces *faces;

    //number of tetrahedrons in actual use by RootTetrahedron
    UINT32 n_tetras;
    UINT32 n_faces;
    UINT32 rootIdx;
} HostThreadState;

extern void usage(const char *prog_name);

extern void ShowVersion(char const *version);

// Parse the command-line arguments.
// Return 0 if successful or a +ive error code.
extern int interpret_arg(int argc, char *argv[], char **fpath_o, char **fpath_m,
                         char **fpath_b, unsigned long long *seed, unsigned int *num_GPUs);

extern int read_simulation_data(char *filename, SimulationStruct **simulations);

extern void FreeSimulationStruct(SimulationStruct *sim, int n_simulations);

// IO functions to be used with mesh and bias files
extern int readints(int n_ints, int *temp, FILE *pFile);

extern int readfloats(int n_floats, float *temp, FILE *pFile);

// Meshing
extern int read_mesh_param(char *filename, int *number_of_vertices,
                           int *number_of_tetrahedrons);

extern TetrahedronStruct *MeshGraph(char *filename, SimulationStruct **simulations);

extern TetrahedronStruct *FindRootofGraph(TetrahedronStruct *tetrahedrons, SimulationStruct **simulations);

extern void SerializeGraph(TetrahedronStruct *RootTetrahedron, UINT32 number_of_vertices, UINT32 number_of_faces,
                           UINT32 number_of_tetrahedrons,
                           Vertex *d_vertices, TriangleFaces *d_faces, TetrahedronStructGPU *d_root);

// Biasing
extern int Bias_Read(char *filename, SimulationStruct **simulations);

extern int Write_Bias_Parameters(SimState *HostMem, SimulationStruct *sim,
                                 float simulation_time);

/***********************************************************
 *	Routine prototypes for dynamic memory allocation and
 *	release of arrays and matrices.
 *	Modified from Numerical Recipes in C.
 **********************************************************/
FLOAT *AllocVector(short, short);

FLOAT **AllocMatrix(short, short, short, short);

FLOAT ***AllocHyperMatrix(short, short, short, short, short, short);

void FreeVector(FLOAT *, short, short);

void FreeMatrix(FLOAT **, short, short, short, short);

void FreeHyperMatrix(FLOAT ***, short, short, short, short, short, short);

void nrerror(char *);

extern FLOAT Rspecular(FLOAT ni, FLOAT nt);

extern void InitOutputData(SimulationStruct In_Parm, OutStruct *Out_Ptr);

extern void WriteResult(SimulationStruct *In_Parm, SimState *Out_Parm,
                        float *TimeReport);

#endif  // _OCTMPS_H_
