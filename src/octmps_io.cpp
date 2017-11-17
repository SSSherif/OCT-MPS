#define NFLOATS 5
#define NDOUBLES 5
#define NINTS 5

#include <time.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>

#include "octmps.h"

/***************************************************
**  Center a string according to column width (80)
***************************************************/
#define COL_WIDTH 80
void CtrPuts(char const *InStr) {
  short       nspaces;  //number of spaces to be left-filled
  char        outstr[STR_LEN];

  nspaces = (COL_WIDTH - strlen(InStr)) / 2;
  if (nspaces < 0)
    nspaces = 0;

  strcpy(outstr, "");
  while (nspaces--)
    strcat(outstr, " ");

  strcat(outstr, InStr);

  puts(outstr);
}

/*******************************************************************
**  General information - Authors, affiliation, version, year, etc
*******************************************************************/
void ShowVersion(char const *version) {
  char str[STR_LEN];

  CtrPuts(" ");
  CtrPuts("Massively Parallel Simulator of Optical Coherence Tomography of Inhomogeneous Turbid Media");
  CtrPuts(version);
  puts("\n");

  CtrPuts("Siavash Malektaji, Mauricio R. Escobar I., Ivan T. Lima Jr., and Sherif S. Sherif");
  puts("\n");

  CtrPuts("Department of Electrical and Computer Engineering, University of "
                  "Manitoba");
  CtrPuts("Winnipeg, Manitoba, Canada");
  CtrPuts("Email: Sherif.Sherif@umanitoba.ca");
  puts("\n");

  CtrPuts("\tThe program is written based on the MCML, GPUMCML and TIM-OS codes, and the following paper:\n\n");

  printf("\t(1) S. Malektaji, Ivan .T Lima Jr. Sherif S. Sherif \"Monte Carlo Simulation of Optical\n");
  printf("\tCoherence Tomography for Turbid Media with Arbitrary Spatial Distribution\", Journal of\n");
  printf("\tBiomedical Optics 19.4 (2014): 046001-046001.\n\n");

  printf("\t(2) L.-H. Wang, S. L. Jacques, and L.-Q. Zheng, \"MCML - Monte Carlo modeling of photon\n");
  printf("\ttransport in multi-layered tissues\", Computer Methods and Programs in Biomedicine, 47,\n");
  printf("\t131-146, 1995.\n\n");

  printf("\t(3) H. Shen and G. Wang. \"Tetrahedron-based inhomogeneous Monte-Carlo optical simulator.\"\n");
  printf("\tPhys. Med. Biol. 55:947-962, 2010.\n\n");

  printf("\t(4) Alerstam, Erik, Tomas Svensson, and Stefan Andersson-Engels. \"Parallel computing with\n");
  printf("\tgraphics processing units for high-speed Monte Carlo simulation of photon migration.\"\n");
  printf("\tJournal of biomedical optics 13.6 (2008): 060504-060504..\n");
  puts("\n");

}
#undef COLWIDTH

/***********************************************************
 *	Get the file name of the input data file from the
 *	argument to the command line.
 **********************************************************/
void GetFnameFromArgv(int argc,
                      char * argv[],
                      char * input_filename) {
  if(argc>=2) /* filename in command line */
    strcpy(input_filename, argv[1]);
  else
    input_filename[0] = '\0';
}

/************************************************************************
**  Print Command Line Help - How to run program and pass in parameters
************************************************************************/
void usage(const char *prog_name) {
  printf("\nUsage: %s [-S<seed>] [-G<num GPUs>] <input opt file> <input mesh file> <input bias file>\n\n",prog_name);

  printf("  -S: seed for random number generation (MT only)\n");
  printf("  -G: set the number of GPUs this program uses\n");
  printf("\n");
  fflush(stdout);
}

/*********************************
**  Parse command line arguments
*********************************/
int interpret_arg(int argc, char* argv[], char **fpath_o, char **fpath_m,
                  char **fpath_b, unsigned long long* seed,  unsigned int *num_GPUs) {
  int i;
  char *fpath_opt = NULL;
  char *fpath_mesh = NULL;
  char *fpath_bias = NULL;

  for (i = 1; i < argc; ++i)
  {
    char *arg = argv[i];

    if (arg[0] != '-')
    {
      // This is the input file path.
      fpath_opt  = arg;
      fpath_mesh = argv[++i];
      fpath_bias = argv[++i];
      // Ignore the remaining args.
      break;
    }

    // Skip the '-'.
    ++arg;

    if (sscanf(arg, "S%llu", seed) == 1)
    {
      // <seed> has been set.
    }
    else if (sscanf(arg, "G%u", num_GPUs) == 1)
    {
      // <num_GPUs> has been set.
    }

  }

  if (fpath_o != NULL) *fpath_o = fpath_opt;
  if (fpath_m != NULL) *fpath_m = fpath_mesh;
  if (fpath_b != NULL) *fpath_b = fpath_bias;

  return (fpath_opt == NULL);
}

/********************************************
**	Write the input parameters to the file.
*********************************************/
void WriteInParm(FILE *file, SimulationStruct * sim) {
  unsigned int i;

  fprintf(file,
          "InParm \t\t\t# Input parameters. cm is used.\n");

  fprintf(file, "%s \tA\t\t# output file name, ASCII.\n", sim->outp_filename);
  fprintf(file, "%u \t\t\t# No. of photons\n", sim->number_of_photons);

  fprintf(file, "%u\t\t\t\t\t# Number of regions\n", sim->n_regions);

  fprintf(file, "#n\tmua\tmus\tg\td\t# One line for each region\n");
  fprintf(file, "%G\t\t\t\t\t# n for medium above\n", sim->regions[0].n);

  for(i=1; i<=sim->n_regions; i++)  {
    fprintf(file, "%G\t%G\t%G\t%G\t# region %hd\n",
            sim->regions[i].n, sim->regions[i].mua, 1/sim->regions[i].mutr-sim->regions[i].mua, sim->regions[i].g, i);
  }
  fprintf(file, "%G\t\t\t\t\t# n for medium below\n", sim->regions[i].n);
}

/***********************************************************
**	Allocate an array with index from nl to nh inclusive.
**
**	Original matrix and vector from Numerical Recipes in C
**	don't initialize the elements to zero. This will
**	be accomplished by the following functions.
************************************************************/
FLOAT *AllocVector(short nl, short nh) {
  FLOAT *v;
  short i;

  v=(FLOAT *)malloc((unsigned) (nh-nl+1)*sizeof(FLOAT));
  if (!v) {
    perror("allocation failure in vector()");
    exit(1);
  }

  v -= nl;
  for(i = nl; i <= nh; i++) v[i] = 0.0;	/* init. */
  return v;
}

/***********************************************************
**	Allocate the arrays in OutStruct for one run, and
**	array elements are automatically initialised to zeros
***********************************************************/
void InitOutputData(SimulationStruct In_Parm,
                    OutStruct *Out_Ptr) {
  short number_of_regions = In_Parm.number_of_regions;

  short nt = In_Parm.num_tetrahedrons;

  Out_Ptr->ReflectanceClassI        = AllocVector(0,
                                                  In_Parm.NumOpticalDepthLengthSteps + 1);
  Out_Ptr->ReflectanceClassI_Max    = AllocVector(0,
                                                  In_Parm.NumOpticalDepthLengthSteps + 1);
  Out_Ptr->ReflectanceClassI_Sum    = AllocVector(0,
                                                  In_Parm.NumOpticalDepthLengthSteps + 1);
  Out_Ptr->ReflectanceClassI_SumSq  = AllocVector(0,
                                                  In_Parm.NumOpticalDepthLengthSteps + 1);

  Out_Ptr->ReflectanceClassII        = AllocVector(0,
                                                   In_Parm.NumOpticalDepthLengthSteps + 1);
  Out_Ptr->ReflectanceClassII_Max    = AllocVector(0,
                                                   In_Parm.NumOpticalDepthLengthSteps + 1);
  Out_Ptr->ReflectanceClassII_Sum    = AllocVector(0,
                                                   In_Parm.NumOpticalDepthLengthSteps + 1);
  Out_Ptr->ReflectanceClassII_SumSq  = AllocVector(0,
                                                   In_Parm.NumOpticalDepthLengthSteps + 1);

  Out_Ptr->NumClassI_PhotonsFilteredInRange = (unsigned long *) malloc(
          (In_Parm.NumOpticalDepthLengthSteps + 1) * sizeof(unsigned long int ) );
  Out_Ptr->NumClassII_PhotonsFilteredInRange = (unsigned long *) malloc(
          (In_Parm.NumOpticalDepthLengthSteps + 1) * sizeof(unsigned long int ) );

  unsigned long int jj;
  for (jj = 0; jj < In_Parm.NumOpticalDepthLengthSteps; jj++) {
    Out_Ptr->NumClassI_PhotonsFilteredInRange[jj] = 0;
    Out_Ptr->NumClassII_PhotonsFilteredInRange[jj] = 0;
  }

}

/******************************************
**  Preparing Bias file by Ivan Lima 2009
******************************************/
int Write_Bias_Parameters(SimState *HostMem,
                          SimulationStruct *sim,
                          float simulation_time) {
  FILE *output_bias_file_ptr;
  sim->output_file_ptr = fopen("FilteredPhotons.mco","w");
  output_bias_file_ptr = fopen("BiasParam.mco","w");

  if (output_bias_file_ptr == NULL){
    perror ("Error opening output file"); return 0;
  }

  fprintf(output_bias_file_ptr, "%llu\t\t\t\t\t# Number of filtered photons\n",
          HostMem->NumFilteredPhotons);

  fprintf(output_bias_file_ptr,
          "%llu\t\t\t\t\t# Number of Class I filtered photons\n",
          HostMem->NumFilteredPhotonsClassI );

  fprintf(output_bias_file_ptr,
          "%llu\t\t\t\t\t# Number of Class II filtered photons\n",
          HostMem->NumFilteredPhotonsClassII );

  if (sim->RecordPhotonsFlag) {
    fprintf(output_bias_file_ptr,"\n\n\n\n");
    fprintf(output_bias_file_ptr,
            "Description of fields in file FilteredPhotons.mco\n\n");

    fprintf(output_bias_file_ptr,
            "================================================================\n");

    fprintf(output_bias_file_ptr,
            "Weight | LikelihoodRatio | Radius | AngleRad | OpticalPath | MaxDepth");

  }

  fprintf(output_bias_file_ptr,"%hd\t\t\t\t\t# Type of simulation\n",
          sim->TypeSimulation);
  fprintf(output_bias_file_ptr,"%hd\t\t\t\t\t# Type of bias\n",
          sim->TypeBias);
  fprintf(output_bias_file_ptr,"%u\t\t\t\t\t# Number of photon packets\n",
          sim->number_of_photons);
  fprintf(output_bias_file_ptr,"%G\t\t\t\t\t# Backward bias coefficient\n",
          sim->BackwardBiasCoefficient);
  fprintf(output_bias_file_ptr,"%G\t\t\t\t\t# Target depth min [cm]\n",
          sim->TargetDepthMin);
  fprintf(output_bias_file_ptr,"%G\t\t\t\t\t# Target depth max [cm]\n",
          sim->TargetDepthMax);
  fprintf(output_bias_file_ptr,"%G\t\t\t\t\t# Coherence length source [cm]\n",
          sim->CoherenceLengthSource);
  fprintf(output_bias_file_ptr,"%G\t\t\t\t\t# Max collecting radius [cm]\n",
          sim->MaxCollectingRadius);
  fprintf(output_bias_file_ptr,"%G\t\t\t\t\t# Max collecting angle [grad]\n",
          sim->MaxCollectingAngleDeg);
  fprintf(output_bias_file_ptr,"%hd\t\t\t\t\t# Record photons flag\n",
          sim->RecordPhotonsFlag);
  fprintf(output_bias_file_ptr,"%G\t\t\t\t\t# Probability of additional bias\n",
          sim->ProbabilityAdditionalBias);
  fprintf(output_bias_file_ptr,
          "%G\t\t\t\t\t# Maximum relative contribution to bin per post-processed sample\n",
          sim->MaxRelativeContributionToBinPerPostProcessedSample);

  fclose(output_bias_file_ptr);
  return 0;
}

void WriteReflectanceClassI_and_ClassII(SimulationStruct *In_Ptr,
                                        SimState *Out_Ptr,
                                        FILE *OutFileClassI_Scatt,
                                        FILE *OutFileClassII_Scatt) {

  for (long int depth_index = 0; depth_index < In_Ptr->NumOpticalDepthLengthSteps; depth_index++) {

    //## Class I parameters
    FLOAT MeanDiffusiveReflectance =  Out_Ptr->ReflectanceClassI_Sum[depth_index]
                                      /In_Ptr->number_of_photons;

    /* The mean value will be stored in Out_Ptr->MeanRefulectanceClassI_Sum[jj]. */
    Out_Ptr->ReflectanceClassI[depth_index] = MeanDiffusiveReflectance;

    FLOAT VarianceDiffusiveReflectanceClassI = ( Out_Ptr->ReflectanceClassI_SumSq[depth_index]/In_Ptr->number_of_photons
                                                 -MeanDiffusiveReflectance*MeanDiffusiveReflectance)
                                               /(In_Ptr->number_of_photons - 1);

    //## Class II parameters
    FLOAT MeanDiffusiveReflectanceClassII =  Out_Ptr->ReflectanceClassII_Sum[depth_index]
                                             /In_Ptr->number_of_photons;

    /* The mean value will be stored in Out_Ptr->MeanReflectanceClassII_Sum[jj]. */
    Out_Ptr->ReflectanceClassII[depth_index] = MeanDiffusiveReflectanceClassII;

    FLOAT VarianceDiffusiveReflectanceClassII
            = ( Out_Ptr->ReflectanceClassII_SumSq[depth_index]/In_Ptr->number_of_photons
                -MeanDiffusiveReflectanceClassII
                 *MeanDiffusiveReflectanceClassII)
              /(In_Ptr->number_of_photons - 1);

    FLOAT OpticalDepthTmp = depth_index*In_Ptr->CoherenceLengthSource
                            /NUM_SUBSTEPS_RESOLUTION
                            +In_Ptr->OpticalDepthShift;

    fprintf(OutFileClassI_Scatt,"%G,%G,%G,%G,%llu\n", In_Ptr->probe.current_scanning_x, OpticalDepthTmp,
            MeanDiffusiveReflectance,
            sqrt(VarianceDiffusiveReflectanceClassI),
            Out_Ptr->NumClassI_PhotonsFilteredInRange[depth_index]);

    fprintf(OutFileClassII_Scatt,"%G,%G,%G,%G,%llu\n", In_Ptr->probe.current_scanning_x, OpticalDepthTmp,
            MeanDiffusiveReflectanceClassII,
            sqrt(VarianceDiffusiveReflectanceClassII),
            Out_Ptr->NumClassII_PhotonsFilteredInRange[depth_index]);
  }

}


/*********************************************************************************
**  Write the reflectance of classes I and II Filtering up to one sample per bin
*********************************************************************************/
void WriteReflectanceClassI_and_ClassII_Filtered(SimulationStruct *In_Ptr,
                                                 SimState *Out_Ptr,
                                                 FILE *OutFileClassI_Scatt,
                                                 FILE *OutFileClassII_Scatt) {

  long int depth_index;
  for (depth_index = 0; depth_index < In_Ptr->NumOpticalDepthLengthSteps; depth_index++) {
    if ( Out_Ptr->ReflectanceClassI_Max[depth_index] >  In_Ptr->MaxRelativeContributionToBinPerPostProcessedSample
                                                        *Out_Ptr->ReflectanceClassI_Sum[depth_index] ) {
      Out_Ptr->ReflectanceClassI_Sum[depth_index]   -= Out_Ptr->ReflectanceClassI_Max[depth_index];
      Out_Ptr->ReflectanceClassI_SumSq[depth_index] -= SQ(Out_Ptr->ReflectanceClassI_Max[depth_index]);
      Out_Ptr->MeanReflectanceClassI_Sum[depth_index]   -= Out_Ptr->ReflectanceClassI_Max[depth_index];
      Out_Ptr->MeanReflectanceClassI_SumSq[depth_index] -= SQ(Out_Ptr->ReflectanceClassI_Max[depth_index]);
    }

    if ( Out_Ptr->ReflectanceClassII_Max[depth_index] >  In_Ptr->MaxRelativeContributionToBinPerPostProcessedSample
                                                         *Out_Ptr->ReflectanceClassII_Sum[depth_index] ) {
      Out_Ptr->ReflectanceClassII_Sum[depth_index]   -= Out_Ptr->ReflectanceClassII_Max[depth_index];
      Out_Ptr->ReflectanceClassII_SumSq[depth_index] -= SQ(Out_Ptr->ReflectanceClassII_Max[depth_index]);
      Out_Ptr->MeanReflectanceClassII_Sum[depth_index]   -= Out_Ptr->ReflectanceClassII_Max[depth_index];
      Out_Ptr->MeanReflectanceClassII_SumSq[depth_index] -= SQ(Out_Ptr->ReflectanceClassII_Max[depth_index]);
    }


    //Class I parameters
    FLOAT MeanDiffusiveReflectanceClassI =  Out_Ptr->ReflectanceClassI_Sum[depth_index]
                                            /In_Ptr->number_of_photons;

    /* The mean value will be stored in Out_Ptr->MeanReflectanceClassI_Sum[depth_index]. */
    Out_Ptr->ReflectanceClassI[depth_index] = MeanDiffusiveReflectanceClassI;

    FLOAT VarianceDiffusiveReflectance
            = ( Out_Ptr->ReflectanceClassI_SumSq[depth_index]/In_Ptr->number_of_photons
                - MeanDiffusiveReflectanceClassI * MeanDiffusiveReflectanceClassI)
              / (In_Ptr->number_of_photons - 1);

    // Class II parameters
    FLOAT MeanDiffusiveReflectanceClassII =  Out_Ptr->ReflectanceClassII_Sum[depth_index]
                                             /In_Ptr->number_of_photons;

    /* The mean value will be stored in Out_Ptr->MeanReflectanceClassII_Sum[depth_index]. */
    Out_Ptr->ReflectanceClassII[depth_index] = MeanDiffusiveReflectanceClassII;

    FLOAT VarianceDiffusiveReflectanceClassII
            = ( Out_Ptr->ReflectanceClassII_SumSq[depth_index] / In_Ptr->number_of_photons
                -MeanDiffusiveReflectanceClassII
                 *MeanDiffusiveReflectanceClassII)
              /(In_Ptr->number_of_photons - 1);

    FLOAT OpticalDepthTmp = depth_index*In_Ptr->CoherenceLengthSource
                            /NUM_SUBSTEPS_RESOLUTION
                            +In_Ptr->OpticalDepthShift;

    fprintf(OutFileClassI_Scatt,"%G,%G,%G,%G,%llu\n",
            In_Ptr->probe.current_scanning_x, OpticalDepthTmp,
            MeanDiffusiveReflectanceClassI,
            sqrt(VarianceDiffusiveReflectance),
            Out_Ptr->NumClassI_PhotonsFilteredInRange[depth_index]);

    fprintf(OutFileClassII_Scatt,"%G,%G,%G,%G,%llu\n",
            In_Ptr->probe.current_scanning_x, OpticalDepthTmp,
            MeanDiffusiveReflectanceClassII,
            sqrt(VarianceDiffusiveReflectanceClassII),
            Out_Ptr->NumClassII_PhotonsFilteredInRange[depth_index]);
  }

}

void WriteResult(SimulationStruct *In_Parm,
                 SimState *Out_Parm,
                 float *TimeReport) {

  FILE *OutFileClassI_Scatt = fopen("ClassI_Scatt.out", "a");
  FILE *OutFileClassII_Scatt = fopen("ClassII_Scatt.out", "a");
  WriteReflectanceClassI_and_ClassII(In_Parm, Out_Parm, OutFileClassI_Scatt, OutFileClassII_Scatt);
  fclose(OutFileClassI_Scatt);
  fclose(OutFileClassII_Scatt);

  OutFileClassI_Scatt = fopen("ClassI_ScattFilt.out", "a");
  OutFileClassII_Scatt = fopen("ClassII_ScattFilt.out", "a");
  WriteReflectanceClassI_and_ClassII_Filtered(In_Parm, Out_Parm, OutFileClassI_Scatt, OutFileClassII_Scatt);
  fclose(OutFileClassI_Scatt);
  fclose(OutFileClassII_Scatt);
}

int isnumeric(char a) {
  if(a>=(char)48 && a<=(char)57) return 1;
  else return 0;
}

int readfloats(int n_floats, float* temp, FILE* pFile) {
  int ii=0;
  char mystring [200];

  if(n_floats>NFLOATS) return 0; //cannot read more than NFLOATS floats

  while(ii<=0) {
    if(feof(pFile)) return 0; //if we reach EOF here something is wrong with the file!
    fgets(mystring , 200 , pFile);
    memset(temp,0,NFLOATS*sizeof(float));
    ii=sscanf(mystring,"%f %f %f %f %f",&temp[0],&temp[1],&temp[2],&temp[3],&temp[4]);
    if(ii>n_floats) return 0;
    //if we read more number than defined something is wrong with the file!
    //printf("ii=%d temp=%f %f %f %f %f\n",ii,temp[0],temp[1],temp[2],temp[3],temp[4]);
  }
  return 1; // Everyting appears to be ok!
}

int readdoubles(int n_doubles, FLOAT* temp, FILE* pFile) {
  int ii=0;
  char mystring [200];

  if(n_doubles>NFLOATS) return 0; //cannot read more than NFLOATS floats

  while(ii<=0) {
    if(feof(pFile)) return 0; //if we reach EOF here something is wrong with the file!
    fgets(mystring , 200 , pFile);
    memset(temp,0,NFLOATS*sizeof(FLOAT));
    ii=sscanf(mystring,"%lf %lf %lf %lf %lf",&temp[0],&temp[1],&temp[2],&temp[3],&temp[4]);
    if(ii>n_doubles) return 0;
    //if we read more number than defined something is wrong with the file!
  }
  return 1; // Everyting appears to be ok!
}

int readints(int n_ints, int* temp, FILE* pFile) {
  int ii=0;
  char mystring[STR_LEN];

  if(n_ints>NINTS) return 0; //cannot read more than NINTS integers

  while(ii<=0) {
    if(feof(pFile)) return 0; //if we reach EOF here something is wrong with the file!
    fgets(mystring , STR_LEN , pFile);
    memset(temp,0,NINTS*sizeof(int));
    ii=sscanf(mystring,"%d %d %d %d %d",&temp[0],&temp[1],&temp[2],&temp[3],&temp[4]);
    if(ii>n_ints) return 0;
    //if we read more number than defined something is wrong with the file!
  }
  return 1; // Everything appears to be ok!
}

int ischar(char a) {
  if((a>=(char)65 && a<=(char)90)||(a>=(char)97 && a<=(char)122)) return 1;
  else return 0;
}

/********************************
**  Parse simulation input file
********************************/
int read_simulation_data(char* filename,
                         SimulationStruct** simulations) {
  int i=0;
  int ii=0;
  unsigned int number_of_photons;
  int n_simulations = 0;
  int n_regions = 0;
  FILE * pFile;
  char mystring [STR_LEN];
  char str[STR_LEN];
  char AorB;
  FLOAT dtot=0;


  float ftemp[NFLOATS];//Find a more elegant way to do this...
  FLOAT dtemp[NDOUBLES];
  int itemp[NINTS];

  FLOAT n1, n2, r;

  pFile = fopen(filename , "r");
  if (pFile == NULL){
    perror ("Error opening file");
    return 0;
  }

  // First read the first data line (file version) and ignore
  if(!readfloats(1, ftemp, pFile)){
    perror ("Error reading file version");
    return 0;
  }
  //printf("File version: %f\n",ftemp[0]);

  // Second, read the number of runs
  if(!readints(1, itemp, pFile)){
    perror ("Error reading number of runs");
    return 0;
  }
  n_simulations = itemp[0];

  // Allocate memory for the SimulationStruct array
  *simulations = (SimulationStruct*) malloc(sizeof(SimulationStruct)*n_simulations);
  if(*simulations == NULL){
    perror("Failed to malloc simulations.\n");
    return 0;
  }

  for(i=0; i<n_simulations; i++)
  {
    // Store the input filename
    strcpy((*simulations)[i].inp_filename,filename);

    // Read the output filename and determine ASCII or Binary output
    ii=0;
    while(ii<=0)
    {
      (*simulations)[i].begin=ftell(pFile);
      fgets (mystring , STR_LEN , pFile);
      ii=sscanf(mystring,"%s %c",str,&AorB);
      if(feof(pFile)|| ii>2){perror("Error reading output filename");return 0;}
      if(ii>0)ii=ischar(str[0]);
    }

    strcpy((*simulations)[i].outp_filename,str);
    (*simulations)[i].AorB=AorB;

    // Read the number of photons
    ii=0;
    while(ii<=0) {
      fgets(mystring , STR_LEN , pFile);
      number_of_photons=0;
      ii=sscanf(mystring,"%u",&number_of_photons);
      if(feof(pFile) || ii>1){
        perror("Error reading number of photons");
        return 0;
      }
    }

    //printf("Number of photons: %lu\n",number_of_photons);
    (*simulations)[i].number_of_photons=number_of_photons;

    // Read No. of regions (1xint)
    if(!readints(1, itemp, pFile)){
      perror ("Error reading No. of regions");
      return 0;
    }
    printf("\n====================================\n");
    printf("Simulation Parameters:\n");

    n_regions = itemp[0]+1;
    printf("No. of regions (including ambient medium) =%d\n",n_regions);
    (*simulations)[i].n_regions = n_regions;

    // Allocate memory for the regions (including one for the upper and one for the lower)
    (*simulations)[i].regions = (RegionStruct*) malloc(sizeof(RegionStruct)*(n_regions));
    if((*simulations)[i].regions == NULL){
      perror("Failed to malloc regions.\n");
      return 0;
    }

    // Read ambient medium refractive index (1x double)
    if(!readdoubles(1, dtemp, pFile)){
      perror ("Error reading ambient medium refractive index");
      return 0;
    }
    printf("Ambient medium refractive index=%f\n",ftemp[0]);
    (*simulations)[i].regions[0].n=dtemp[0];

    dtot=0;
    for(ii=1;ii<n_regions;ii++)
    {
      // Read Region data (5x double)
      if(!readdoubles(4, dtemp, pFile)){
        perror ("Error reading region data");
        return 0;
      }
      printf("Region %d optical Properties: n=%f,\tmua=%f,\tmus=%f,\tg=%f\n",ii,dtemp[0],dtemp[1],dtemp[2],dtemp[3]);
      (*simulations)[i].regions[ii].n     = dtemp[0];
      (*simulations)[i].regions[ii].mua   = dtemp[1];
      (*simulations)[i].regions[ii].mus   = dtemp[2];
      (*simulations)[i].regions[ii].g     = dtemp[3];
      if(dtemp[2]==0.0L)(*simulations)[i].regions[ii].mutr=DBL_MAX; //Glass region
      else(*simulations)[i].regions[ii].mutr = FP_ONE / (dtemp[1]+dtemp[2]);
    }
    printf("====================================\n");


    (*simulations)[i].end=ftell(pFile);

    // Read Probe Position Data
    if(!readdoubles(3, dtemp, pFile)){
      perror ("Error reading probe position data parameters");
      return 0;
    }
    simulations[i]->probe.start_x  = dtemp[0];
    simulations[i]->probe.current_scanning_x = dtemp[0]; // First time
    simulations[i]->probe.distance_Ascans      = dtemp[1];
    simulations[i]->probe.end_x    = dtemp[2];

  }
  return n_simulations;
}

/*************************************************************
**  Read number of vertices and tetrahedrons from mesh file
*************************************************************/
int read_mesh_param(char *filename,
                    int *number_of_vertices,
                    int *number_of_tetrahedrons) {
  FILE *pFile;
  int itemp[NINTS];
  int i;

  pFile = fopen(filename , "r");
  if(pFile == NULL){
    perror ("Error opening mesh file");
    return 0;
  }

  // First, read the number of vertices
  if(!readints(1, itemp, pFile)){
    perror ("Error reading number of vertices");
    return 0;
  }
  *number_of_vertices = itemp[0];

  // Second, read the number of tetrahedrons
  if(!readints(1, itemp, pFile)){
    perror ("Error reading number of tetrahedrons");
    return 0;
  }
  *number_of_tetrahedrons = itemp[0];

  return 1;
}

void FreeSimulationStruct(SimulationStruct* sim, int n_simulations) {
  int i;
  for(i=0;i<n_simulations;i++)free(sim[i].regions);
  free(sim);
}
