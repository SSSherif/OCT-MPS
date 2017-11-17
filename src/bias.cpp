#define NFLOATS 5
#define NINTS 5

#include "octmps.h"

/****************************
**  Reading Bias parameters
/****************************/
int Bias_Read(char *filename,
              SimulationStruct **simulations) {

    FILE *pFile;
    FILE *output_bias_file_ptr;
    float ftemp[NFLOATS];
    int itemp[NINTS];

    pFile = fopen(filename, "r");
    if (pFile == NULL) {
        perror("Error opening bias parameter file!");
        return 0;
    }

    // Read all bias parameters
    if (!readints(1, itemp, pFile)) {
        perror("Error reading type of simulation!");
        return 0;
    }
    (*simulations)->TypeSimulation = itemp[0];

    if (!readints(1, itemp, pFile)) {
        perror("Error reading type of bias!");
        return 0;
    }
    (*simulations)->TypeBias = itemp[0];

    if (!readfloats(1, ftemp, pFile)) {
        perror("Error reading backward bias coefficient!");
        return 0;
    }
    (*simulations)->BackwardBiasCoefficient = ftemp[0];

    if (!readfloats(1, ftemp, pFile)) {
        perror("Error reading minimum target depth!");
        return 0;
    }
    (*simulations)->TargetDepthMin = ftemp[0];

    if (!readfloats(1, ftemp, pFile)) {
        perror("Error reading maximum target depth!");
        return 0;
    }
    (*simulations)->TargetDepthMax = ftemp[0];

    if (!readfloats(1, ftemp, pFile)) {
        perror("Error reading coherence length source!");
        return 0;
    }
    (*simulations)->CoherenceLengthSource = ftemp[0];

    // Read recording photons flag: 1=>All, 2=>Filtered
    if (!readints(1, itemp, pFile)) {
        perror("Error reading recording type!");
        return 0;
    }
    (*simulations)->RecordPhotonsFlag = itemp[0];

    // Read rest of bias
    if (!readfloats(1, ftemp, pFile)) {
        perror("Error reading maximum collecting radius!");
        return 0;
    }
    (*simulations)->MaxCollectingRadius = ftemp[0];

    if (!readfloats(1, ftemp, pFile)) {
        perror("Error reading maximum collecting angle!");
        return 0;
    }
    (*simulations)->MaxCollectingAngleDeg = ftemp[0];

    if (!readfloats(1, ftemp, pFile)) {
        perror("Error reading probability additional bias!");
        return 0;
    }
    (*simulations)->ProbabilityAdditionalBias = ftemp[0];

    if (!readfloats(1, ftemp, pFile)) {
        perror("Error reading maximum relative contribution per bin!");
        return 0;
    }
    (*simulations)->MaxRelativeContributionToBinPerPostProcessedSample = ftemp[0];

    const float ConstMaxFixedOpticalDepth = 0.12;
    if ((*simulations)->TargetDepthMax < ConstMaxFixedOpticalDepth)
        (*simulations)->MaxFilteredOpticalDepth = ConstMaxFixedOpticalDepth;
    else
        (*simulations)->MaxFilteredOpticalDepth = (*simulations)->TargetDepthMax;


    (*simulations)->NumOpticalDepthLengthSteps = NUM_SUBSTEPS_RESOLUTION *
                                                 (*simulations)->MaxFilteredOpticalDepth /
                                                 (*simulations)->CoherenceLengthSource;

    (*simulations)->OpticalDepthShift = fmod((*simulations)->TargetDepthMin,
                                             (*simulations)->CoherenceLengthSource);

    if ((*simulations)->TypeSimulation == 2 || (*simulations)->TypeSimulation == 3)
        return 1;

    return 1;
}
