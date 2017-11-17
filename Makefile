RM := rm -rf

CUDAPATH := /usr/local/cuda
FLAGS := --use_fast_math
BUILDDIR := $(shell mkdir -p build; echo build)

CPP_SRCS += \
src/bias.cpp \
src/mesh.cpp \
src/octmps_io.cpp

CU_SRCS += \
src/main.cu

CU_DEPS += \
$(BUILDDIR)/main.d

OBJS += \
$(BUILDDIR)/bias.o \
$(BUILDDIR)/main.o \
$(BUILDDIR)/mesh.o \
$(BUILDDIR)/octmps_io.o

CPP_DEPS += \
$(BUILDDIR)/bias.d \
$(BUILDDIR)/mesh.d \
$(BUILDDIR)/octmps_io.d

$(BUILDDIR)/%.o: src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc $(FLAGS) -I$(CUDAPATH)/samples/common/inc/ -G -g -O0 -gencode arch=compute_35,code=sm_35  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc $(FLAGS) -I$(CUDAPATH)/samples/common/inc/ -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

$(BUILDDIR)/%.o: src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc $(FLAGS) -I$(CUDAPATH)/samples/common/inc/ -G -g -O0 -gencode arch=compute_35,code=sm_35  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc $(FLAGS) -I$(CUDAPATH)/samples/common/inc/ -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

all: OCT-MPS

OCT-MPS: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: NVCC Linker'
	nvcc $(FLAGS) --cudart static --relocatable-device-code=false -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35 -link -o  "OCT-MPS" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(CU_DEPS) $(OBJS) $(C++_DEPS) $(C_DEPS) $(CC_DEPS) $(CPP_DEPS) $(EXECUTABLES) $(CXX_DEPS) $(C_UPPER_DEPS) OCT-MPS
	-$(RM) $(BUILDDIR)
	-@echo ' '

.PHONY: all clean dependents
.SECONDARY:

