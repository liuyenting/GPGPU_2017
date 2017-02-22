# Target homework folder
HOMEWORK	= lab0

# Gencode arguments
SMS 		?= 30

# Common includes and paths for CUDA
INCLUDES	= utils
LIBRARIES	=

################################################################################

# Location of the CUDA Toolkit
CUDA_PATH	?= /Developer/NVIDIA/CUDA-8.0

# C++ compiler
CXX		= clang++
# NVIDIA compiler
NVCC	:= $(CUDA_PATH)/bin/nvcc -ccbin $(CXX)

# Detact system architecture
HOST_ARCH   := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)
ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 ppc64le))
    TARGET_SIZE := 64
else ifeq ($(TARGET_ARCH),armv7l)
    TARGET_SIZE := 32
else
    $(error ERROR - unsupported value $(TARGET_ARCH) for TARGET_ARCH!)
endif

################################################################################

# Default flags
NVCCFLAGS   := -m $(TARGET_SIZE) -std c++11
CCFLAGS     := -arch $(HOST_ARCH)
LDFLAGS     := -rpath $(CUDA_PATH)/lib

# Generate complete flag sets
ALL_CCFLAGS := $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

ALL_LDFLAGS := $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

################################################################################

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
endif

################################################################################

# Source files in the designated homework directory
CPP_SRCS	:= $(wildcard $(HOMEWORK)/*.cpp)
C_SRCS		:= $(wildcard $(HOMEWORK)/*.c)
CU_SRCS		:= $(wildcard $(HOMEWORK)/*.cu)

# Target object files
OBJS	:= $(filter %.o,$(CPP_SRCS:.cpp=.o) $(C_SRCS:.c=.o) $(CU_SRCS:.cu=.o))

# Target binary filename
EXE			= gpgpu_$(HOMEWORK)

################################################################################

all: build

build: $(EXE)

%.o: %.c
%.o: %.cpp
	@echo Compiling "$<"...
	@$(CXX) -I $(INCLUDES) $(ALL_CCFLAGS) -o $@ -c $<

%.o: %.cu
	@echo Compiling "$<"...
	@$(NVCC) -I $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(EXE): $(OBJS)
	@$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

run: build
	@./$(EXE)

clean:
	rm -f $(EXE) $(HOMEWORK)/*.o
