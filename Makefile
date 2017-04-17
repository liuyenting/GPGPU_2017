# Target homework folder
SRC_DIR		= lab1

# Target binary filename
#EXE		= gpgpu

# Gencode arguments
SMS 		?= 30

# Common includes and libraries for CUDA
INC_PATH	= /opt/local/include utils
LIBRARIES	= m

LIB_PATH	= /opt/local/lib

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

# Prepends parameters
INC_PARMS	:= $(foreach d, $(INC_PATH), -I$d)
LIB_PARMS	:= $(foreach d, $(LIB_PATH), -L$d) $(foreach d, $(LIBRARIES), -l$d)

# Default flags
NVCCFLAGS   := -m $(TARGET_SIZE) -std c++11
CCFLAGS     := $(INC_PARMS) -arch $(HOST_ARCH)
LDFLAGS     := $(LIB_PARMS) -rpath $(CUDA_PATH)/lib

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

# List all the folders in the source directory
FILE_LIST	:= $(shell find $(SRC_DIR) -type d)

# Source files in the designated homework directory
CPP_SRCS	:= $(wildcard $(addsuffix /*.cpp,$(FILE_LIST)))
C_SRCS		:= $(wildcard $(addsuffix /*.c,$(FILE_LIST)))
CU_SRCS		:= $(wildcard $(addsuffix /*.cu,$(FILE_LIST)))

# Target object files
OBJS	:= $(filter %.o,$(CPP_SRCS:.cpp=.o) $(C_SRCS:.c=.o) $(CU_SRCS:.cu=.o))

# Use source directory name as the default
EXE	?= gpgpu_$(SRC_DIR)

################################################################################

all: build

build: $(EXE)

$(EXE): $(OBJS)
	@$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+

%.o: %.c
%.o: %.cpp
	@echo Compiling "$<"...
	@$(CXX) $(INC_PARMS) $(ALL_CCFLAGS) -o $@ -c $<

%.o: %.cu
	@echo Compiling "$<"...
	@$(NVCC) $(INC_PARMS) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

print-%:
	@echo '$*=$($*)'

run: build
	@./$(EXE)

clean:
	rm -f $(EXE) $(addsuffix /*.o,$(FILE_LIST))
