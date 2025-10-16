# V2 - Naive CUDA KLT build

NVCC      ?= nvcc
CXX       ?= g++
CC        = gcc

INCLUDES  := -I. -I../V1
CXXFLAGS  := -O2 -std=c++17 $(INCLUDES)
CFLAGS    := -O2 $(INCLUDES)
NVCCFLAGS := -O2 -std=c++17 $(INCLUDES)

# Note: kernels.cu is included by main.cu; do not compile separately
SRCS_CPP  := utils.cpp
SRCS_CU   := main.cu
SRCS_C    := ../V1/pnmio.c ../V1/error.c

OBJS_CPP  := $(SRCS_CPP:.cpp=.o)
OBJS_CU   := $(SRCS_CU:.cu=.o)
OBJS_C    := $(SRCS_C:.c=.o)

TARGET    := v2_klt

all: $(TARGET)

$(TARGET): $(OBJS_CU) $(OBJS_CPP) $(OBJS_C)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.c
	$(NVCC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET) ../../data/img0.pgm ../../data/img1.pgm

# Also allow running the CPU (V1) from here
cpu:
	$(MAKE) -C ../V1

run_cpu: cpu
	$(MAKE) -C ../V1 run

clean:
	rm -f $(OBJS_CPP) $(OBJS_CU) $(OBJS_C) $(TARGET)

.PHONY: all clean run
