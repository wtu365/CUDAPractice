# CUDA
CUDA_CFLAGS = -std=c++11 -O3
CUDA_LDFLAGS =

RM = rm -f

TARGETS = test

all: $(TARGETS)

test: test.cu
	nvcc $(CUDA_CFLAGS) $(CUDA_LDFLAGS) -o $(@) test.cu

clean:
	$(RM) $(TARGETS) *.lib *.a *.exe *.obj *.o *.exp *.pyc