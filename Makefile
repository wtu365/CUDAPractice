# CUDA
CUDA_CFLAGS = -std=c++11 -O3
CUDA_LDFLAGS =

RM = rm -f

TARGETS = transpose

all: $(TARGETS)

transpose: transpose.cu
	nvcc $(CUDA_CFLAGS) $(CUDA_LDFLAGS) -o $(@) transpose.cu

clean:
	$(RM) $(TARGETS) *.lib *.a *.exe *.obj *.o *.exp *.pyc