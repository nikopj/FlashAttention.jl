CXX = icpc
CXXFLAGS = -std=c++11 -O3 -march=native -fopenmp
CUDA_PATH = /usr/local/cuda
NVCC = nvcc
NVCCFLAGS = -std=c++11 -O3

LDLIBS = -mkl

CPP_SOURCES = $(wildcard *.cpp)
CUDA_SOURCES = $(wildcard *.cu)

CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)

TARGETS = $(basename $(CPP_SOURCES) $(CUDA_SOURCES))

all: $(TARGETS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%: %.o
	$(CXX) $(CXXFLAGS) $< $(LDLIBS) -o $@

clean:
	-$(RM) $(TARGETS) $(CPP_OBJECTS) $(CUDA_OBJECTS) *~

.PHONY: all clean
