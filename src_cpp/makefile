CXX = g++-12
CXXFLAGS = -std=c++11 -O3 -march=native -fopenmp
LDLIBS = -lopenblas
TARGETS = $(basename $(wildcard *.cpp))

all : $(TARGETS)

%:%.cpp *.h
	$(CXX) $(CXXFLAGS) $< $(LDLIBS) -o $@

clean:
	-$(RM) $(TARGETS) *~

.PHONY: all, clean
