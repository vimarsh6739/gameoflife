# Specify Variables

CC= gcc
CXX= g++
NVCC := nvcc

CFLAGS= -std=c++11 -g -O3
LIBS= -lGL -lGLU -lglut

DSPOBJS= GoL.o DisplayEngine.o main.o
DSPTARGET= gui

BENOBJS= GoL.o Benchmark.o
BENTARGET= bench

# Specifying build rules

all : $(DSPTARGET) $(BENTARGET)

display : $(DSPTARGET)

benchmark : $(BENTARGET)

$(BENTARGET) : $(BENOBJS)
	$(NVCC) $^ -o $(BENTARGET) $(CFLAGS)

$(DSPTARGET) : $(DSPOBJS)
	$(NVCC) $^ -o $(DSPTARGET) $(CFLAGS) $(LIBS)

Benchmark.o : Benchmark.cu
	$(NVCC) -c $< $(CFLAGS)

GoL.o : GoL.cu
	$(NVCC) -c $< $(CFLAGS)

DisplayEngine.o : DisplayEngine.cu
	$(NVCC) -c $< $(CFLAGS) $(LIBS)

main.o : main.cu
	$(NVCC) -c $< $(CFLAGS) $(LIBS)

clean : 
	rm -f $(DSPTARGET) $(DSPOBJS) $(BENTARGET) $(BENOBJS)
