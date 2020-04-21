
# Makefile to build game

CC := gcc
CXX := g++
NVCC := nvcc

CFLAGS= -g -O3
LIBS= -lGL -lGLU -lglut

OBJS= GoL_backend.o DisplayEngine.o main.o

TARGET= bin


# Specifying build rules

all : $(TARGET)

$(TARGET) : $(OBJS)
	$(NVCC) $^ -o $(TARGET) $(CFLAGS) $(LIBS)

GoL_backend.o : GoL_backend.cu
	$(NVCC) -c $< $(CFLAGS) $(LIBS)

DisplayEngine.o : DisplayEngine.cpp
	$(NVCC) -c $< $(CFLAGS) $(LIBS)

main.o : main.cpp
	$(CXX) -c $< $(CFLAGS) $(LIBS)

clean : 
	rm -f $(TARGET) $(OBJS)

