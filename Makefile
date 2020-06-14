
# Makefile to build game

CC= gcc
CXX= g++
NVCC := nvcc

CFLAGS= -std=c++11 -g -O3
LIBS= -lGL -lGLU -lglut

OBJS= GoL.o DisplayEngine.o main.o

TARGET= bin

# Specifying build rules

all : $(TARGET)

$(TARGET) : $(OBJS)
	$(NVCC) $^ -o $(TARGET) $(CFLAGS) $(LIBS)

GoL.o : GoL.cu
	$(NVCC) -c $< $(CFLAGS) $(LIBS)

DisplayEngine.o : DisplayEngine.cu
	$(NVCC) -c $< $(CFLAGS) $(LIBS)

main.o : main.cu
	$(NVCC) -c $< $(CFLAGS) $(LIBS)

clean : 
	rm -f $(TARGET) $(OBJS)
