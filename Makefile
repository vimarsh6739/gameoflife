
# Makefile to build game

CC= gcc
CXX= g++

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

DisplayEngine.o : DisplayEngine.cu
	$(NVCC) -c $< $(CFLAGS) $(LIBS)

main.o : main.cu
	$(NVCC) -c $< $(CFLAGS) $(LIBS)

clean : 
	rm -f $(TARGET) $(OBJS)
