
# Makefile to build game

CC= gcc
CXX= g++

CFLAGS= -Wall -g -O3
LIBS= -lGL -lGLU -lglut

OBJS= CPU_gol.o DisplayEngine.o main.o

TARGET= bin


# Specifying build rules

all : $(TARGET)

$(TARGET) : $(OBJS)
	$(CXX) $^ -o $(TARGET) $(CFLAGS) $(LIBS)

CPU_gol.o : CPU_gol.cpp
	$(CXX) -c $< $(CFLAGS) $(LIBS)

DisplayEngine.o : DisplayEngine.cpp
	$(CXX) -c $< $(CFLAGS) $(LIBS)

main.o : main.cpp
	$(CXX) -c $< $(CFLAGS) $(LIBS)

clean : 
	rm -f $(TARGET) $(OBJS)



