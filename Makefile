
# Makefile to build game

CC := gcc
CXX := g++
NVCC := nvcc

CFLAGS := -Wall -g -O3
LIBS := -lGL -lGLU -lglut

OBJS := $(patsubst %.o,%.cpp)
PROG := main
DEPS := 

# Specifying targets


