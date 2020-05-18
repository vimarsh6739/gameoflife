
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>
#include <functional>
#include <cstdlib>
#include <ctime>
#include <cuda.h>

#ifndef CPU_GOL_H
#define CPU_GOL_H

__global__ void findNextStateKernel(int* curr_state_dev, int rows, int columns, int* next_state_dev);
__global__ void updateStateColorKernel(int* curr_state_dev, int rows, int columns, int* next_state_dev, float* state_color_dev);

class CPU_gol
{
	private:

	//Defining the dimensions of the grid
	int rows;
	int columns;
	int N;

    //Keeps track of nodes on CPU
	int* curr_state;

	//Keeps track of nodes on GPU
	int* curr_state_dev;

	//Keeps track of next generation on CPU
	int* next_state;

	//Keeps track of next generation on GPU
	int* next_state_dev;

	//Keeps track of current state colors on CPU
	float* state_color;

	//Keeps track of current state colors on GPU
	float* state_color_dev;
	
	//Keeps track of #iterations
	int updateIter;

    //If a particular object's computation is going to happen entirely on cpu or gpu 
	bool isGpu;

	public:
	
	CPU_gol();

	//Construcctor for implicit computation only on the CPU
	CPU_gol(int rows, int columns);
	
	//Constructor to initialize the size of the grid and computation type of the object
	CPU_gol(int rows, int columns, bool isGpu);

	//Function to set a random initial state to both the host and the device pointers
	void randInit();
	
	int getNeighbourCount(int t,int mr,int b,int l,int c,int r);
	void findNextState();
	void updateState();

	bool isAlive(int i, int j);
	void printCells();
	void printColors();
	float* getStateColours();
};

#endif // CPU_GOL_H
