#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>
#include <functional>
#include <cstdlib>
#include <ctime>
#include <cuda.h>

#ifndef GOL_H
#define GOL_H

__global__ void findNextStateKernel(int* curr_state_dev, int rows, int columns, int* next_state_dev);
__global__ void updateStateColorKernel(int* curr_state_dev, int rows, int columns, int* next_state_dev, float* state_color_dev);

class GoL
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
	
	GoL();

	//Constructor for implicit computation only on the CPU
	GoL(int rows, int columns);
	
	//Constructor to initialize the size of the grid and computation type of the object
	GoL(int rows, int columns, bool isGpu);

	//Functions for initializing the conway grid---
	void randomInitialState();
	bool getInitialState(std::string filename);
	void setInitialState(int _m, int _n, bool isCpuOrGpu, int* arr);

	//Functions for next state computation in the CPU
	int getNeighbourCount(int t,int mr,int b,int l,int c,int r);
	void findNextState();

	//Function called by display object to update state
	//Automatically decides if the computation is handled by CPU or GPU 
	//depending on the set value in isGpu
	void updateState();

	//Function called by display object during formation/update of texture
	//object
	float* getStateColours();
	
	//Function to switch computation from CPU <-> GPU
	void toggleComputation();

	//Functions for displaying/returning metrics
	bool isAlive(int i, int j);
	void printCells();
	void printColors();
	int getIterationNumber();
	
};

#endif // GOL_H
