
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>
#include <functional>
#include <cstdlib>
#include <ctime>

#ifndef CPU_GOL_H
#define CPU_GOL_H

class CPU_gol
{
	private:
	int rows;
	int columns;
	int N;

	int* curr_state;
	int* next_state;

	float* state_color;
	int updateIter;

	public:
	
	CPU_gol();
	CPU_gol(int rows, int columns);
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
