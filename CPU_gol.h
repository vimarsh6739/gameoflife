
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
	int* state;
	//Keeps track of iteration number
	int iter;
public:
	
	CPU_gol();

	CPU_gol(int rows, int columns);

	~CPU_gol();

	void randInit();

	void update_state();

	bool isAlive(int i, int j);

	void printCells();
};

#endif // CPU_GOL_H
