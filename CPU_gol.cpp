#include "CPU_gol.h"
#include <random>

CPU_gol::CPU_gol(){

}

//Allot memory for state bitmap
CPU_gol::CPU_gol(int rows, int columns) {
	
	//Seed the random number generator
	srand(time(NULL));
	//Set rows, columns and bitmap
	this->rows = rows;
	this->columns = columns;
	this->state = (int*)calloc(rows*columns,sizeof(int));
	//Set global iteration counter
	this->iter = 0;

}

//Initialize the grid with a random value to start
void CPU_gol::randInit(){
	
	//Randomly initialize the state values to 0 or 1
	for(int i = 0; i < rows; ++i){
		for(int j = 0; j <columns ; ++j){
			state[i*columns + j] = (rand() % 2);
		}
	}
}

//Perform 1 iteration of update
void CPU_gol::update_state() {

	iter++;
	//Do not update state for the first iteration
	if(iter == 1){return;}

	//Temporary array to store the next state values
	std::vector<std::vector<int>> temp(rows,std::vector<int>(columns,0));

	for(int i=0;i<rows;++i)
	{
		for(int j=0;j<columns;++j)
		{
			int curr_pos=i*columns+j;
			
			temp[i][j]=state[curr_pos];
			
			int neighbour_val=-state[curr_pos];
			
			for(int i1=-1;i1<=1;++i1)
			{
				for(int j1=-1;j1<=1;++j1)
				{
					//the entire grid is warped around
					int x=(i+i1+rows)%rows;
					int y=(j+j1+columns)%columns;
					
					//computing the number of live neighbours
					neighbour_val+=state[x*columns+y];
				}
			}

			//enters if your current node is currently alive
			if(state[curr_pos])
			{
				//current node dies either by under population or over population respectively
				if(neighbour_val<2||neighbour_val>3)
				{
					temp[i][j]=0;
				}
			}
			//enters if your current node is currently dead
			else
			{
				//current dead node comes to life due to reproduction of neighbours
				if(neighbour_val==3)
				{
					temp[i][j]=1;
				}
			}
		}
	}

	//copy the temporary vector into the host state pointer after computation of all states
	for(int i=0;i<rows;++i)
	{
		for(int j=0;j<columns;++j)
		{
			state[i*columns+j]=temp[i][j];
		}
	}
}

bool CPU_gol::isAlive(int i, int j) {
	return state[i*columns + j] == 1;
}

void CPU_gol::printCells(){
	for(int i=0;i < rows; ++i){
		for(int j = 0; j < columns ;++j){
			std::cout<<state[i*columns + j]<<" ";
		}
		std::cout<<"\n";
	}
}