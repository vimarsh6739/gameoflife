#include "CPU_gol.h"
#include <random>

CPU_gol::CPU_gol()
{

}

//Allot memory for state bitmap
CPU_gol::CPU_gol(int rows, int columns)
{
	srand(time(NULL));

	this->rows = rows;
	this->columns = columns;
	N = rows*columns;

	this->curr_state = (int*)calloc(N,sizeof(int));
	this->next_state = (int*)calloc(N,sizeof(int));
	this->state_color = (float*)calloc(3*N,sizeof(float));

	this->updateIter = 0;

}

void CPU_gol::randInit()
{
	int doa = 0;
	int currPos = 0;
	for(int i=0; i<rows; ++i)
	{
		for(int j=0;j<columns;++j)
		{
			currPos = i*columns + j;
			doa = rand() % 2;
			curr_state[currPos] = doa;
			state_color[3*currPos] = (float)doa;
			state_color[3*currPos+1] = (float)doa;
			state_color[3*currPos+2] = (float)doa;
		}
	}
	findNextState();
}

int CPU_gol::getNeighbourCount(int top, int mid_r, int bottom, int left, int mid_c, int right)
{
	return   curr_state[top + left] 
		   + curr_state[top + mid_c] 
		   + curr_state[top + right]
		   + curr_state[mid_r + left]  
		   + curr_state[mid_r + right]
		   + curr_state[bottom + left] 
		   + curr_state[bottom + mid_c] 
		   + curr_state[bottom + right];
}

void CPU_gol::findNextState()
{
	int currPos =0;
	int nbrCnt = 0;
	int top,bottom,mid_r,left,mid_c,right;
	for(int i=0;i<rows;++i)
	{
		top=((i-1+rows)%rows)*columns;
		bottom=((i+1)%rows)*columns;
		mid_r = i*columns;

		for(int j=0;j<columns;++j)
		{
			currPos = mid_r + j;
			mid_c =j;
			left = (j-1+columns)%columns;
			right = (j+1)%columns;
			nbrCnt = getNeighbourCount(top,mid_r,bottom,left,mid_c,right);
			if(nbrCnt == 3 || ((nbrCnt==2)&&(curr_state[currPos]==1)))
			{
				next_state[currPos] = 1;
			}
			else
			{
				next_state[currPos] = 0;
			}
		}
	}
}

void CPU_gol::updateState()
{
	updateIter++;
	int currPos =0;
	for(int i=0;i<rows;++i)
	{
		for(int j=0;j<columns;++j)
		{
			currPos= i*columns + j;
			curr_state[currPos] = next_state[currPos];

			//Can be changed later for colored update keeping track of previous state
			state_color[3*currPos] = (float)next_state[currPos];
			state_color[3*currPos+1] = (float)next_state[currPos];
			state_color[3*currPos+2] = (float)next_state[currPos];
		}
	}

	//Update next_state
	findNextState();
}

bool CPU_gol::isAlive(int i, int j) 
{
	int currPos = i*columns + j;
	return curr_state[currPos]==1 ;
}

float* CPU_gol::getStateColours()
{
	return state_color;
}

void CPU_gol::printCells()
{
	int currPos = 0;
	for(int i=0;i < rows; ++i)
	{
		for(int j = 0; j < columns ;++j)
		{
			currPos = i*columns + j;
			std::cout<<curr_state[currPos]<<" ";
		}
		std::cout<<"\n";
	}
}

void CPU_gol::printColors()
{
	int currPos=0;
	for(int i=0;i<rows;++i)
	{
		for(int j=0;j<columns;++j)
		{
			currPos = i*columns + j;
			std::cout<<"("<<state_color[3*currPos]<<","<<state_color[3*currPos+1]<<","<<state_color[3*currPos+2]<<") ";
		}
		std::cout<<"\n";
	}
}