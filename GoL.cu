#include "GoL.h"
#include <random>

//Defining the kernel to update the color bitmap for the texture
__global__ void updateStateColorKernel(int* curr_state_dev, int rows, int columns, int* next_state_dev, float* state_color_dev)
{
	//Finding the unique id of this thread
	int tid = blockIdx.x*blockDim.x+threadIdx.x;

	if(tid < rows * columns)
	{
		curr_state_dev[tid] = next_state_dev[tid];
		
		//Can be changed later for colored update keeping track of previous state
		state_color_dev[3*tid] = (float)next_state_dev[tid];
		state_color_dev[3*tid+1] = (float)next_state_dev[tid];
		state_color_dev[3*tid+2] = (float)next_state_dev[tid];
	}
}

//Defining the kernel for the next state computation
__global__ void findNextStateKernel(int* curr_state_dev, int rows, int columns, int* next_state_dev)
{
	//Finding the unique id of this thread
	int tid=blockIdx.x*blockDim.x+threadIdx.x;

	//performing the operations only for the threads corresponding to a cell in the grid
	if(tid < rows*columns)
	{
		//Get the cyclic count of neighbours around the current cell
		int nbrCnt = 0;
		int top,bottom,mid_r,left,mid_c,right;
		
		//Current row and column
		int i = tid / columns;
		int j = tid % columns;
		
		//Defining variables for the grid surrounding (i,j)
		top=((i-1+rows)%rows)*columns;
		bottom=((i+1)%rows)*columns;
		mid_r = i*columns;
		mid_c =j;
		left = (j-1+columns)%columns;
		right = (j+1)%columns;

		//Counting the number of neighbours around (i,j)
		nbrCnt =  curr_state_dev[top + left] 
		   		+ curr_state_dev[top + mid_c] 
		   		+ curr_state_dev[top + right]
		   		+ curr_state_dev[mid_r + left]  
		   		+ curr_state_dev[mid_r + right]
		   		+ curr_state_dev[bottom + left] 
		   		+ curr_state_dev[bottom + mid_c] 
		   		+ curr_state_dev[bottom + right];
		
		if(nbrCnt == 3 || ((nbrCnt==2)&&(curr_state_dev[tid]==1)))
		{
			next_state_dev[tid] = 1;
		}
		else
		{
			next_state_dev[tid] = 0;
		}
	}
}

GoL::GoL()
{
	//Empty default constructor
	// :)
}

//Allot memory for state bitmap - assume only CPU computation
GoL::GoL(int rows, int columns)
{
	srand(time(NULL));

	this->rows = rows;
	this->columns = columns;
	N = rows*columns;

	this->curr_state = (int*)calloc(N,sizeof(int));
	this->next_state = (int*)calloc(N,sizeof(int));
	this->state_color = (float*)calloc(3*N,sizeof(float));

	this->updateIter = 0;

	this->isGpu = false;
}

//Allot memory for state bitmap - assume CPU <-> GPU computation
GoL::GoL(int rows, int columns, bool isGpu)
{
	srand(time(NULL));

	this->rows = rows;
	this->columns = columns;
	N = rows*columns;

	this->curr_state = (int*)calloc(N,sizeof(int));
	this->next_state = (int*)calloc(N,sizeof(int));
	this->state_color = (float*)calloc(3*N,sizeof(float));

	this->updateIter = 0;

	this->isGpu = isGpu;
	this->config_file = "random.txt";

	if(isGpu)
	{
		//Allocate memory in GPU for all cuda arrays
		cudaMalloc(&curr_state_dev, N*sizeof(int));
		cudaMalloc(&next_state_dev, N*sizeof(int));
		cudaMalloc(&state_color_dev, 3*N*sizeof(float));

		//Set all arrays to 0
		cudaMemset(curr_state_dev, 0, N*sizeof(int));
		cudaMemset(next_state_dev, 0, N*sizeof(int));
		cudaMemset(state_color_dev, 0, 3*N*sizeof(float));
	}
}

//Store config file name and boolean for gpu computation
GoL::GoL(bool isGpu, std::string fname)
{
	this->isGpu = isGpu;
	this->config_file = fname;
	this->updateIter = 0;
}

//Initialize the state matrix to a random initial state 
void GoL::setRandomInitialState()
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

	//Update the next state for the current state in CPU or corresponding GPU
	if(isGpu)
	{
		//Copy current state and colors
		cudaMemcpy(curr_state_dev, curr_state, N*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(state_color_dev, state_color, 3*N*sizeof(float), cudaMemcpyHostToDevice);

		int nblocks = ceil((N)/1024.0);;

		//Call a kernel to compute the next state value
		findNextStateKernel<<<nblocks,1024>>>(curr_state_dev,rows,columns,next_state_dev);

		//Store a copy of the next state in CPU
		cudaMemcpy(next_state, next_state_dev, N*sizeof(int), cudaMemcpyDeviceToHost);
	}
	else
	{
		findNextState();
	}
}

//Get initial state details from a file
bool GoL::getInitialState(std::string filename)
{
	std::ifstream file(filename);

	if(!(file.is_open()))
	{
		return false;
	}

	int _m,_n;
	bool isCpuOrGpu;
	int* arr;
	
	file>>_m>>_n;
	file>>isCpuOrGpu;
	
	arr=(int*)malloc(m*n*sizeof(int));
	
	for(int i=0;i<_m;++i)
	{
		for(int j=0;j<_n;++j)
		{
			file>>arr[i*_n+j];
		}
	}

	setInitialState(_m,_n,isCpuOrGpu,arr);
	file.close();

	return true;
}

//Set member variables to values in file input
void GoL::setInitialState(int _m, int _n, bool isCpuOrGpu, int* arr)
{
	//Initialize all member vars
	this->N = _m*_n;
	this->rows = _m;
	this->columns = _n;

	this->isGpu = isCpuOrGpu;
	this->curr_state = arr;

	this->next_state = (int*)calloc(N,sizeof(int));
	this->state_color = (float*)calloc(3*N,sizeof(float));
	
	//Initialize color for all states
	for(int i=0;i<rows;++i)
	{
		for(int j=0;j<columns;++j)
		{
			int currPos = i*columns + j;
			//Set cell position to white
			if(arr[currPos]==1)
			{
				state_color[3*currPos] = 1.0;
				state_color[3*currPos+1] = 1.0;
				state_color[3*currPos+2] = 1.0;
			}
		}
	}

	// findNextState();

	if(isGpu)
	{

		//Allocate memory in GPU for all cuda arrays
		cudaMalloc(&curr_state_dev, N*sizeof(int));
		cudaMalloc(&next_state_dev, N*sizeof(int));
		cudaMalloc(&state_color_dev, 3*N*sizeof(float));

		//Copy host arrays to device
		cudaMemcpy(curr_state_dev, curr_state, N*sizeof(int), cudaMemcpyHostToDevice);
		// cudaMemcpy(next_state_dev, next_state, N*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(state_color_dev, state_color, 3*N*sizeof(int), cudaMemcpyHostToDevice);
		
	}

	//Update the next state exclusively in either CPU or GPU - save some preprocessing time
	if(isGpu)
	{
		int nblocks = ceil((N)/1024.0);;

		//Call a kernel to compute the next state value
		findNextStateKernel<<<nblocks,1024>>>(curr_state_dev,rows,columns,next_state_dev);
	}
	else
	{
		findNextState();
	}
}

//Get the number of neighbors around the current grid -> Only 0/1 
//values present for next cells
int GoL::getNeighbourCount(int top, int mid_r, int bottom, int left, int mid_c, int right)
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

//Compute the next state information in the CPU
void GoL::findNextState()
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

//Black box function to update game state(called by DisplayEngine)
void GoL::updateState()
{
	updateIter++;
	
	//Update next_state
	if(isGpu)
	{
		int nblocks = ceil((N)/1024.0);;
		//Update colurs for next state in GPU
		updateStateColorKernel<<<nblocks,1024>>>(curr_state_dev,rows,columns,next_state_dev,state_color_dev);
		
		//Call a kernel to compute the next state value
		findNextStateKernel<<<nblocks,1024>>>(curr_state_dev,rows,columns,next_state_dev);

	}
	else
	{	
		//Update colours for current state in CPU
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

		//Find the next state value
		findNextState();
	}
}

//Function which returns the current state colors
//Used to update the texture in DisplayEngine
float* GoL::getStateColours()
{
	//To be replaced later while implementing interop
	if(isGpu)
	{
		cudaMemcpy(state_color, state_color_dev, 3*N*sizeof(float), cudaMemcpyDeviceToHost);
	}
	
	return state_color;
}

// A set of helper and checker functions

bool GoL::isAlive(int i, int j) 
{
	int currPos = i*columns + j;
	return curr_state[currPos]==1 ;
}

void GoL::printCells()
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

void GoL::printColors()
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

//Returns the current generation number for benchmarking
int GoL::getIterationNumber()
{
	return this->updateIter;
}

void GoL::toggleComputation()
{
	//Check if computation is in GPU
	if(isGpu)
	{
		isGpu = false;
		
		//Copy all required vars to CPU memory
		cudaMemcpy(curr_state, curr_state_dev, N*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(next_state, next_state_dev, N*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(state_color, state_color_dev, 3*N*sizeof(float), cudaMemcpyDeviceToHost);
		
		//Free gpu memory
		cudaFree(curr_state_dev);
		cudaFree(next_state_dev);
		cudaFree(state_color_dev);
	}
	else
	{
		isGpu = true;

		//Allocate memory in GPU for all cuda arrays
		cudaMalloc(&curr_state_dev, N*sizeof(int));
		cudaMalloc(&next_state_dev, N*sizeof(int));
		cudaMalloc(&state_color_dev, 3*N*sizeof(float));

		//Copy arrays from host to device
		cudaMemcpy(curr_state_dev, curr_state, N*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(next_state_dev, next_state, N*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(state_color_dev, state_color, 3*N*sizeof(float), cudaMemcpyHostToDevice);

	}
}