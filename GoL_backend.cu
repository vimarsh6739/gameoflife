#include "GoL_backend.h"

__global__ void k1(int* curr_state_device,int m,int n,int* temporary_arr_for_device)
{
	//finding the unique id of this thread
	int tid=blockIdx.x*blockDim.x+threadIdx.x;
	
	//performing the operations only for the threads corresponding to a cell in the grid
	if(tid<m*n)
	{
		temporary_arr_for_device[tid]=curr_state_device[tid];
		int neighbour_val=-curr_state_device[tid];
		int i=tid/n;
		int j=tid%n;
		for(int i1=-1;i1<=1;++i1)
		{
			for(int j1=-1;j1<=1;++j1)
			{
				int x=(i+i1+m)%m;
				int y=(j+j1+n)%n;
				neighbour_val+=curr_state_device[x*n+y];
			}
		}
		if(curr_state_device[tid])
		{
			if(neighbour_val<2||neighbour_val>3)
			{
				temporary_arr_for_device[tid]=0;
			}
		}
		else
		{
			if(neighbour_val==3)
			{
				temporary_arr_for_device[tid]=1;
			}
		}
	}
}

__global__ void k2(int* curr_state_device,int m,int n,int* temporary_arr_for_device)
{
	int tid=blockIdx.x*blockDim.x+threadIdx.x;
	if(tid<m*n)
	{
		//copy the contents of the temporary device pointer to the device state pointer
		curr_state_device[tid]=temporary_arr_for_device[tid];
	}
}

void GoL::copyHostToDevice()
{
	cudaMemcpy(curr_state_device,curr_state_host,m*n*sizeof(int),cudaMemcpyHostToDevice);
}

void GoL::copyDeviceToHost()
{
	cudaMemcpy(curr_state_host,curr_state_device,m*n*sizeof(int),cudaMemcpyDeviceToHost);
}
	

GoL::GoL(int _m,int _n,bool ifCpuOrGpu)
{
	srand(time(NULL));
	m=_m;
	n=_n;
	cudaMalloc(&curr_state_device,m*n*sizeof(int));
	cudaMalloc(&temporary_arr_for_device,m*n*sizeof(int));
	curr_state_host=(int*)malloc(m*n*sizeof(int));
	cpuorgpu=ifCpuOrGpu;
	iteration_number=0;
}

void GoL::randInitialState()
{
	for(int i=0;i<m;++i)
		for(int j=0;j<n;++j)
			curr_state_host[i*n+j]=(rand()%2);
	copyHostToDevice();
}

void GoL::setInitialState(int _m,int _n,bool isCpuOrGpu,int* arr)
{
	//copy the input parameter pointer to the host pointer
	m=_m;
	n=_n;
	cpuorgpu=ifCpuOrGpu;
	for(int i=0;i<m;++i)
	{
		for(int j=0;j<n;++j)
		{
			curr_state_host[i*n+j]=arr[i*n+j];
		}
	}
	//copy the host pointer to the device pointer
	copyHostToDevice();
}

bool GoL::getInitialState(string filename)
{
	std::ifstream file(filename);
	if(!(file.is_open()))
		return false;
	int _m,_n;
	bool isCpuOrGpu;
	int* arr;
	file>>_m>>_n;
	file>>isCpuOrGpu;
	arr=(int*)malloc(m*n*sizeof(int));
	for(int i=0;i<_m;++i)
		for(int j=0;j<_n;++j)
			file>>arr[i*_n+j];
	setInitialState(_m,_n,isGpuOrCpu,arr);
	file.close();
	free(arr);
	return true;
}

void GoL::change_of_state_gpu()
{
	//terminate if the object is not allowed to call the gpu function
	if(!cpuorgpu)
	{
		return;
	}
	int nblocks=ceil((m*n)/1024.0);
	
	//kernel k1 computes the next state values and stores it in the temporary array pointer available on the device, which is done in order to avoid data race
	k1<<<nblocks,1024>>>(curr_state_device,m,n,temporary_arr_for_device);
	
	//kernel k2 copies the temporary values into the device state variable, which is separated from k1 in order to implement a global device barrier
	k2<<<nblocks,1024>>>(curr_state_device,m,n,temporary_arr_for_device);
}

void GoL::change_of_state_cpu()
{
	//terminates if the object is not allowed to call the cpu function
	if(cpuorgpu)
	{
		return;
	}
	//temporary array to store the next state values in order to not create data race
	std::vector<std::vector<int>> temp(m,std::vector<int>(n));
	for(int i=0;i<m;++i)
	{
		for(int j=0;j<n;++j)
		{
			int curr_pos=i*n+j;
			temp[i][j]=curr_state_host[curr_pos];
			int neighbour_val=-curr_state_host[curr_pos];
			for(int i1=-1;i1<=1;++i1)
			{
				for(int j1=-1;j1<=1;++j1)
				{
					//the entire grid is warped around
					int x=(i+i1+m)%m;
					int y=(j+j1+n)%n;
					
					//computing the number of live neighbours
					neighbour_val+=curr_state_host[x*n+y];
				}
			}
			//enters if your current node is currently alive
			if(curr_state_host[curr_pos])
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
	for(int i=0;i<m;++i)
	{
		for(int j=0;j<n;++j)
		{
			curr_state_host[i*n+j]=temp[i][j];
		}
	}
}

void GoL::change_of_state()
{
	iteration_number++;
	if(iteration_number==1)
		return;
	if(cpuorgpu)
		change_of_state_gpu();
	else
		change_of_state_cpu();
}

bool GoL::isAlive(int i,int j)
{
	if(cpuorgpu)
	{
		cudaMemcpy((curr_state_host+(i*n+j)),(curr_state_device+(i*n+j)),sizeof(int),cudaMemcpyDeviceToHost);	
	}
	return (curr_state_host[i*n+j]==1);
}

void GoL::printCells()
{
	if(cpuorgpu)
	{
		copyDeviceToHost();
	}
	for(int i=0;i<m;++i)
	{
		for(int j=0;j<n;++j)
			printf("%d ",curr_state_host[i*n+j]);
		printf("\n");
	}		
}

int* GoL::getStateGPU()
{
	return curr_state_device;
}

int* GoL::getStateCPU()
{
	return curr_state_host;
}

bool GoL::getIfCpuOrGpu()
{
	return cpuorgpu;
}

void GoL::switchComputation(bool switchTo)
{
	if(switchTo==cpuorgpu)
		return;
	cpuorgpu=switchTo;
	if(cpuorgpu)
	{
		copyHostToDevice();
	}
	else
	{
		copyDeviceToHost();
	}
}
