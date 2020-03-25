#include "GoL_backend.h"

__global__ void k1(int* curr_state_device,int m,int n,int* temporary_arr_for_device)
{
	int tid=blockIdx.x*blockDim.x+threadIdx.x;
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
		curr_state_device[tid]=temporary_arr_for_device[tid];
	}
}

class GoL
{
	private:
		int* curr_state_device;
		int* curr_state_host;
		int m=0,n=0;
		bool cpuorgpu=true;
		int* temporary_arr_for_device;
	
		void copyHostToDevice()
		{
			cudaMemcpy(curr_state_device,curr_state_host,m*n*sizeof(int),cudaMemcpyHostToDevice);
		}

		void copyDeviceToHost()
		{
			cudaMemcpy(curr_state_host,curr_state_device,m*n*sizeof(int),cudaMemcpyDeviceToHost);
		}
	
	public:	

		GoL(int _m,int _n,bool ifCpuOrGpu)
		{
			m=_m;
			n=_n;
			cudaMalloc(&curr_state_device,m*n*sizeof(int));
			cudaMalloc(&temporary_arr_for_device,m*n*sizeof(int));
			curr_state_host=(int*)malloc(m*n*sizeof(int));
			cpuorgpu=ifCpuOrGpu;
		}
	
		void setInitialState(int* arr);
		void change_of_state_cpu();
		void change_of_state_gpu();
		
};

void GoL::setInitialState(int* arr)
{
	for(int i=0;i<m;++i)
	{
		for(int j=0;j<n;++j)
		{
			curr_state_host[i*n+j]=arr[i*n+j];
		}
	}
	copyHostToDevice();
}

void GoL::change_of_state_gpu()
{
	if(!cpuorgpu)
	{
		return;
	}
	int nblocks=ceil((m*n)/1024.0);
	k1<<<nblocks,1024>>>(curr_state_device,m,n,temporary_arr_for_device);
	k2<<<nblocks,1024>>>(curr_state_device,m,n,temporary_arr_for_device);
}

void GoL::change_of_state_cpu()
{
	if(cpuorgpu)
	{
		return;
	}
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
					int x=(i+i1+m)%m;
					int y=(j+j1+n)%n;
					neighbour_val+=curr_state_host[x*n+y];
				}
			}
			if(curr_state_host[curr_pos])
			{
				if(neighbour_val<2||neighbour_val>3)
				{
					temp[i][j]=0;
				}
			}
			else
			{
				if(neighbour_val==3)
				{
					temp[i][j]=1;
				}
			}
		}
	}
	for(int i=0;i<m;++i)
	{
		for(int j=0;j<n;++j)
		{
			curr_state_host[i*n+j]=temp[i][j];
		}
	}
}
