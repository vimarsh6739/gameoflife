#include <bits/stdc++.h>
#include <cuda.h>

//cuda kernels needed as a part of computing the next state of the grid from the gpu
__global__ void k1(int* curr_state_device,int m,int n,int* temporary_arr_for_device);
__global__ void k2(int* curr_state_device,int m,int n,int* temporary_arr_for_device);

//class that contains all the necessary software implementations of game of life
class GoL
{
	private:
    //device array pointer that stores all the nodes on gpu
		int* curr_state_device;
    
    //host array pointer that stores all the nodes on cpu
		int* curr_state_host;
    
    //row and column dimensions of the grid respectively
		int m,n;
    
    //boolean value to store if a particular object's computation is going to happen entirely on cpu or gpu 
		bool cpuorgpu;
    
    //this device array pointer is used in order to avoid computational overhead while computing the next state of the game on the gpu
		int* temporary_arr_for_device;
	
    //copies the host array pointer to the device array pointer
		void copyHostToDevice();

    //copies the device array pointer to the host array pointer
		void copyDeviceToHost();
	
	public:	

    //constructor to initialize the size of the grid and computation type of the object
		GoL(int _m,int _n,bool ifCpuOrGpu);
	
    //function to get the initial state as a host input pointer, configuration of the grid, whether computation should be performed on the CPU or GPU, and update the state variables of the class
		void setInitialState(int m,int n,bool ifCpuOrGpu,int* arr);
    
	//function to get inputs from a file, whose filename is given as a string parameter to the function
		bool getInitialState(string filename);
	
    //function to compute the next state of the grid on the cpu using the host pointer state of the class, and update it 
		void change_of_state_cpu();
    
    //function to compute the next state of the grid on the gpu using the device pointer state of the class, and update it using a temporary device pointer
		void change_of_state_gpu();
	
	//function to return the state pointer of the game on the gpu
		int* getStateGPU();
	
	//function to return the state pointer of the game on the cpu
		int* getStateCPU();
	
	//function to return if the computations are to be performed on the cpu (false) or gpu (true)
		bool getIfCpuOrGpu();
		
};
	
