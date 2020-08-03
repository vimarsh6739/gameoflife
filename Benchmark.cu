#include "GoL.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <string>
#include <cmath>
#include <ctime>

//Function to convert clock ticks to milliseconds 
double clockToMilliseconds(clock_t ticks)
{	
	//CLOCKS_PER_SEC is system dependent
	return 1000.0 * ((double)ticks/CLOCKS_PER_SEC);
}

int main(int argc, char* argv[]){
    
    // Default number of grid rows and columns
    int rows = 10;
    int columns = 10;
    
    // Have a default computation of 10000 generations
    int maxgeneration = 10000;
    
    // Define variables for timing code in both CPU and GPU
    clock_t beginCPU = 0;
    clock_t endCPU = 0;

    clock_t beginGPU = 0;
    clock_t endGPU = 0;

    // Define stream for output file
    std::ofstream myfile;
    myfile.open ("times.txt");


    // Parse cmd line arguments
    for(int i=0;i<argc;++i)
    {
        if(argv[i] == "--rows" || argv[i] == "-r")
        {
            rows = std::stoi(argv[i+1]);
            i++;
        }
        else if(argv[i] == "--columns" || argv[i] == "-c")
        {
            columns = std::stoi(argv[i+1]);
            i++;
        }
        else if(argv[i] == "-g")
        {
            maxgeneration = std::stoi(argv[i+1]);
            i++;
        }
    }

    if(rows <= 0 || columns <= 0 || maxgeneration <= 0)
    {
        std::cout<<"Error in inputs - terminating benchmarking"<<std::endl;
    }

    // std::cout<<rows<<"\n"<<columns<<"\n"<<maxgeneration<<"\n";

    // Begin timing cpu computation for random input on (r,c)

    beginCPU = clock();
    
    GoL* cpu_object = new GoL(rows, columns, false);
    cpu_object->setRandomInitialState();
    
    for(auto i=0;i < maxgeneration ; ++i)
    {
        cpu_object->updateState();
    }

    endCPU = clock();

    double timeCPU = clockToMilliseconds(endCPU - beginCPU);
    myfile<<timeCPU<<"\n";

    // Begin timing GPU computation for random input on (r,c)

    beginGPU = clock();

    GoL* gpu_object = new GoL(rows, columns, true);
    gpu_object->setRandomInitialState();

    for(auto i=0;i<maxgeneration;++i)
    {
        gpu_object->updateState();
    }

    endGPU = clock();

    double timeGPU = clockToMilliseconds(endGPU - beginGPU);
    myfile<<timeGPU;

    // save and exit
    myfile.close();
    return 0;
    
}
