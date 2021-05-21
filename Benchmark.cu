#include "GoL.h"
#include <iostream> 
#include <fstream>
#include <iomanip>

int max_generations = 1000;
int n_small = 9;
int n_large = 20;

// Perform a single run 
void timedRun(int sz, bool small, std::ofstream &myfile, cudaEvent_t start, cudaEvent_t stop){

    // Set input parameters
    auto rows = sz*10;
    auto cols = sz*10;
    
    if(!small){
        rows = sz*100;
        cols = sz*100;

        std::cout << std::setw(5) << (n_small+sz) ;  
    }
    else{
        std::cout << std::setw(5) << (sz) ; 
    }

    auto N = (size_t)rows * (size_t)cols;
    myfile << N << ",";

    std::cout   << std::setw(15) << rows 
                << std::setw(15) << cols 
                << std::setw(15) << N << std::flush;
                
    /* Time CPU execution */
    GoL o_c(rows,cols,false);

    cudaEventRecord(start,0);

    o_c.setRandomInitialState();
    for(auto i=0;i<max_generations;++i){
        o_c.updateState();
    }

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    
    float cpu_time;
    cudaEventElapsedTime(&cpu_time,start,stop);
    
    myfile << cpu_time << ",";
    std::cout << std::setw(15) << cpu_time << std::flush;

    /* Time GPU execution */
    GoL o_g(rows,cols,true);

    cudaEventRecord(start,0);

    o_g.setRandomInitialState();
    for(auto i=0;i<max_generations; ++i){
        o_g.updateState();
    }

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    
    float gpu_time;
    cudaEventElapsedTime(&gpu_time,start,stop);

    myfile << gpu_time << std::endl;
    std::cout << std::setw(15) << gpu_time << std::endl;
}

int main(int argc, char* argv[]){
    
    std::string csv_name = "times.csv";     // default file for storing results

    // Parse cmd line arguments
    for(int i=0;i<argc;++i){
        if(argv[i] == "-g" || argv[i] == "--gen"){
            max_generations = std::stoi(argv[i+1]);
            i++;
        }
        else if(argv[i] == "--csv"){
            csv_name = argv[i+1];
            i++;
        }
    }

    if(max_generations <= 1){
        std::cerr << "Invalid no of max_generations = " << max_generations << std::endl;
        std::cerr << "Terminating benchmarking" << std::endl;
        return 1;
    }

    std::cout << "Running Compute-Only Benchmarks for GameOfLife (CPU vs GPU)" << std::endl;
    std::cout << "# of generations for each grid = " << max_generations  << std::endl;

    // Define stream for output csv file
    std::ofstream myfile;
    myfile.open (csv_name);
    
    if(!myfile.is_open()){
        std::cerr << "Error in opening" << csv_name << std::endl;
        std::cerr << "Terminating benchmarking" << std::endl;
        return 1;
    }

    myfile << "Number of Cells,CPU Time(ms),GPU Time(ms)" << std::endl;
    
    std::cout   << std::setw(5) << "Id"
                << std::setw(15) << "n_rows"
                << std::setw(15) << "n_cols"
                << std::setw(15) << "n_cells"
                << std::setw(15) << "CPU Time(ms)"
                << std::setw(15) << "GPU Time(ms)" 
                << std::endl;
    
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Small random inputs
    for(auto t = 1; t < 10 ; ++t){
        timedRun(t,true,myfile,start,stop);
    }

    // Big random inputs
    for(auto t = 1; t <= 20 ; ++t){
        timedRun(t,false,myfile,start,stop);
    }

    // teardown
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    myfile.close();
    
    return 0; 
}
