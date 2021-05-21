#include "GoL.h"

#include <iostream> 
#include <fstream>
#include <iomanip>

#include <time.h>
#include <sys/time.h>
#define USECPSEC 1000000ULL

unsigned long long dtime_usec(unsigned long long start){

  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

int main(int argc, char* argv[]){
    
    // Have a default computation length of 1000 generations
    int max_generations = 1000;
    std::string csv_name = "times.csv";

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

    myfile      << "Number of Cells, CPU Time(us), GPU Time(us)" << std::endl;
    
    std::cout   << std::setw(3) << "Id"
                << std::setw(5)  << "Rows"
                << std::setw(5)  << "Cols"
                << std::setw(12) << "CPU Time(s)"
                << std::setw(12) << "GPU Time(s)" << std::endl;

    auto rows = 0;
    auto cols = 0;
    
    for(auto t = 1; t <= 25 ; ++t){
        
        rows = t*100;
        cols = t*100;
        auto N = (size_t)rows * (size_t)cols;
        
        std::cout << std::setw(3) << t << std::setw(5) << rows << std::setw(5) << cols << std::flush;
        myfile << N << ",";

        /* CPU execution */
        GoL o_c(rows,cols,false);
        auto cpu_time = dtime_usec(0);

        o_c.setRandomInitialState();
        for(auto i=0;i<max_generations;++i){
            o_c.updateState();
        }

        cpu_time = dtime_usec(cpu_time);

        std::cout << std::setw(12) << cpu_time/(float)USECPSEC << std::flush;
        myfile << cpu_time << ",";

        /* GPU execution */
        GoL o_g(rows,cols,true);
        auto gpu_time = dtime_usec(0);

        o_g.setRandomInitialState();
        for(auto i=0;i<max_generations; ++i){
            o_g.updateState();
        }

        gpu_time = dtime_usec(gpu_time);

        std::cout << std::setw(12) << gpu_time/(float)USECPSEC << std::endl;
        myfile << gpu_time << std::endl;
    }

    // save and exit
    myfile.close();
    
    return 0; 
}
