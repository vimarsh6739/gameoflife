#include "GoL.h"

#include <fstream>
#include <iostream>
#include <sstream>

class Generator
{
    private:
        
        //variable to hold the number of files created earlier in order to avoid naming clashes
        int test_number;
    
    public:
        
        //constructor to initialize the variable test_number to 1
        Generator();
    
        /*function to generate 1 test file for particular values of m, n and the computation device used,
        and returns the name of the file*/
        std::string genFileInput(int m,int n,bool ifCpuOrGpu);
    
        /*function to copy the contents of a given file and just create a new file with the same input
        but with a different computation device*/
        std::string switchedComputationFile(std::string filename);
};
