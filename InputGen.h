#include "GoL_backend.h"

class InputGen
{
    private:
        
        //variable to hold the number of files created earlier in order to avoid naming clashes
        int test_number;
    
    public:
        
        //constructor to initialize the variable test_number to 0
        InputGen();
    
        /*function to generate 1 test file each for cpu and gpu (in order to benchmark against the same input)
        for particular values of m and n*/
        bool genFileInput(int m,int n);
};
