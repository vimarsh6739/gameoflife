#include "Generator.h"

Generator::Generator()
{
    test_number=1;
}

//since each call to the function creates two files, both files will have the prefix "cpu" and "gpu" as needed, followed by the test number
std::string Generator::genFileInput(int m,int n,bool ifCpuOrGpu)
{
    std::string filename="";
    if(ifCpuOrGpu)
    {
        filename="gpu";
    }
    else
    {
        filename="cpu";
    }
    filename+=std::to_string(test_number);
    test_number++;
    std::vector<std::vector<int>> arr(m,std::vector<int>(n));
    srand(time(NULL));
    for(int i=0;i<m;++i)
        for(int j=0;j<n;++j)
            arr[i][j]=rand()%2;
    ofstream myfile;
    myfile.open(filename);
    myfile<<m<<" "<<n<<"\n"<<ifCpuOrGpu<<"\n";
    for(int i=0;i<m;++i)
    {
        for(int j=0;j<n;++j)
            myfile<<arr[i][j]<<" ";
        myfile<<"\n";
    }
    myfile.close();
    return filename;
}

std::string Generator::switchComputationFile(std::string filename)
{
    std::string outstring=filename;
    int m,n;
    bool ifCpuOrGpu;
    if(filename[0]=='c')
        outstring[0]='g';
    else
        outstring[0]='c';
    ifstream fin;
    fin.open(filename);
    fin>>m>>n>>ifCpuOrGpu;
    ifCpuOrGpu=!ifCpuOrGpu;
    std::vector<std::vector<int>> arr(m,std::vector<int>(n));
    for(int i=0;i<m;++i)
        for(int j=0;j<n;++j)
            fin>>arr[i][j];
    fin.close();
    ofstream myfile;
    myfile.open(outstring);
    myfile<<m<<" "<<n<<"\n"<<ifCpuOrGpu<<"\n";
    for(int i=0;i<m;++i)
    {
        for(int j=0;j<n;++j)
            myfile<<arr[i][j]<<" ";
        myfile<<"\n";
    }
    myfile.close();
    return outstring;
}
