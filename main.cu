
//Starter class for entire program

#include "DisplayEngine.h"

int main(int argc, char* argv[])
{

	DisplayEngine* obj;
	std::string nodisp;
	std::string isfile;
	std::string fname;
	int rows,cols;

	std::cout<<"Is there a configuration file? (y/n): ";
	std::cin>>isfile;

	if(isfile[0] == 'y')
	{
		//Input from file
		std::cout<<"Enter filename: ";
		std::cin>>fname;
		obj = new DisplayEngine(fname);
	}
	else
	{
		//Random Input for some (rows,columns)
		std::cout<<"Enter the number of rows in the grid: ";
		std::cin>>rows;

		std::cout<<"Enter the number of columns in the grid: ";
		std::cin>>cols;

		obj = new DisplayEngine(rows,cols);
	}

	//Start rendering
	obj->start(argc,argv);

	return 0;
}

