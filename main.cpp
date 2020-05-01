
#include "DisplayEngine.h"

int main(int argc, char* argv[])
{
	//Create instance 
	DisplayEngine* e = new DisplayEngine(argc,argv);
	
	//Start simulation
	e->start();
	
	return 0;
}

