/*Header file for rendering points in glut and OpenGL*/



#ifdef __APPLE_CC__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <vector>
#include <iostream>
#include <algorithm>
#include <string>
#include <cmath>

//Include headers for cuda
#include <cuda.h>

#ifndef DISPLAY_ENGINE_H
#define DISPLAY_ENGINE_H

class DisplayEngine{

private: 
	int* points;
	int num_cells;
	int window_width;
	int window_ht;

public:
	
	DisplayEngine();
	
	//Standard convention GLUT functions
	void display();
	void init();
	void reshape(int width,int height);

	//Starter code - to be added to later.
	void start();
	
}

#endif // DISPLAY_ENGINE_H