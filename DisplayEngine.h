
/*Making Generic engine for a grid based game*/

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <GLUT/glut.h>
#else
#ifdef _WIN32
  #include <windows.h>
#endif
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#endif

#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <cmath>

//including the game class header
#include "GoL_backend.h"

//Add more headers here


#ifndef DISPLAY_ENGINE_H
#define DISPLAY_ENGINE_H

class DisplayEngine{

private: 
	//Add data members here
	static DisplayEngine* game; // Stores this pointer for game
	
	int rows; //breadth of grid
	int columns; //height of grid
	int n_cells; //number of cells
	
	int fps; // frames per second
	int width; // width of window 
	int height; //height of window
	
	bool useGPU; //flag to parallelize state computations using GPU
	
	GoL_backend game_object;
	//Define more member variables here
	
public:
	
	DisplayEngine();
	DisplayEngine(int argc, char* argv[]);
	~DisplayEngine();
	
	//Callback functions 
	static void initializeWindow();
	static void displayWindowCallback(void);
	static void reshapeWindowCallback(int _w, int _h);
	static void timerCallback(int _t);
	static void mousePositionCallback(int _x, int _y);
	static void mouseClickCallback(int button, int state, int _x, int _y);
	static void keyboardInput(unsigned char _c, int _x, int _y);
	
	void start();
	void displayWindow();
	void reshapeWindow(int _w, int _h);
	
	void renderImageCPU();
	void renderImageCUDA();
	
}

#endif // DISPLAY_ENGINE_H
