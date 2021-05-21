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
#include <sstream>
#include <algorithm>
#include <string>
#include <cmath>
#include <ctime>

#include <cuda.h>
#include "GoL.h"

#ifndef DISPLAY_ENGINE_H
#define DISPLAY_ENGINE_H

class DisplayEngine
{

private: 
	
	static DisplayEngine* game; // Stores this pointer for game
	
	/* Environment variables for grid game */
	
	bool isRandInit;			//Flag to indicate if config file has been provided
	std::string config_file;	//File which contains config details
	
	int windowWidth;			// Dynamic width of display window 	 
	int windowHeight; 			// Dynamic height of display window
	
	int N;						//Number of cells
	int rows; 					//Number of rows in the grid
	int columns; 				//Number of columns in the grid

	
	bool pause;					//Pause animation flag
	bool isGpu;					//Flag to check area of computation(CPU or GPU)

	float* state_color;			// Color encoded state variable(in host mem.)

	GLuint texture_map;			// Texture buffer id

	/* Benchmarking Variables */

	int FPS_DEBUG; 				// Debugging frame rate(maually set in debug mode)

	double totalTime;			// Total time taken to render all frames in millis
	double avgFrameTime; 		// Average time taken to draw 1 frame
	double FPS;					// Current avg frame rate

	clock_t startTick;			// Start time of program

	clock_t beginDrawTick;		// Begin and end ticks 
	clock_t endDrawTick;		// for drawing image
	clock_t deltaTime;			//Actual time taken to draw current frame

	int frameCount;				//No of frames drawn in 1 sec	

	int generationCount;		//Total number of updates in the game so far

	//Object of the game class that contains the state variables 
	//and the update and initialization functions and upddates the grid 
	//according to the specified rules
	GoL* game_object;
	
		

public:
	
	//Defining constructors for random input and 
	//configuration files

	DisplayEngine();
	DisplayEngine(int rows, int cols);
	DisplayEngine(std::string fname);
	~DisplayEngine();
	
	//Callback functions 

	static void displayWindowCallback(void);
	static void reshapeWindowCallback(int _w, int _h);
	static void timerWindowCallback(int _t);
	static void idleWindowCallback();
	static void keyboardInputCallback(unsigned char _c, int _x, int _y);

	// static void mousePositionCallback(int _x, int _y);
	// static void mouseClickCallback(int button, int state, int _x, int _y);
	
	void displayWindow();
	void reshapeWindow(int _w, int _h);
	void timerWindow(int _t);
	void idleWindow();
	void keyboardInput(unsigned char _c, int _x, int _y);

	void drawLoop();
	void drawCell(int x, int y);
	void drawTexture();
	void updateTexture();

	void updateStats();
	void displayStats();
	void windowPrint(int x, int y, const char* str);
	double clockToMilliseconds(clock_t ticks);
	
	void initializeTexture();
	void initializeParameters();
	void start(int argc, char* argv[]);
};

#endif // DISPLAY_ENGINE_H
