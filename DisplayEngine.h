
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

#include "CPU_gol.h"

#ifndef DISPLAY_ENGINE_H
#define DISPLAY_ENGINE_H

class DisplayEngine
{

private: 
	
	static DisplayEngine* game; // Stores this pointer for game
	
	//Basic window variables for grid game---------------
	
	int windowWidth; 	 
	int windowHeight; 	
	
	//Number of cells
	int N;

	//Number of rows in the grid
	int rows; 		
	
	//Number of columns in the grid
	int columns; 	

	//Pause animation
	bool pause;

	// Benchmarking Variables---------------

	//Debugging frame rate(maually set in debug mode)
	int FPS_DEBUG; 	

	//Total time taken to render all frames in millis
	double totalTime;

	//Average time taken to draw 1 frame
	double avgFrameTime; 

	//Current avg frame rate
	double FPS;		

	//Start time of program
	clock_t startTick;	

	// Begin and end ticks for drawing image
	clock_t beginDrawTick;	
	clock_t endDrawTick;	

	//Actual time taken to draw current frame
	clock_t deltaTime;	

	//No of frames drawn in 1 sec
	int frameCount;		

	//Total number of updates in the game so far
	int generationCount;
	//----------------------------------------------

	//Object of the game class that contains the state variables 
	//and the update and initialization functions and upddates the grid 
	//according to the specified rules
	CPU_gol* game_object;
	
	//Texture buffer id
	GLuint texture_map;	

public:
	
	DisplayEngine();
	DisplayEngine(int argc, char* argv[]);
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

	void initTexture();
	void updateTexture();
	void drawTexture();

	void updateStats();
	void displayStats();
	void windowPrint(int x, int y, const char* str);
	double clockToMilliseconds(clock_t ticks);
	
	void start();
	void initParams();

};

#endif // DISPLAY_ENGINE_H