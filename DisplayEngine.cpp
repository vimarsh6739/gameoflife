#include "DisplayEngine.h"

#define DEBUG 0

//Declaration for static variable game
DisplayEngine* DisplayEngine::game = NULL;

//Function to convert clock ticks to milliseconds 
//for frame rate calculation
double DisplayEngine::clockToMilliseconds(clock_t ticks)
{	
	//CLOCKS_PER_SEC is system dependent
	return 1000.0 * ((double)ticks/CLOCKS_PER_SEC);
}

//Default constructor
DisplayEngine::DisplayEngine()
{
	game = this;
	windowWidth = 512;
	windowHeight = 512;
	rows = windowHeight;
	columns = windowWidth;
	N = rows * columns;
	FPS_DEBUG = 10;

	totalTime = 0.0;
	avgFrameTime = 0.0;
	FPS = 0.0;
	startTick = 0;
	beginDrawTick = 0;
	endDrawTick = 0;
	deltaTime = 0;
	frameCount = 0;
	generationCount = 0;
}

DisplayEngine::DisplayEngine(int argc, char* argv[])
{
	game = this;
	
	windowWidth = 1024;
	windowHeight = 1024;
	rows = 128;
	columns = 128;
	N = rows * columns;

	FPS_DEBUG = 10;

	totalTime = 0.0;
	avgFrameTime = 0.0;
	FPS = 0.0;
	startTick = 0;
	beginDrawTick = 0;
	endDrawTick = 0;
	deltaTime = 0;
	frameCount = 0;
	generationCount = 0;

	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutInitWindowSize(windowWidth, windowHeight);
	glutInitWindowPosition(80,80);
}

//Reshape callback
void DisplayEngine::reshapeWindowCallback(int _w, int _h)
{
	game->reshapeWindow(_w, _h);
}

void DisplayEngine::reshapeWindow(int _w, int _h)
{
	windowWidth = _w;
	windowHeight = _h;

	//Set the viewing area from (0,0) with width w and height h
	glViewport(0,0,(GLsizei)_w,(GLsizei)_h);

	//Generate new projection matrix
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	//Set coordinate space for window
	gluOrtho2D(0.0,1.0*rows,0.0,1.0*columns);
	glMatrixMode(GL_MODELVIEW);	
}

//Initialize all parameters for game
void DisplayEngine::initParams()
{	
	//Set black background color
	glClearColor(0,0,0,1);
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

	//Initializing grid for computation - random initialization
	game_object = new CPU_gol(rows,columns);
	game_object->randomInitialState();
	
	std::cout<<"Creating texture\n";
	initTexture();
	std::cout<<"Finished creating texture\n";
}

// Loop over all cells and draw
void DisplayEngine::drawLoop()
{
	//loop over all cells and display
	for(int x = 0; x < rows ; ++x)
	{
		for(int y = 0; y < columns; ++y)
		{
			if(game_object->isAlive(x,y))
			{
				glColor3f(1.0f, 1.0f, 1.0f); 
				drawCell(x,y);
			}
		}
	}
}

//Draw a single cell at position (x,y)
void DisplayEngine::drawCell(int x, int y)
{
	glBegin(GL_QUADS);                 
    
    glVertex3f(x,y,0.0);
    glVertex3f(x+1,y,0.0);
    glVertex3f(x+1,y-1,0.0);
    glVertex3f(x,y-1,0.0);
   	
   	glEnd();
}

//Initialize texture components
void DisplayEngine::initTexture()
{

	glGenTextures(1,&texture_map);
	glBindTexture(GL_TEXTURE_2D,texture_map);

	//Set interpolation to nearest for points outside bounding box
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	//We want to clamp the bitmap we have to the borders of the image
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

	//Map the color array state_colors in the game object to the current texture
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,rows,columns,0,GL_RGB,GL_FLOAT,game_object->getStateColours());
	glBindTexture(GL_TEXTURE_2D,0);
}

//Update the texture object after updating state information
void DisplayEngine::updateTexture()
{
	glBindTexture(GL_TEXTURE_2D, texture_map);
	glTexSubImage2D(GL_TEXTURE_2D,0,0,0,rows,columns,GL_RGB,GL_FLOAT,game_object->getStateColours());
	glBindTexture(GL_TEXTURE_2D,0);
}

//Draw from texture map
void DisplayEngine::drawTexture()
{
	glBindTexture(GL_TEXTURE_2D,texture_map);
	glEnable(GL_TEXTURE_2D);

	glBegin(GL_QUADS);

		//Specify bottom left
		glTexCoord2i(0,0);
		glVertex2i(0,0);

		//Specify top left
		glTexCoord2i(0,1);
		glVertex2i(0,columns);
		
		//Specify top right
		glTexCoord2i(1,1);
		glVertex2i(rows,columns);
		
		//Specify bottom right
		glTexCoord2i(1,0);
		glVertex2i(rows,0);
	
	glEnd();

	glDisable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D,0);
}

//Update the dynamic FPS and display it
void DisplayEngine::updateStats()
{
	++frameCount;
	++generationCount;

	//No of clock cycles spent in rendering current img
	deltaTime = endDrawTick - beginDrawTick;

	//Total time is measured in milliseconds
	totalTime += clockToMilliseconds(deltaTime);

	//Number of milliseconds to render a frame on avg
	avgFrameTime = totalTime/frameCount;

	//Dynamic FPS
	FPS = 1000.0 * frameCount/totalTime;

	// std::cout<<FPS<<"\n";
}

//Print the text in msg starting at coordinates (x,y)
void DisplayEngine::windowPrint(int x, int y, const char* msg)
{
	glColor3f(0.0,1.0,0.0);
	glRasterPos2f(x,y);
	int len = (int)strlen(msg);
	for(int i=0;i<len;++i)
	{
		glutBitmapCharacter(GLUT_BITMAP_8_BY_13,msg[i]);
	}
	glColor3f(1.0,1.0,1.0);
}

//Display Frame Rate, time to render and 
//other parameters in the display window itself
void DisplayEngine::displayStats()
{
	//Pointer to constat string
	const char* ptr;

	//Display FPS at (0,0)
	
	std::stringstream ss;
	ss<<"FPS = "<<FPS;
	ptr = (ss.str()).c_str();
	windowPrint(0,0,ptr);

}

//Display callback
void DisplayEngine::displayWindowCallback()
{
	game->displayWindow();
}

//Update automation state and display
void DisplayEngine::displayWindow()
{
	//Clear screen
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);	
	
	//Update current grid state, display and benchmark it
	
	beginDrawTick = clock();

	game_object->updateState();
	
	updateTexture();
	drawTexture();
	displayStats();
	
	endDrawTick = clock();
	
	//Update and display render stats
	updateStats();

	//Swap buffers
    glutSwapBuffers();
}

//Process keyboard character entered
void DisplayEngine::keyboardInputCallback(unsigned char _c, int _x, int _y)
{
	game->keyboardInput(_c, _x, _y);	
}

void DisplayEngine::keyboardInput(unsigned char _c, int _x, int _y)
{
	//Spacebar pressed for pausing animation
	if(_c == ' ')
	{
		pause = !pause;
		if(pause==false){
			glutPostRedisplay();
		}
	}
}

//Idle callback
void DisplayEngine::idleWindowCallback()
{
	game->idleWindow();
}

void DisplayEngine::idleWindow()
{
	if(!pause){
		glutPostRedisplay();
	}
}

//Timer callback
void DisplayEngine::timerWindowCallback(int _t)
{
	game->timerWindow(_t);
}

void DisplayEngine::timerWindow(int _t)
{
	glutPostRedisplay();
	glutTimerFunc(1000/FPS_DEBUG,DisplayEngine::timerWindowCallback,0);
}

void DisplayEngine::start()
{
	std::cout<<"Starting display\n";

	//Start by setting up window.
	glutCreateWindow("Conway's Game of Life");

	//Register callback functions
	glutDisplayFunc(DisplayEngine::displayWindowCallback);
	glutReshapeFunc(DisplayEngine::reshapeWindowCallback);
	glutKeyboardFunc(DisplayEngine::keyboardInputCallback);

	if(DEBUG) 
	{
		//Limiting Frame Rate for Debugging
		glutTimerFunc(0, DisplayEngine::timerWindowCallback, 0);
	}
	else
	{
		//Maximum speed rendering
		glutIdleFunc(DisplayEngine::idleWindowCallback);
	}

	initParams();
	glutMainLoop();

}