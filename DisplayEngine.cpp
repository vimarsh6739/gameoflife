#include "DisplayEngine.h"

//Declaration for static variable game
DisplayEngine* DisplayEngine::game = NULL;

DisplayEngine::DisplayEngine()
{
	//Safety decl
	game = this;
	width = 512;
	height = 512;
	rows = height;
	columns = width;
	N = rows * columns;
	fps = 10;
}

DisplayEngine::DisplayEngine(int argc, char* argv[])
{
	game = this;
	
	width = 512;
	height = 512;
	
	rows = 32;
	columns = 32;
	N = rows * columns;
	fps = 10;

	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutInitWindowSize(width, height);
	glutInitWindowPosition(80,80);
}

//reshape callback
void DisplayEngine::reshapeWindowCallback(int _w, int _h)
{
	game->reshapeWindow(_w, _h);
}

void DisplayEngine::reshapeWindow(int _w, int _h)
{
	this->width = _w;
	this->height = _h;

	//Set the viewing area from (0,0) with width w and height h
	glViewport(0,0,(GLsizei)_w,(GLsizei)_h);

	//Generate new projection matrix
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	//Set coordinate space for window
	gluOrtho2D(0.0,1.0*rows,0.0,1.0*columns);
	glMatrixMode(GL_MODELVIEW);	
}

//Timer callback
void DisplayEngine::refreshWindowCallback(int _t)
{
	game->refreshWindow(_t);
}

void DisplayEngine::refreshWindow(int _t)
{
	glutPostRedisplay();
	glutTimerFunc(1000/fps,DisplayEngine::refreshWindowCallback,0);
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

//Update the texture object after updating state information
void DisplayEngine::updateTexture()
{
	glBindTexture(GL_TEXTURE_2D, texture_map);
	glTexSubImage2D(GL_TEXTURE_2D,0,0,0,rows,columns,GL_RGB,GL_FLOAT,game_object->getStateColours());
	glBindTexture(GL_TEXTURE_2D,0);
}

//display callback
void DisplayEngine::displayWindowCallback()
{
	game->displayWindow();
}

//Update automation state and display
void DisplayEngine::displayWindow()
{
	//Clear screen
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);	
	
	//Update state and corresponding texture object
	game_object->updateState();
	updateTexture();

	//Display grid in texture
	drawTexture();
	//drawLoop();

	//Swap buffers
    glutSwapBuffers();
}

void DisplayEngine::start()
{
	std::cout<<"Starting display\n";

	//Start by setting up window.
	glutCreateWindow("Conway's Game of Life");
	glutDisplayFunc(DisplayEngine::displayWindowCallback);
	glutReshapeFunc(DisplayEngine::reshapeWindowCallback);
	glutTimerFunc(0, DisplayEngine::refreshWindowCallback, 0);
	initParams();
	glutMainLoop();

	std::cout<<"Finished displaying game of life\n";
}

//Initialize all parameters for game
void DisplayEngine::initParams()
{	
	//Set black background color
	glClearColor(0,0,0,1);
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

	//Initializing grid for computation - random initialization
	game_object = new CPU_gol(rows,columns);
	game_object->randInit();
	
	std::cout<<"Creating texture\n";
	initTexture();
	std::cout<<"Finished creating texture\n";
}

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