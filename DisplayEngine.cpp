#include "DisplayEngine.h"

//Declaration for static variable game
DisplayEngine* DisplayEngine::game = NULL;

DisplayEngine::DisplayEngine(){
	//Safety decl
	game = this;
	width = 500;
	height = 500;
	rows = height;
	columns = width;
	n_cells = rows * columns;
	fps = 10;
}

DisplayEngine::DisplayEngine(int argc, char* argv[]){
	
	//Init parameters
	game = this;
	
	width = 1000;
	height = 1000;
	
	rows = 1000;
	columns = 1000;
	n_cells = rows * columns;
	
	fps = 60;

	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutInitWindowSize(width, height);
	glutInitWindowPosition(80,80);
}

//display callback
void DisplayEngine::displayWindowCallback(){
	game->displayWindow();
}

//reshape callback
void DisplayEngine::reshapeWindowCallback(int _w, int _h){
	game->reshapeWindow(_w, _h);
}

//timer callback
void DisplayEngine::refreshWindowCallback(int _t){
	game->refreshWindow(_t);
}

//Initialize all parameters for game
void DisplayEngine::initParams(){
	
	//Set black background color
	glClearColor(0,0,0,1);

	//Initialize game_object 
	game_object = new CPU_gol(rows,columns);
	game_object->randInit();
	// game_object->printCells();
}

//Update automation state and display
void DisplayEngine::displayWindow(){
	
	glClear(GL_COLOR_BUFFER_BIT);
<<<<<<< Updated upstream
	glLoadIdentity();
	
	if(this->useGPU){
		updateStateCPU();
=======
	// glLoadIdentity();
	// std::cout<<"In display window::"<<"\n";
	/*if(game_object.getIfCpuOrGpu()==false){
		game_object.change_of_state_cpu();
>>>>>>> Stashed changes
		renderImageCPU();
	}
	else{
		updateStateCUDA();
		renderImageCUDA();
	}*/
	
	game_object->update_state();
	renderImageCPU();
	
    glutSwapBuffers();
   	
}

void DisplayEngine::reshapeWindow(int _w, int _h){
	
	this->width = _w;
	this->height = _h;
	
	//Set the viewing area from (0,0) with width w and height h
	glViewport(0,0,(GLsizei)_w,(GLsizei)_h);
	
	//Generate a new matrix 
	glMatrixMode(GL_PROJECTION_MATRIX);
	glLoadIdentity();

	// gluPerspective(60,(double)_w/_h,0.1,10);
	gluOrtho2D(0.0,1.0*rows,0.0,1.0*columns);

	//Switch back to model view
	glMatrixMode(GL_MODELVIEW);	
	
}

//Refresh window at given fps
void DisplayEngine::refreshWindow(int _t){
	glutPostRedisplay();
	glutTimerFunc(1000/fps,DisplayEngine::refreshWindowCallback,0);
}

// Method to render game grid in the cpu
void DisplayEngine::renderImageCPU(){

	//loop over all cells and display if they are part of the automation
	for(int x = 0; x < rows ; ++x){
		for(int y = 0; y < columns; ++y){
			if(game_object->isAlive(x,y)){
				glColor3f((float)x/rows, (float)(x^y)/(rows + columns), (float)y/rows); 
				drawCell(x,y);
			}
		}
	}

	// drawCell(20,20);
}

void DisplayEngine::drawCell(int x, int y){
	//Draw a cell at (x,y)
	
	glBegin(GL_QUADS);                 
       	glVertex3f(x,y,0.0);
      	glVertex3f(x+1,y,0.0);
      	glVertex3f(x+1,y+1,0.0);
      	glVertex3f(x,y+1,0.0);
   glEnd();
}

void DisplayEngine::start(){
	
	//Initialize data
<<<<<<< Updated upstream
	this->initializeInputs();
	
	//Create a window
	glutCreateWindow("Conway's Game of Life");
=======
	// game_object.getInitialState("input");
	// game = this;

	glutCreateWindow("Bouncy Blobby Bob");
>>>>>>> Stashed changes
	glutDisplayFunc(DisplayEngine::displayWindowCallback);
	glutReshapeFunc(DisplayEngine::reshapeWindowCallback);
	initParams();
	glutTimerFunc(0, DisplayEngine::refreshWindowCallback, 0);
	glutMainLoop();

}
