#include "DisplayEngine.h"

DisplayEngine::DisplayEngine(){
	//Safety decl
	game = this;
	width = 512;
	height = 512;
	fps = 60;
	// glutInit(1,"default");
}

DisplayEngine::DisplayEngine(int argc, char* argv[]){
	//Init parameters
	game = this;
	width = 512;
	height = 512;
	fps = 60;
	//Initialize glut window
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutInitWindowPosition(80,80);
}

void DisplayEngine::initializeWindow(){
	//Set black background color
	glClearColor(0,0,0,1);
}

//static fn.
void DisplayEngine::displayWindowCallback(){
	game->displayWindow();
}

//static fn.
void DisplayEngine::reshapeWindowCallback(int _w, int _h){
	game->reshapeWindow(_w, _h);
}

//static fn.
void DisplayEngine::refreshWindowCallback(int _t){
	game->refreshWindow(_t);
}

//actually display
void DisplayEngine::displayWindow(){
	
	glClear(GL_COLOR_BUFFER_BIT);
	glLoadIdentity();
	
	if(game_object.getIfCpuOrGpu()==false){
		game_object.change_of_state_cpu();
		renderImageCPU();
	}
	else{
		game_object.change_of_state_gpu();
		renderImageCUDA();
	}
	
	//Actually post
    glutSwapBuffers();
	glutPostRedisplay();

}

void DisplayEngine::reshapeWindow(int _w, int _h){
	
	this->width = _w;
	this->height = _h;
	
	//Set the viewing area from (0,0) with width w and height h
	glViewport(0,0,(GLsizei)_w,(GLsizei)_h);
	
	//Generate a new matrix 
	glMatrixMode(GL_PROJECTION_MATRIX);
	glLoadIdentity();

	//Faster than gluOrtho2d
	gluPerspective(60,(double)_w/_h,0.1,10);

	//Switch back to model view
	glMatrixMode(GL_MODELVIEW);
	glutPostRedisplay();	
	
}

//Refresh window at given fps
void DisplayEngine::refreshWindow(int _t){
	glutPostRedisplay();
	glutTimerFunc(1000/fps,DisplayEngine::refreshWindowCallback,0);
}

void DisplayEngine::start(){
	
	//Initialize data
	game_object.getInitialState("input");
	
	//Create a window
	glutCreateWindow("Conway's Game of Life");
	glutDisplayFunc(DisplayEngine::displayWindowCallback);
	glutReshapeFunc(DisplayEngine::reshapeWindowCallback);
	initializeWindow();
	// glutTimerFunc(50, DisplayEngine::refreshWindowCallback, 0);

	glutMainLoop();
}



