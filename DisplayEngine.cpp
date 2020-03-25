#include "DisplayEngine.h"


DisplayEngine::DisplayEngine(){
	//To be filled later
}

void DisplayEngine::init(){
	//Set background as black
	glClearColor(0,0,0,1);
}

void DisplayEngine::reshape(int w, int h){

	//Set the viewing area from (0,0) with width w 
	//and height h
	glViewport(0,0,(GLsizei)w,(GLsizei)h);
	
	//Generate a new matrix 
	glMatrixMode(GL_PROJECTION_MATRIX);
	glLoadIdentity();

	//Faster than gluOrtho2d
	gluPerspective(45,(double)w/h,0.1,10);

	//Switch back to model view
	glMatrixMode(GL_MODELVIEW);
	glutPostRedisplay();
}

void DisplayEngine::display(){
	
	glClear(GL_COLOR_BUFFER_BIT);
	glLoadIdentity();
	
	//Draw a triangle for now :P
	glBegin(GL_TRIANGLES);
    glColor3f( 1, 1, 1 ); // red
    glVertex2f( -0.8, -0.8 );
    glColor3f( 1, 1, 1 ); // green
    glVertex2f( 0.8, -0.8 );
    glColor3f( 1, 1, 1 ); // blue
    glVertex2f( 0, 0.9 );
    glEnd(); 
	// draw2d();

    //Swap animation buffers
    glutSwapBuffers();
    //Display again
	glutPostRedisplay();
}

void DisplayEngine::start(){

	//Initialize env with argc and argv
	glutInit(&argc,argv);
	//Color display with double buffer for smooth anim
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);

	glutInitWindowPosition(80,80);
	glutCreateWindow("Conway's Game of Life");
	
	glutReshapeFunc(DisplayEngine::reshape);
  	glutDisplayFunc(DisplayEngine::display);
  	init();//Initialize display params

  	glutMainLoop();
}


