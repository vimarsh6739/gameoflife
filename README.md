# gameoflife
CS6023(GPU Programming) Project to simulate and render Conway's Game of Life in Cuda C++ and OpenGL

To compile the project for the GUI visualization, run
`make display`

To begin the display after that, do
`./gui`

The display supports resizing, and has the option to toggle between CPU and GPU computation(using the key `T`). You can pause the display using the `Space` key.

The rendering is done using a 2D texture with the help of the GLUT Library. FPS is also displayed on the screen during the running of the program.
