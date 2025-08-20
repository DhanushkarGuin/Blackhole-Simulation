## Importing Necessary Libraries
import glfw # For Rendering
from OpenGL.GL import * # For Rendering
import numpy as np # For Mathematical Calculations
from scipy import integrate # For Numerical Integration

## Initializing GLFW
if not glfw.init():
    raise Exception("Failed to initialize GLFW")

## GLFW Window
width,height = 800,600
window = glfw.create_window(width, height, "Black Hole Simulation", None, None)
if not window:
    glfw.terminate()
    raise Exception("Failed to create GLFW window")

## Make the context current
glfw.make_context_current(window)

## Set viewport size
glViewport(0, 0, width, height)

## Visible Window
while not glfw.window_should_close(window):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # For clear screen

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()