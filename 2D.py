## Importing Necessary Libraries
import glfw # For Rendering
from OpenGL.GL import *         # For Rendering
import numpy as np              # For Mathematical Calculations
from scipy import integrate     # For Numerical Integration

## Orthographic Projection
def orthographic_projection(width, height, left, right, bottom, top):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(left,right,bottom,top,-1.0,1.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

## Defining Orthographic Volume
view_width = 1.0
view_height = 1.0
left = -view_width
right = view_width
bottom = -view_height
top = view_height

## Initializing GLFW
if not glfw.init():
    raise Exception("Failed to initialize GLFW")

## GLFW Window
width,height = 800,600
window = glfw.create_window(width, height, "Black Hole Simulation", None, None)
if not window:
    glfw.terminate()
    raise Exception("Failed to create GLFW window")

glfw.make_context_current(window)
glViewport(0, 0, width, height)

## Constants
G = 6.67430e-11                         # Gravitational Constant
c = 299792458.0                         # Speed of light
mass_bh = 8.54e36                       # Mass of our Blackhole
r_s = 2 * G * mass_bh / (c ** 2)        # Schwarzschild radius

## Blackhole Class
class Blackhole:
    def __init__(self, position, mass, r_s,viewport_size=2.0, fit_fraction=0.20):
        self.position = np.array(position, dtype=np.float32)
        self.mass = mass
        self.r_s = r_s

        # Scale factor so horizon fits nicely inside viewport
        # viewport_size is (right-left) = 2.0 if [-1,1]
        max_radius = (viewport_size / 2) * fit_fraction
        self.scale = max_radius / self.r_s

    def draw(self, segments=100):
        glPushMatrix()
        # Move to black hole position
        glTranslatef(self.position[0], self.position[1], 0.0)

        # Draw the event horizon as a filled circle
        glColor3f(1.0, 0.0, 0.0)
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(0.0, 0.0)
        for i in range(segments + 1):
            angle = 2.0 * np.pi * i/segments
            x = (self.r_s * self.scale) * np.cos(angle)
            y = (self.r_s * self.scale) * np.sin(angle)
            glVertex2f(x,y)
        glEnd()
        glPopMatrix()

bh = Blackhole([0.0, 0.0], mass_bh, r_s, viewport_size=2.0, fit_fraction=0.20)

class Ray:
    def __init__(self, pos, dir, r_s):
        self.x = pos[0]
        self.y = pos[1]

        # polar coordinates
        self.r = np.sqrt(self.x**2 + self.y**2)
        self.phi = np.arctan2(self.y, self.x)

        # Derivatives: convert velocity vector from Cartesian to polar
        # dr = radial component, dphi = angular velocity
        dir_x, dir_y = dir[0], dir[1]
        self.dr = dir_x * np.cos(self.phi) + dir_y * np.sin(self.phi)
        self.dphi = (-dir_x * np.sin(self.phi) + dir_y * np.cos(self.phi)) / self.r

        # Conserved angular momentum
        self.L = self.r**2 * self.dphi

        # Schwarzschild metric factor f = 1 - r_s / r
        f = 1.0 - r_s / self.r

        # dt/dλ factor based on initial velocities
        dt_dlambda = np.sqrt((self.dr**2) / (f**2) + (self.r**2 * self.dphi**2) / f)
        self.E = f * dt_dlambda
        
        # Trail of positions for visualization (list of (x,y) tuples)
        self.trail = [(self.x, self.y)]

    def update_cartesian(self):
        # Update Cartesian coordinates from polar
        self.x = self.r * np.cos(self.phi)
        self.y = self.r * np.sin(self.phi)

    def record_trail(self):
        # Add current position to trail
        self.trail.append((self.x, self.y))

    def draw(self):
    # Draw current point
        glPointSize(3.0)
        glColor3f(1.0, 1.0, 1.0)  # White color for ray
        glBegin(GL_POINTS)
        glVertex2f(self.x, self.y)
        glEnd()
    
    # Draw trail
        if len(self.trail) < 2:
            return
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glLineWidth(1.0)
        glBegin(GL_LINE_STRIP)
        for i, (tx, ty) in enumerate(self.trail):
            alpha = i / max(len(self.trail) - 1, 1)
            glColor4f(1.0, 1.0, 1.0, alpha)
            glVertex2f(tx, ty)
        glEnd()
        glDisable(GL_BLEND)

left_edge = left  # use your orthographic left boundary
start_x = left_edge  # or a little inside, e.g. left + margin

num_rays = 10
y_values = np.linspace(bottom, top, num_rays)
direction = np.array([1e-1, 0.0])  # magnitude scales speed

rays = []
for y in y_values:
    pos = np.array([start_x, y])
    rays.append(Ray(pos, direction, bh.r_s * bh.scale))

## Visible Window
while not glfw.window_should_close(window):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)              # For clear screen
    orthographic_projection(width, height, left, right, bottom, top)

    # For Blackhole
    bh.draw()

    # For Rays
    for ray in rays:
        ray.r += 0.001
        ray.phi += 0.001
        ray.update_cartesian()
        ray.record_trail()
        ray.draw()

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()