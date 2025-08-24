import sys
import math
import glfw
import numpy as np
from OpenGL.GL import *

# --- Constants --- #
c = 299792458.0
G = 6.67430e-11


# --- Engine --- #
class Engine:
    def __init__(self, width=600, height=400):
        if not glfw.init():
            print("Failed to initialize GLFW")
            sys.exit(1)

        self.WIDTH = width
        self.HEIGHT = height
        self.window = glfw.create_window(width, height, "Black Hole Simulation", None, None)
        if not self.window:
            print("Failed to create GLFW window")
            glfw.terminate()
            sys.exit(1)

        glfw.make_context_current(self.window)

        # Viewport scaling in meters
        self.width = 1.0e11
        self.height = 7.5e10

        # Navigation state
        self.offsetX = 0.0
        self.offsetY = 0.0
        self.zoom = 1.0

        glViewport(0, 0, width, height)

    def run(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        left   = -self.width + self.offsetX
        right  =  self.width + self.offsetX
        bottom = -self.height + self.offsetY
        top    =  self.height + self.offsetY
        glOrtho(left, right, bottom, top, -1.0, 1.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()


engine = Engine()


# --- Black Hole --- #
class BlackHole:
    def __init__(self, pos, mass):
        self.position = np.array(pos, dtype=float)
        self.mass = mass
        self.r_s = 2.0 * G * mass / (c * c)  # Schwarzschild radius

    def draw(self):
        glBegin(GL_TRIANGLE_FAN)
        glColor3f(1.0, 0.0, 0.0)  # Red color
        glVertex2f(0.0, 0.0)
        for i in range(101):
            angle = 2.0 * math.pi * i / 100
            x = self.r_s * math.cos(angle)
            y = self.r_s * math.sin(angle)
            glVertex2f(x, y)
        glEnd()


SagA = BlackHole((0.0, 0.0, 0.0), 8.54e36)  # Sagittarius A


# --- Ray --- #
class Ray:
    def __init__(self, pos, direction):
        # Cartesian coords
        self.x, self.y = pos
        # Polar coords
        self.r = math.sqrt(self.x**2 + self.y**2)
        self.phi = math.atan2(self.y, self.x)

        # Velocities
        self.dr = direction[0] * math.cos(self.phi) + direction[1] * math.sin(self.phi)
        self.dphi = (-direction[0] * math.sin(self.phi) + direction[1] * math.cos(self.phi)) / self.r

        # Conserved quantities
        self.L = self.r**2 * self.dphi
        f = 1.0 - SagA.r_s / self.r
        dt_dλ = math.sqrt((self.dr**2) / (f*f) + (self.r**2 * self.dphi**2) / f)
        self.E = f * dt_dλ

        # Trail
        self.trail = [(self.x, self.y)]

    def draw(self, rays):
        # Draw current ray positions
        glPointSize(2.0)
        glColor3f(1.0, 0.0, 0.0)
        glBegin(GL_POINTS)
        for ray in rays:
            glVertex2f(ray.x, ray.y)
        glEnd()

        # Trails
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glLineWidth(1.0)

        for ray in rays:
            N = len(ray.trail)
            if N < 2:
                continue
            glBegin(GL_LINE_STRIP)
            for i, (tx, ty) in enumerate(ray.trail):
                alpha = i / (N - 1)
                glColor4f(1.0, 1.0, 1.0, max(alpha, 0.05))
                glVertex2f(tx, ty)
            glEnd()

        glDisable(GL_BLEND)

    def step(self, dλ, rs):
        if self.r <= rs:
            return
        rk4Step(self, dλ, rs)

        # Back to Cartesian
        self.x = self.r * math.cos(self.phi)
        self.y = self.r * math.sin(self.phi)

        self.trail.append((self.x, self.y))


# --- Geodesic Function --- #
def geodesicRHS(ray, rs):
    r, dr, dphi, E = ray.r, ray.dr, ray.dphi, ray.E
    f = 1.0 - rs / r

    rhs = [0.0] * 4
    rhs[0] = dr
    rhs[1] = dphi
    dt_dλ = E / f
    rhs[2] = (
        - (rs / (2 * r * r)) * f * (dt_dλ**2)
        + (rs / (2 * r * r * f)) * (dr**2)
        + (r - rs) * (dphi**2)
    )
    rhs[3] = -2.0 * dr * dphi / r
    return rhs


def addState(a, b, factor):
    return [a[i] + b[i] * factor for i in range(4)]

# --- Range Kutta 4 Function --- #
def rk4Step(ray, dλ, rs):
    y0 = [ray.r, ray.phi, ray.dr, ray.dphi]

    k1 = geodesicRHS(ray, rs)
    temp = addState(y0, k1, dλ/2.0)
    r2 = Ray((ray.x, ray.y), (0, 0))
    r2.r, r2.phi, r2.dr, r2.dphi, r2.E = temp[0], temp[1], temp[2], temp[3], ray.E
    k2 = geodesicRHS(r2, rs)

    temp = addState(y0, k2, dλ/2.0)
    r3 = Ray((ray.x, ray.y), (0, 0))
    r3.r, r3.phi, r3.dr, r3.dphi, r3.E = temp[0], temp[1], temp[2], temp[3], ray.E
    k3 = geodesicRHS(r3, rs)

    temp = addState(y0, k3, dλ)
    r4 = Ray((ray.x, ray.y), (0, 0))
    r4.r, r4.phi, r4.dr, r4.dphi, r4.E = temp[0], temp[1], temp[2], temp[3], ray.E
    k4 = geodesicRHS(r4, rs)

    ray.r    += (dλ/6.0) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    ray.phi  += (dλ/6.0) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    ray.dr   += (dλ/6.0) * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    ray.dphi += (dλ/6.0) * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3])

    ray.x = ray.r * math.cos(ray.phi)
    ray.y = ray.r * math.sin(ray.phi)


# --- Main Loop --- #
rays = []
x0 = -1e11   # far left
for y0 in np.linspace(-5e10, 5e10, 10):  # vertical spread
    rays.append(Ray((x0, y0), (c, 0.0)))

while not glfw.window_should_close(engine.window):
    engine.run()
    SagA.draw()

    for ray in rays:
        ray.step(1.0, SagA.r_s)
        ray.draw(rays)

    glfw.swap_buffers(engine.window)
    glfw.poll_events()

glfw.terminate()