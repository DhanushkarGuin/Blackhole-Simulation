import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import math
import numpy as np

from trajectory_generation import TrajectoryNN
from simulation2D import engine, Ray, SagA, c

USE_EVENT_HORIZON_STOP  = True

class NNRay(Ray):
    def __init__(self, pos, direction, nn: TrajectoryNN):
        super().__init__(pos,direction)
        self.nn = nn

    def step(self,rs):
        if USE_EVENT_HORIZON_STOP and self.r <= rs:
            return
        
        state = np.array([self.r, self.phi, self.dr, self.dphi, self.E, self.L], dtype=np.float64)

        r_next, phi_next, dr_next, dphi_next, E_next, L_next = self.nn.predict_next(state)

        self.r = float(r_next)
        self.phi = float(phi_next)
        self.dr = float(dr_next)
        self.dphi = float(dphi_next)
        self.E = float(E_next)
        self.L = float(L_next)

        self.x = self.r * math.cos(self.phi)
        self.y = self.r * math.sin(self.phi)

        self.trail.append((self.x, self.y))

def main():
    nn = TrajectoryNN(model_path='neuralODE_model.keras',
                    scaler_X_path='scaler_X.pkl',
                    scaler_y_path='scaler_y.pkl')
    
    rays = []
    x0 = -1e11
    for y0 in np.linspace(-5e10,5e10,10):
        rays.append(NNRay((x0,y0), (c,0.0), nn))

    import glfw
    while not glfw.window_should_close(engine.window):
        engine.run()
        SagA.draw()

        for ray in rays:
            ray.step(0.01, SagA.r_s)

        if rays:
            rays[0].draw(rays)

        glfw.swap_buffer(engine.window)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    main()