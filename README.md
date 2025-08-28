### Still working on this project, probably a normal project trying to implement Machine Learning later on!

Inspiration for this project, [YouTube](https://youtu.be/8-B6ryuBkCM?si=-pYplb0DPwtRbTXP)

Using the above youtube video as a guide but actually implementing on Python.

Hope so I actually complete this project, readme will be updated when things are implemented later on!

Project is completed readme will be updated further with the explanations and installation steps.

### [Updated]

# Black Hole Simulation with Neural ODEs

A hybrid physics–AI project that simulates light ray trajectories around a black hole using both:

- Traditional physics (RK4 integrator)

- Neural ODE approximation (trained with TensorFlow)

This project explores how Neural Networks can accelerate costly differential equation simulations, making it possible to scale from 10 rays to thousands of rays efficiently.

## 📌 Features

- 2D simulation of light rays around a Schwarzschild black hole

- Physics engine using Runge–Kutta 4th order (RK4) integration

- Neural ODE trained to approximate geodesic equations

- Real-time comparison: RK4 vs Neural ODE

- Visualization with OpenGL / PyOpenGL

## 📂 Project Structure

Not in order

```
├── final_simulation.py     # Finalized Simulation using Neural ODE
├── simulation2D.py         # Physics + visualization engine (initial)
├── neuralODE.py            # Neural ODE training & model definition
├── testing_simulation.py   # Checking trajectory generation before the actual simulation
├── neuralODE_model.keras   # Saved models
├── scaler_X.pkl            # Scaler for X (inputs)
├── scaler_y.pkl            # Scaler for y (outputs)
├── dataset.py              # Dataset creation
├── ray_dataset.csv         # Training Dataset
├── requirements.txt        # Dependencies for project
├── trajectory_generation.py# Class and function for final simulation
└── README.md               # Documentation
```

## ⚖️ Mathematical Background
Without Neural ODE

- Equations: 6 coupled differential equations (geodesics)

- Integration method: RK4

- Steps per ray: ~1000

- Cost per ray: ~140k FLOPs

With Neural ODE

- Neural net architecture: Dense NN, ~11k FLOPs per step

- Steps per ray: ~1000

- Cost per ray: ~11M FLOPs (but batched efficiently on GPU)

Even though original rk4 is cheaper for simulation but it is not efficient in real-life application where Neural ODE will be faster.

RK4 produces slow results on single core CPU compared to Neural ODE with GPU support.

## ⚙️ Installation

- Install Dependencies

```
pip install -r requirements.txt
```

- Run the files as usual

```
python <filename>.py
```

## 🚀 Physics

- Refer [YouTube](https://youtu.be/8-B6ryuBkCM?si=-pYplb0DPwtRbTXP) for explanation of physics implemeted in the project.
