# ABOUT CAR MODELS

Yeah so it just so happens that we need to be able to model the car's motion for a metric shitton of stuff.
MPC, Kalman, SLAM, a lot of the pipeline depend brutally on an accuarate car mathematical model, this is, a way to predict the state of the car given the current state an a series of control inputs.

These models are implemented using CasADi, a sick ass symbolic library that allows the user (you) to define equations mathematically and then export them as optimized C code


## Modelling Overview
For the chuds who dont control this shyt, here's a rundown:

A model can be summarized as mathematically transfer function with the signature 
\[
    F: X, U \rightarrow X
\]
Where \(X\) is a vector expressing the **state** of a system, and \(U\) a vector expressing the **control inputs** being applied to the system  

Normally, a model just specifies \(\dot{X}\), this is, \(\frac{\partial X}{dt}\), and a discrete transfer function is constructed by either Forward or Backward Euler, which is described below

### State Representation
The vehicle state is represented by the vector:

\[
\mathbf{X} =
\begin{bmatrix}
x \\ y \\ \varphi\\ u \\ v \\ \omega
\end{bmatrix}
\]

where:
- \(x, y\): Global position of the vehicle
- \(\varphi\): Heading angle (yaw)
- \(u, v\): Velocity components in the body frame (longitudinal & lateral)
- \(\omega\): Yaw rate (angular velocity)

### Control Inputs
The control input vector incorporates the time step \( dt \) explicitly:

\[
\mathbf{u} =
\begin{bmatrix}
a \\ \delta_f \\ \Delta t
\end{bmatrix}
\]

where:
- \(a\): Longitudinal acceleration (gas)
- \(\delta_f\): Front wheel steering angle
- \(\Delta t\): Time step (discretization interval)



## About Stability
Math is weird an for some reason it kinda doesnt always like when you discretize continous systems. This is a big fucking issue if we dont want our models to spontanously combust into a particularly fast mountain of trash.

Stability means that the model will not diverge from reality and enter a spiraling loop of absurd self-sustained growth

For this, we need to talk about basic math and about my #3 side hoe Euler

### Asymptotes
Ok i feel this is pretty fucking ovbious and if you feel the need to read any further in this subsection you should reconsider wether you are ready to work on models

- Dont put asymptotes in your models

Yeah thats about it. If there are asympotes in a given model, understand the physical meaning of those asymptotes and neutralize them appropiately with one of the 400329 methods available

### Euler
When discretizing, there the problem turns to:
Given \(X\) at k and \(U\) at k, what is \(X) at k+1

How this is approached affects stability a LOT, like a BIG FUCKING LOT
- Forward Euler:
\[X_{k+1} = X_k + \Delta t \ \cdot \ \dot{X}_k \]
This method is NOT unconditionally stable
It can be proven that for \(\Delta t\) small enough, it becomes stable. However, using it is risky since everything can you to shit if the system for some reason lags behind and needs to do bigger steps.
However, it is easy as fucc to implement

and his evil brother: 

- Backward Euler:
\[X_{k+1} - \Delta t \ \cdot \ \dot{X}_{k+1} = X_k  \]
Yeah basically the opposite of the Fordward one and the preferred one in most cases
Is is harder to compute, needing to solve a root finding problem but it is unconditionally stable which means, naturally, as the ancient chinese wisdom goes:
    - ***Sarandonga, Sarandonga, Cuchibiri Cuchibiri***
        - Confuncious, probably (???)*


# Bibliography 
My descriptions do NOT make justice to the models they're based on. Check the sources please please i beg pretty please PLEASE
- [Chinese paper idfk.](https://arxiv.org/abs/2011.09612) Explains barebone Dynamic and Kinematic Bicycle mode
- [Korena paper cause fuck the west.](https://www.researchgate.net/publication/357040239_Game-Based_Lateral_and_Longitudinal_Coupling_Control_for_Autonomous_Vehicle_Trajectory_Tracking) This one is "*based as fucc*" and "*swaggg moneyyy*" according to numerous sources[^[1]^](https://en.wikipedia.org/wiki/Bullshit). It directly adresses some models in MPC's and other controllers