# Joint-Space "Extended" Kalman Filter

This document goes over the joint-space extended Kalman filter that is used to output both a $\Sigma_\mathbf{q}$ and $\Sigma_\dot{\mathbf{q}}$ to be fed into the main invariant filter. Here, we'll go through each part of the filter and to which part of the file structure it corresponds to.

## Process Model (Prediction Step)

We define the state as the joint positions, velocities, and gyro biases:

$$
\hbf{x} = \begin{bmatrix} \mathbf{q} \\ \dbf{q} \\ \mathbf{b}_\boldsymbol{\omega} \end{bmatrix} \in \mathbb{R}^{2n+3m}
$$ (eq:state)

where $n$ is the number of 1 DoF joints, and $m$ is the number of IMUs around the robot as a whole (excluding the base/pelvis IMU).

From this, we define the process dynamics to simply be:

$$
\begin{aligned}
\dbf{q} &= \dbf{q} \\
\ddot{\mathbf{q}} &= \mathbf{w}_a \\
\dbf{b}_\boldsymbol{\omega} &= \mathbf{w}_b
\end{aligned}
$$ (eq:process)

where $\mathbf{w}_a \sim \mathcal{N}(0,Q_a)$ and $\mathbf{w}_b \sim \mathcal{N}(0,Q_b)$ are both zero-mean, additive white Gaussian noise processes.

This leads to a nilpotent $A$ matrix (i.e. $A^2=0$) in state space form:

$$
A = \begin{bmatrix} 0 & I_n & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}
$$ (eq: a_mat)

And because this is a nilpotent matrix, this means that we can discretize it using an approximation that ends up being exact to first order, i.e. $e^{A \Delta t}=I+A \cdot \Delta t$:

$$
F = e^{A \Delta t}=\begin{bmatrix} I_n & I_n \cdot \Delta t & 0 \\ 0 & I_n & 0 \\ 0 & 0 & I_{3m} \end{bmatrix}
$$ (eq: f_mat)

And for the process noise $\mathbf{w}_a$, we can obtain the discrete form (purely for joint coupling) as the following:

$$
Q_{d}^{qq}=\begin{bmatrix}  \frac{\Delta t^3}{3}Q_a && \frac{\Delta t^2}{2} Q_a \\ \frac{\Delta t^3}{3}Q_a && \Delta t Q_a \end{bmatrix}
$$ (eq: joint_proc_noise)

Similarly, for the biases, the process noise is strictly diagonal, so this means that:

$$
Q_{d}^{bb} = \sigma_b^2 \Delta t I_{3m}
$$ (eq: bias_proc_noise)

Where we specify that since the IMUs are already being pre-filtered by a Mahony filter, that the constant $\sigma_{bb} \ll \sigma_\mathrm{IMU}$, otherwise we can risk large convergence times.

So the full prediction step in state form is as follows:

$$
\begin{gathered}
\hbf{x}_{k+1|k} = F\hbf{x}_{k|k} \\
P_{k+1|k} = F P_{k|k} F^\top + Q_d
\end{gathered}
$$
