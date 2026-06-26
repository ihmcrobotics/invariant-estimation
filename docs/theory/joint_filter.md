# Joint-Space "Extended" Kalman Filter

This document goes over the joint-space extended Kalman filter that is used to output both a $\Sigma_\mathbf{q}$ and $\Sigma_\dot{\mathbf{q}}$ to be fed into the main invariant filter. Here, we'll go through each part of the filter and to which part of the file structure it corresponds to.

## Process Model (Prediction Step)

<!--NOTE: need to add sub-sections to denote different parts of the process model.-->

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

Because this state propagates acceleration and not torques, we have to model the process noise $\mathbf{w}_a$ as the unmodeled portion of the torque acting on the system. This means that we append the definition of $\mathbf{w}_a$ above:

$$
\mathbf{w}_a=M(\mathbf{q})^{-1}\mathbf{w}_\boldsymbol{\tau}
$$ (eq:torque_accel_rel)

And since we know the general identity that if $\mathbf{y}=L\mathbf{x}$, then $\Sigma_y=L\Sigma_x L^\top$, we apply this relation to $\mathbf{w}_a$ to obtain $Q_a$:

$$
Q_A = M(\mathbf{q})^{-1}\Sigma_\boldsymbol{\tau}M(\mathbf{q})^{-\top}
$$ (eq:accel_proc_noise)

Furthermore, since $M(\mathbf{q})$ is a symmetric matrix, we know by construction that $M(\mathbf{q})^{-\top}=M(\mathbf{q})^{-1}$. And if we also assume that $\Sigma_\boldsymbol{\tau}$ is a scalar, i.e. $\Sigma_\boldsymbol{\tau}=\sigma_\boldsymbol{\tau}^2I$, then we an simplify $Q_a$ to be:

$$
Q_a = \sigma_\boldsymbol{\tau}^2 M(\mathbf{q})^{-2}
$$

Even though the process noise is state-dependent, the construction of the process model itself leads to a nilpotent $A$ matrix (i.e. $A^2=0$) in state space form:

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
\boxed{
\begin{gathered}
\hbf{x}_{k+1|k} = F\hbf{x}_{k|k} \\
P_{k+1|k} = F P_{k|k} F^\top + Q_d
\end{gathered}
}
$$ (eq: full_state_pred)

## Measurement Model

### Joint Encoders

The first of our measurements is that of the joint encoders, which we treat as a simple direct measurement corrupted by sensor noise:

$$
\tbf{q} = I_n \mathbf{q} +\mathbf{v}_q
$$ (eq: meas_encoders)

Where $\mathbf{v}_q \sim \mathcal{N}(0,R_\mathbf{q})$, and $R_\mathbf{q}=\sigma_\mathbf{q}^2 I_n$ is the noise covariance. We treat this as fully diagonal as the sensors themselves are fully independent, while the kinematics are not.

### Distributed IMUs

This is the key part of the filter that makes it somewhat nonlinear. For each pair of IMUs, we define the kinematic coupling:

$$
\mathbf{z}_{\boldsymbol{\omega},ab} = J_{b}^{b,a}(\hbf{q})S_{ab}\dbf{q}+\mathbf{b}_{\boldsymbol{\omega},ab}+\mathbf{v}_{\boldsymbol{\omega},ab}
$$ (eq: imus)

where $J_{b}^{b,a}(\hbf{q})$ is the kinematic Jacobian of the joints between IMU $a$ and $b$, and $S_{ab}$ is the relevant selector matrix acting on the joint velocity $\dbf{q}$.

### Combined Measurement Vector $\mathbf{z}$

To fuse the measurements into a single vector, we combine them as the following:

$$
\mathbf{z} = \begin{bmatrix} \tbf{q} \\ \boldsymbol{\omega}_{b}^{b,a}\end{bmatrix} = \begin{bmatrix}I_n & 0 & 0 \\ 0 & J_b^{b,a}(\hbf{q})S_{ab} & I_3 \end{bmatrix}\hbf{x}+\mathbf{v}
$$ (eq: full_measurement)

where $\mathbf{v}\sim \mathcal{N}(0,R)$ is the additive white Gaussian noise on the sensors, with <br/> $R=\mathrm{blkdiag}(\sigma_\mathrm{enc}^2 I_n, R_\boldsymbol{\omega})$. $R_\boldsymbol{\omega}$ is separate to account for the covariances of the individual IMU Mahony filters fused together.

## Full Filter Equations

At every time step $k$, the filter runs the following:

$$
\begin{gathered}
\textbf{Predict:} & \hbf{x}_{k+1|k} = F\hbf{x}_{k|k} \\
& P_{k+1|k} = FP_{k|k}F^\top + Q_d(M(\mathbf{q})) \\
\textbf{Update:} & \hbf{x}_{k+1|k+1} = \hbf{x}_{k+1|k} + K(\mathbf{z}-H\hbf{x}_{k+1|k}) \\
& P_{k+1|k+1} = (I-KH)P_{k+1|k}(I-KH)^{\top} \\
& K=P_{k+1|k}H^\top(HP_{k+1|k}H^\top+R)^{-1}
\end{gathered}
$$
