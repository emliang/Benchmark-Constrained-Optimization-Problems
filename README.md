# Benchmark-Constrained-Optimization-Problems

## Benchmark of Machine Learning for Constrained Optimization Problems

### Numerical cases included:

1. Quadratic Programming (QP)
   
$$
\begin{array}{ll}
\text {minimize} & \frac{1}{2}x^TQx + q^Tx\\
\text {subject to} & Ax = b\\
& Gx \leq h\\
& lb \leq x \leq ub
\end{array}
$$


2. Second-order Cone Programming (SOCP)

$$
\begin{array}{ll}
\text {minimize} & \frac{1}{2}x^TQx + q^Tx\\
\text {subject to} & \vert\vert A_ix+b_i\vert\vert_2 \leq {C_i}^Tx+d_i, \quad i=1,...,m\\
& Fx = g\\
& lb \leq x \leq ub
\end{array}
$$

3. Convex Quadratically Constrained Quadratic Program (Convex-QCQP)

$$
\begin{array}{ll}
\text {minimize} & \frac{1}{2} x^{\mathrm{T}} P_0 x+q_0^{\mathrm{T}} x \\
\text {subject to} & \frac{1}{2} x^{\mathrm{T}} P_i x+q_i^{\mathrm{T}} x+r_i \leq 0, \quad i=1,..., m, \\
& A x=b\\
& P_i \succeq 0, \quad i=0,...,m\\
& lb \leq x \leq ub
\end{array}
$$

4. Semidefinite Programming (SDP)

$$
\begin{array}{ll}
\text {minimize} & \textbf{tr}(CX)\\
\text {subject to} & \textbf{tr}(A_iX) = b_i, \quad i=1,...,p\\
& X \succeq 0\\
& lb \leq X \leq ub
\end{array}
$$

5. Joint Chance Constraints (JCC)

$$
\begin{array}{ll}
\text {minimize} & p^Tx\\
\text {subject to} & Ax \geq b+c_i, \quad i=1,...,p\\
& Gx \leq h\\
& lb \leq x \leq ub
\end{array}
$$

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Contributing
For any suggestions, please fork the repo and create a pull request.<br>
Any contributions are greatly appreciated.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
