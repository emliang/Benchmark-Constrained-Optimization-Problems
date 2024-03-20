# Benchmark-Constrained-Optimization-Problems 

## Benchmark of Machine Learning for Constrained Optimization Problems
[toc]




### Continuous cases included:

- [x] Quadratic Programming (QP)
   
$$
\begin{array}{ll}
\min & \frac{1}{2}x^TQx + q^Tx\\
\text {s.t.} & Ax = b\\
& Gx \leq h\\
& lb \leq x \leq ub
\end{array}
$$


- [x] Second-order Cone Programming (SOCP)

$$
\begin{array}{ll}
\min  & \frac{1}{2}x^TQx + q^Tx\\
\text {s.t.} & \vert\vert A_ix+b_i\vert\vert_2 \leq {C_i}^Tx+d_i, \quad i=1,...,m\\
& Fx = g\\
& lb \leq x \leq ub
\end{array}
$$

- [x] Convex Quadratically Constrained Quadratic Program (Convex-QCQP)

$$
\begin{array}{ll}
\min  & \frac{1}{2} x^{\mathrm{T}} P_0 x+q_0^{\mathrm{T}} x \\
\text {s.t.} & \frac{1}{2} x^{\mathrm{T}} P_i x+q_i^{\mathrm{T}} x+r_i \leq 0, \quad i=1,..., m, \\
& A x=b\\
& P_i \succeq 0, \quad i=0,...,m\\
& lb \leq x \leq ub
\end{array}
$$

- [x] Semidefinite Programming (SDP)

$$
\begin{array}{ll}
\min  & \textbf{tr}(CX)\\
\text {s.t.} & \textbf{tr}(A_iX) = b_i, \quad i=1,...,p\\
& X \succeq 0\\
& lb \leq X \leq ub
\end{array}
$$



### Combinatorial cases included:

- [ ] Maximum clique

$$
\begin{array}{ll}
\max\limits_x \; & \sum_{i\in V}x_i \\
   \text{s.t.}\;  & x_i + x_j \le 1,\; \forall (i,j)\in\overline{E}   \nonumber\\
                  &  x_i\in\{0,1\},\; \forall i \in V  \nonumber 
\end{array}
$$

- [ ] Maximum independent set

$$
\begin{array}{ll}
\max\limits_x \;  & \sum_{i\in V} x_i, \; \\
    \text{s.t.}\; & x_i + x_j \le 1,\; \forall (i,j)\in {E}  \nonumber\\
                  & x_i\in\{0,1\},\; \forall i \in V  \nonumber 
\end{array}
$$

- [ ] Maximum  cut
  
$$
\begin{array}{ll}
\max\limits_x & \sum_{(i,j)\in E}\frac{1-x_ix_j}{2}\;, \\
   \text{s.t.}\;  & x_i\in\{-1,1\},\; \forall i \in V \nonumber
\end{array}
$$

- [ ] Travel Salesman Problem
- [ ] Vehicle Routing Problem




### Engineering cases included:

- [ ] AC Optimal Power Flow (OPF)

$$
\begin{array}{ll}
    {\min\limits_{p_{g} , q_{g} , v}} \quad & p_{g}^{\rm T} Q p_{g}+b^{\rm T} p_{g} \\
     \text{s.t.} \quad & p_{g}^{\min } \leq p_{g} \leq p_{g}^{\max },\\
    & q_{g}^{\min } \leq q_{g} \leq q_{g}^{\max }, \\
    & v^{\min } \leq|v| \leq v^{\max }, \\
   & (p_{g}-p_{d})+(q_{g}-q_{d}) i ={\rm diag}(v) \bar{W} \bar{v}, \quad \forall i \in \mathcal{N} \\
   & S_{ij} \le S_{ij}^{\rm max}, \quad \forall (i,j) \in \mathcal{E},\\
\end{array}
$$


- [x] Joint Chance Constraints (JCC) Inventory Management

$$
\begin{array}{ll}
  \min\limits_{x\in\mathbb{R}^n} \quad & {c}^{\mathsf{T}} {x} \\ 
     \text{s.t.} \quad & {\rm \bf Prob}(A {x} \ge \theta + \omega) \ge 1- \delta \\
    & G{x} \le {h},\;  {x}^{\rm min} \le {x} \le {x}^{\rm max},
\end{array}
$$



- [ ] Inverse Kinematics Problem

$$
\begin{array}{ll}
\min\limits_{\alpha\in\mathbb{R}^K} \quad & f(\alpha) \\
\text { s.t. } \quad & \sum_{i=1}^k L_i\cos(\sum_{j=1}^i\alpha_j) =x   \\
                      & \sum_{i=1}^k L_i\sin(\sum_{j=1}^i\alpha_j) =y   \\
                      &  \alpha_{\rm min} \le \alpha \le \alpha_{\rm max}
\end{array}
$$

- [ ] Wireless Power Control

$$
\begin{array}{ll}
\max\limits_{p\in\mathbb{R}^K} \quad & \sum_{i=1}^K \alpha_i \log \left(1+\frac{\left|h_{i i}\right|^2 p_i}{\sum_{j \neq i}\left|h_{i j}\right|^2 p_j+\sigma_i^2}\right) \\
\text { s.t. } \quad & p^{\min } \leq p \leq p^{\max } 
\end{array}
$$

<p align="right">(<a href="#readme-top">back to top</a>)</p>






## Contributing
For any suggestions, please fork the repo and create a pull request.<br>
Any contributions are greatly appreciated.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
