# PIGNN + HELM Known-Operator Learning

## Decision

HELM is meaningful as a known operator, but the useful integration is **not**

```text
PIGNN voltage -> HELM solve
```

Canonical HELM does not consume a voltage warm start. It constructs its own
germ, recursively solves for power-series coefficients, and evaluates the
series using analytic continuation. Attaching it after PIGNN would make the
PIGNN output irrelevant and would again be two sequential solvers.

The implemented model instead lets PIGNN control an endpoint-preserving
holomorphic embedding path. The exact AC equations, coefficient recursion,
linear solves, fixed bus constraints, and Pade evaluation remain known
operators.

## What HELMpy actually provides

The local HELMpy implementation:

- represents transmission lines, taps, phase shifters, shunts, PV/PQ/slack
  buses, generator Q-limit switching, and optional distributed slack;
- factorizes a constant sparse real matrix;
- recursively computes voltage-series coefficients;
- evaluates even-order diagonal Pade approximants at the physical endpoint;
- stops when consecutive Pade voltage magnitudes and angles change by less
  than its `mismatch` setting.

The last point is critical. HELMpy's stopping test is not the masked AC
active/reactive-power residual used by the PF corpus. Its message that reaching
the coefficient limit means the problem has no physical solution is also too
strong for a research claim: a particular finite-order analytic continuation
can fail numerically without proving non-existence.

HELMpy is NumPy/SciPy code, operates on case objects constructed from XLSX, is
not batched, and has no PyTorch autograd path. It is licensed under AGPLv3.
The training implementation therefore does not import or copy HELMpy code; it
implements the equation-level embedding in PyTorch.

## Endpoint-preserving learned embedding

Let the rectangular voltage state be

\[
x=[v_r^\top,v_i^\top]^\top.
\]

Define the known equation map `h(x)` bus-wise as

\[
h_i(x)=
\begin{cases}
(v_{r,i},v_{i,i}), & i\in\mathcal S,\\
(P_i(x),v_{r,i}^2+v_{i,i}^2), & i\in\mathcal V,\\
(P_i(x),Q_i(x)), & i\in\mathcal Q,
\end{cases}
\]

where \(\mathcal S,\mathcal V,\mathcal Q\) are the slack, PV and PQ sets and

\[
S(x)=V\odot\overline{YV}.
\]

The target vector `b` contains the fixed slack voltage, PV active power and
voltage magnitude squared, and PQ active/reactive powers.

A canonical germ \(x_0\) is constructed with the specified slack voltage,
specified PV magnitudes at zero angle, and \(1+j0\) at PQ buses. Its implied
equation vector is \(b_0=h(x_0)\).

PIGNN predicts path weights

\[
w_{i,m}=\operatorname{softmax}_m z_{i,m},\qquad
\sum_{m=1}^{M}w_{i,m}=1.
\]

The embedding is

\[
h(x(s))=b_0+\sum_{m=1}^{M}(b-b_0)\odot w_m s^m.
\]

At the physical endpoint,

\[
h(x(1))=b_0+(b-b_0)\odot\sum_m w_m=b.
\]

Consequently, learning changes the analytic path but cannot change the PF
problem at \(s=1\).

## Known coefficient recursion

Expand

\[
x(s)=x_0+\sum_{n=1}^{K}x_n s^n.
\]

Because the rectangular AC equations are quadratic, the coefficient at order
\(n\) has the form

\[
A x_n+r_n(x_1,\ldots,x_{n-1})=c_n,
\qquad A=J_h(x_0).
\]

The same matrix \(A\) is used for every order and every scenario sharing the
same grid and fixed setpoints. It is LU-factorized once and cached. The known
operator then computes

\[
x_n=A^{-1}(c_n-r_n).
\]

For example, with current coefficients \(I_n=YV_n\), the nonlinear power
remainder is

\[
R^S_n=\sum_{k=1}^{n-1}V_k\odot\overline{I_{n-k}}.
\]

PV magnitude equations use the corresponding convolution

\[
R^{|V|^2}_n=\sum_{k=1}^{n-1}
(v_{r,k}v_{r,n-k}+v_{i,k}v_{i,n-k}).
\]

The endpoint voltage is evaluated with a differentiable diagonal Pade
approximant. During evaluation the implementation examines the available even
orders and retains the one with the smallest actual masked P/Q residual.

## Why this is one hybrid solver

PIGNN is not predicting an initial voltage for an independent solver. Its
outputs are parameters of the HELM embedding itself:

\[
z_\phi\rightarrow w_\phi\rightarrow
\{c_n\}\rightarrow\{x_n\}\rightarrow
\operatorname{Pade}(x_0,\ldots,x_K).
\]

Gradients propagate through softmax, all coefficient right-hand sides, cached
LU solves with respect to their right-hand sides, and Pade evaluation back to
the edge-aware PIGNN controller.

## Claim boundary

The endpoint identity belongs to the untruncated embedding. A finite series and
finite Pade approximant need not solve the endpoint exactly. Therefore:

- report the actual masked P/Q residual, not only Pade-to-Pade change;
- call a sample feasible/certified only when its residual is below tolerance;
- never interpret coefficient-limit failure alone as proof that no physical
  solution exists;
- compare against standard HELM, Newton-Raphson, raw PIGNN, and the integrated
  DPF-preconditioned PIGNN;
- report Pade order, failure rate, wall time, factorization time, and peak
  memory.

## Pilot configuration

- HELM series order: 40. Order 8 was rejected because truncation residual
  remained about 5e-2 pu on case24_ieee_rts; on two case300 samples, float64
  order 40 reduced the uniform-path residual to 3.2e-5 and 3.8e-7 pu.
- learned path order: 3
- PIGNN controller: hidden 24, 8 attention heads, 8 layers
- loss: supervised voltage MSE plus log-cosh physical residual
- evaluation: best actual-residual Pade candidate among even orders 4 through 40
- splits/seed: corpus-standard one-third/one-third/one-third, seed 42
- grids: case24_ieee_rts, case57, case89pegase, case145, case300,
  case1354pegase

Case145 remains a data-quality caveat because it contains only 7,885 scenarios.
