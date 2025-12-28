For more details, see https://medium.com/@waadlingaadil/learn-to-build-a-neural-network-from-scratch-yes-really-cac4ca457efc

---

Layer values: $A_{0 \to \text{NB_LAYERS}}$

$A_l \in \mathcal{M}_{n_l, 1}$

Weights: $W_{1 \to \text{NB_LAYERS}}$

$W_l \in \mathcal{M}_{n_l, n_{l - 1}}$

Biases: $B_{1 \to \text{NB_LAYERS}}$

$B_l \in \mathcal{M}_{n_l, 1}$

---

$g(z) := \frac{1}{1 + e^{-z}}$

$G(Z) :=$ $Z$.map($g$)

$Z_l := W_l A_{l - 1} + B_l$

$A_l = G(Z_l)$

$L := \text{NB_LAYERS}$

---

Expected (= real) result: $y$

Guessed result: $\hat y = A_L$

Cost: $C := -(y \ln \hat y + (1 - y) \ln (1 - \hat y))$

---

$g'(z) = g(z) \cdot (1 - g(z))$

---

$\frac{\partial C}{\partial A_L} = \frac{1 - y}{1 - A_L} - \frac{y}{A_L}$

$\frac{\partial C}{\partial Z_l} = \frac{\partial C}{\partial A_l} \times (\mathbb{1}_{i = j} \cdot A_l(i) \cdot (1 - A_l(i)))_{i, j}$

> $\frac{\partial A_l}{\partial Z_l} = (\mathbb{1}_{i = j} \cdot A_l(i) \cdot (1 - A_l(i)))_{i, j}$

$\frac{\partial C}{\partial A_{l - 1}} = \frac{\partial C}{\partial Z_l} \times W_l$

> $\frac{\partial Z_l}{\partial A_{l - 1}} = W_l$

---

$\frac{\partial C}{\partial W_l} = A_l \times \frac{\partial C}{\partial Z_l}$

> $\frac{\partial Z_l^T}{\partial W_l^T} = A_l^T$

---

$\frac{\partial C}{\partial B_l} = \frac{\partial C}{\partial Z_l}$

> $\frac{\partial Z_l}{\partial B_l} = I_{n_l}$
