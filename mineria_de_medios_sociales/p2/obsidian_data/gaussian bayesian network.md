- #bayesiannetwork | #gaussian 

# Concepto
- Son [[bayesian networks|redes bayesianas]] gaussianas en que todas las variables son continuas y definen sus [[decomposition theorem|CPDs]] a través del [[linear gaussian model|modelo lineal gaussiano]].
![[lineal gaussian model-bayesiannetwork.png|400]]

# Modelo condicional lineal gaussiano o CLG
- Algunas variables son continuas y otras discretas.
- Sea $X$ una variable continua y $U=\{U_{1},...,U_{m}\}$ sus padres discretos e $Y=\{Y_{1}, ..., Y_{k}\}$ sus padres continuos. Decimos que $X$ sigue un **CLG** si para cada valor de $u\in Val(U)$ tenemos un conjunto de $k+1$ coeficientes $a_{u,0},...,a_{u,k}$ y una varianza $\sigma_{u}^{2}$ tal que:
$$p(X|u, y) = \mathcal{N}\left(a_{0,u} + \sum_{i=1}^{k} a_{i,y_i}, \sigma^2\right)$$
- Un nodo discreto solo puede tener padres discretos.