- #decompositionthorem | #statistics 

# Concepto
- Dada una [[bayesian networks|red bayesiana]] con variables $X$ entonces la distribución de probabilidad conjunta de estas variables se puede descomponer de la forma:
$$\large{p(x)=\prod_{y\in x}p(y|pa(Y))}$$
- Donde $pa(Y)$ es el conjunto de padre de la variables $Y$.
## Consecuencia
- Para especificar una red bayesiana solo hay que dar, para cada variable de $Y$, una distribución de probabilidad condicionada (**CPD**) dados sus padres $pa(Y)$.
$$\sum_{y\in U_{y}}p(y|pa(Y))=1$$
- Si la variable es raíz, la distribución será marginal (sin condicionar ya que no tiene padres).
![[bayesian networks construction-decomposition-theorem.png]]
## CPD
- Una **CPD** $P(X|Y_{1},...,Y_{k})$ especifica una distribución sobre $X$ para cada asignación $y_{1},...,y_{k}$.
- Para especificar una **CPD**, podemos usar cualquier función $\phi(X, Y_{1},...,Y_{k})$ que verifique: 
	- $\phi(x,y_{1},...,y_{k})=P(x|y_{1},...,y_{k})\forall x,y_{1},...,y_{k}$
- Cualquier **GPD** discreta se puede representar mediante tablas.
- Decimos que una **CPD** para $X$ es una función determinista de sus padres $pa(X)$, cuando existe una función $f:U_{pa(X)}\rightarrow U_{X}$ tal que:
$$P(x \mid pa(X)) = \begin{cases} 
1, & \text{si } x = f(pa(X)) \\
0, & \text{en otro caso}
\end{cases}$$
![[bayesian networks construction-cpd-function-example.png]]
- Se pueden obtener más independencias (de las proporcionadas por la [[d-separation|d-separación]]) de la red bayesiana cuando tenemos [[deterministic node|nodos deterministas]]. Si conocemos el valor de $A$ y $B$, conocen eremos el valor de $C$ al ser este un [[deterministic node|nodo determinista.]] Por tanto, $D$ y $E$ se hacen independientes al conocer $A,B$ ($I(D,E|A,B)$).
![[bayesian networks construction-cpd-independencies-example.png|200]]
### Potenciales
- Los [[potentials|potenciales]] son probabilidades condicionales que contienen:
	- Un conjunto de variables $Y$.
	- Una aplicación de $U_{y}$ en el conjunto de los reales.
- Son un ejemplo de **GPD**.