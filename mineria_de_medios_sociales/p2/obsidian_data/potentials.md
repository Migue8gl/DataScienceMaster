- #probability | #potential

# Concepto
- Puede ocurrir que los cálculos probabilísticos no generen distribuciones normalizadas. Por ello se habla de **potenciales**. Un potencial $\phi$ es una función de valores reales definido de la siguiente forma:
$$\phi=sp(X)\rightarrow \mathbb{R}$$
- Recordamos que $sp(X)$ denota al conjunto formado por todas las combinaciones de posibles valores de las variables contenidas en $X$. Por otra parte, el dominio de un potencial se denota mediante $dom$.
# Características
- Los potenciales pueden multiplicarse. Esta operación presenta las siguientes propiedades:
$$\begin{align*} 1.&\, \text{dom}(\phi_1 \phi_2) = \text{dom}(\phi_1) \cup \text{dom}(\phi_2) \\ 2.&\, \phi_1 \phi_2 = \phi_2 \phi_1 \\ 3.&\, (\phi_1 \phi_2)\phi_3 = \phi_1(\phi_2 \phi_3) \\ 4.&\, \text{existencia de potencial unidad: sus valores son todos igual a 1}. \\ &\, \text{Se representa como } \phi_i \text{ y se cumple que actúa como el elemento} \\ &\, \text{neutro: } \phi \phi = \phi \end{align*}$$
- El operador de marginalización puede aplicarse a potenciales, de forma que $\sum_{A}\phi$ es un potencial definido sobre $dom(\phi)\backslash A$. La marginalización también es conmutativa.
$$\sum_{A}\sum_{B}\phi=\sum_{B}\sum_{A}\phi$$

# Operaciones
## Combinación
- SI $p(x,y)$ y $q(y,z)$ dos dos potenciales sobre $(X,Y)$ y  $(Y,Z)$ respectivamente, entonces su **combinación** es el potencial $p\cdot q(x,y,z)$ sobre $(X,Y,Z)$ dado por $p\cdot q(x,y,z)=p(x,y)\cdot q(y,z)$.
![[potentials-combination.png]]
## Marginalización
- Si tenemos un conjunto de variables $Y=(X,Z)$, entonces la **marginalización** permite obtener la distribución de probabilidad sobre $X$ a partir de $Y$.
- Si $p(x,z)$ es un potencial sobre $(X,Z)$ entonces su marginalización sobre $X$ es:
	- $p(x)=\sum_{z}p(x,z)$
- También se conoce como **borrado** de $Z$.
![[potentials-marginalization.png]]