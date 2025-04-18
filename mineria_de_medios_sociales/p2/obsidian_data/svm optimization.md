- #svm | #optimization | #dualproblem

# Hard margin
- La idea principal es la de minimizar los pesos aprendidos por el modelo, ya que a más pequeños sean, mayor será el margen:
![[dual problem svm-little-weights.png]]
- Además, también se quieren evitar las violaciones del margen, por lo que se necesita que la [[perceptron#^da9b29|la función de decisión]] sea mayor que $1$ para instancias positivas y menor que $-1$ para instancias negativas.
- Se define pues $t(i)=-1$ cuado $y(i)=0$ y $t(i)=1$ para cuando $y(i)=1$. Esta restricción se puede representar como: $t(i)\cdot (w^{T}x(i)+b)\ge1$ para todas las instancias.
- El problema de optimización restringido es:
 $$ \begin{align*} \min_{w,b} \quad & \frac{1}{2} w^T w \ \hspace{2mm}\text{sujeto a} \quad & t^{(i)}(w^T x^{(i)} + b) \geq 1, \quad \text{para } i = 1,2,\ldots,m \end{align*} $$

# Soft margin
- En la práctica, los problemas no suelen ser perfectamente separables, por lo que la restricción anterior no se cumpliría nunca. Por ello se introduce un término de *holgura* $\zeta(i)\ge 0$. Con esa variable se mide cuanto se permite una violación de la instancia $i$ en el margen. Por ello, la nueva fórmula de optimización es:
$$ \begin{align*} \min_{w,b,\zeta} \quad & \frac{1}{2}w^T w + C \sum_{i=1}^m \zeta^{(i)} \ \text{sujeto a} \quad & t^{(i)}(w^T x^{(i)} + b) \geq 1 - \zeta^{(i)} \quad \text{y} \quad \zeta^{(i)} \geq 0 \quad \text{para } i = 1,2,\ldots,m \end{align*} $$