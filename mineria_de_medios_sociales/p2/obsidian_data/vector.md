- #vector

### Definición
- Un vector es una flecha en el espacio, con una longitud y una dirección específica. 
- En física se piensa como una magnitud y una dirección, en matemáticas como una escala y una dirección.
- Su función es representar datos de manera visual.
- En el [[cartesian plane|plano cartesiano]] podríamos pensar en las coordenadas originales, las cuales son $(0,0)$. Estas coordenadas son el origen de todos nuestros vectores en esta representación específica.

![[vector-2d-example.png|400]]

- Este vector representaría un paso en el eje $x$ y dos pasos en el eje $y$.
- Estos pueden existir en tantas dimensiones como se requieran. Ejemplo en $3$ dimensiones:
	- $\vec{v}=\begin{bmatrix}x\\ y\\ z\end{bmatrix}$ 

![[vector-3d-example.png]]

### Suma
- $\vec{v}=\begin{bmatrix}x\\ y\\\end{bmatrix}$
- $\vec{u}=\begin{bmatrix}z\\ t\\\end{bmatrix}$
- $\vec{v}+\vec{u}=\begin{bmatrix}x+z\\ y+t\\\end{bmatrix}$
- La suma es la combinación de vectores de forma que, visualmente, estamos moviéndonos en el espacio en la dirección de uno de los vectores y con su longitud, y luego, moviéndonos en la dirección del otro vector y con su otra longitud.
- La suma es conmutativa, de forma que da igual si sumamos antes un vector que otro, el resultado es el mismo.

![[vector-addition-example.png]]

### Escalado
- El escalado trata de la ampliación/disminución de la longitud de nuestro vector.
- Se consigue multiplicando el vector por un valor, conocido como **escalar**.
- $\vec{v}=\begin{bmatrix}x\\ y\\\end{bmatrix}$
- $a=2$
- $a\vec{v}=\begin{bmatrix}a·x\\ a·y\\\end{bmatrix}$
- Escalar un vector no cambia su dirección, a menos que el **escalar** sea un número negativo.

### Span y Dependencia lineal
- El *span* o espacio generado es un conjunto de todas las combinaciones lineales de vectores pertenecientes al conjunto $S$. 
- Las dos operaciones, escalado y adición, nos permiten obtener cualquier otro vector.
- Cuando tenemos **dos** vectores con direcciones distintas, entonces esos vectores son [[linear dependence|linealmente independientes]]

![[lineal-combination-3d-example.png|500]]