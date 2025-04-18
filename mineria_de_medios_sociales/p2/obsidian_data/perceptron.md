- #perceptron | #ml

## Introducción
- El perceptron es una neurona artificial inspirada por el comportamiento de sus análogas biológicas. El primer concepto de una célula cerebral simplificada viene del paper [A logical calculus of the ideas immanent in nervous activity](https://link.springer.com/article/10.1007/BF02478259) en $1943$. Este paper mostraba una neurona como una simple puerta lógica con *outputs* binarios.
- Si la señal acumulada de los *inputs*  en las *dendritas* excedía cierto límite, entonces el *output* es generado y pasado al *axón*.
- En $1957$ con [The Perceptron — A Perceiving and Recognizing Automaton](https://websites.umass.edu/brain-wars/1957-the-birth-of-cognitive-science/the-perceptron-a-perceiving-and-recognizing-automaton/) se formaliza un algoritmo inspirado en este comportamiento con la principal característica de poder "aprender".

## Definición formal
- Se tiene dos clases, pues es un clasificador binario. 
- Se define una función de decisión $\sigma(z)$ que combina linealmente los *inputs* con un [[vector]] de pesos $w$. Esta combinación lineal $z$ se conoce como **net input**: 
	- $z = w_1x_{1}+w_{2}x_2+...w_mx_m$
	- $\begin{equation}\sigma(z)=\begin{cases}1 & \text{if z} \ge \theta \\ 0 & \text{otherwise}\end{cases}\end{equation}$
- Pare simplificar la implementación en código:
	- $z\ge\theta \hspace{3mm}\rightarrow\hspace{3mm} z-\theta\ge 0$
- Definiendo entonces  el **bias** como:
	- $b=-\theta$
- Así se puede hacer parte del **net input** y hacer de él un parámetro aprendible: ^da9b29
	- $z=w^Tx+b$
	- También se representa como $w_0$.
	- $\begin{equation}\sigma(z)=\begin{cases}1 & \text{if z} \ge 0 \\ 0 & \text{otherwise}\end{cases}\end{equation}$

## Regla de actualización
- El perceptron se puede resumir en los siguientes pasos:
	- Se inicializan los pesos y la unidad de sesgo o **bias** a pequeños valores aleatorios.
	- Por cada ejemplo de entrenamiento $x_i$:
		- Se calcula $\hat{y}_i$ (etiqueta predicha).
		- Se actualizan pesos y sesgo.
- Las actualizaciones son las siguientes:
	- $w_j=w_{j}+\Delta w_j$
	- $b=b+\Delta b$
- Donde los **deltas** son:
	- $\Delta w_j=\eta(y_i-\hat{y}_i)x_{ij}$
	- $\Delta b=\eta(y_i-\hat{y}_i)$
- Donde $\eta\in(0,1)$ es el *learning rate*, el cual determina cómo de rápido converge el algoritmo, pues es un escalar que hará que los pasos hacia la solución sean más o menos grandes.

![[perceptron-algorithm-diagram.png]]

- La convergencia del algoritmo solo está garantizada si el problema es **linealmente separable**.

## Tip
- No se suelen inicializar los pesos y el sesgo a cero debido a que si esto fuese así, el *learning rate* solo afectaría a la escala y no a la dirección. Esto significa, que si los pesos son cero, el *learning rate* al no afectar a la dirección, no afecta al signo y por ello, al depender el *output*  de la función $\sigma(z)$  únicamente del signo, entonces $\eta$ no estaría añadiendo información en la decisión final y sería irrelevante su uso.
- [¿Por qué NO inicializar los pesos a cero?](https://datascience.stackexchange.com/questions/26134/initialize-perceptron-weights-with-zero)