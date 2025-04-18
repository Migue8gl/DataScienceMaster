- #adaline | #ml

## Introducción
- Adaline (ADAptative LInear NEuron) es un algoritmo que simula el comportamiento de una neurona simplificada de forma artificial, al igual que el [[perceptron]].
- La diferencia entre este algoritmo y el perceptron, es que en el segundo se activaba el *output* por medio de una función no diferenciable. Debido a esto Adaline obtiene información del error más exacta y converge más rápido.
- En Adaline, la función que se minimiza es **continua** y diferenciable. Más concretamente se minimiza la función [[sum of squares|MSE]] (Mean Squared Error):
	- $L(w, b)=\frac{1}{N}\sum_{i=1}^{N}(y-\hat{y})^{2}\rightarrow \frac{1}{N}\sum_{i=1}^{N}(y-\sigma(z))^2$
- En Adaline tenemos una función de activación, en este caso la función es la **identidad** y por tanto no hace nada. 

![[adaline-diagram-vs-perceptron.png]]

- Adaline puede utilizar métodos del cálculo para minimizar el error, debido a que se usa una función de pérdida diferenciable. Métodos como [[gradient descent|GD]].

## Actualización
- **Pesos**: $\Delta w=-\eta\bigtriangledown_{w}L(w, b)$
- **Bias**: $\Delta b=-\eta\bigtriangledown_{b}L(w, b)$
- **Actualización**: $w=w+\Delta w$, $b=b+\Delta b$