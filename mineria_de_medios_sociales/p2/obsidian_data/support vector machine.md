- #ml | #svm | #hyperplane
- **Links**:
	- [SVM Scikit Learn](https://scikit-learn.org/1.5/modules/svm.html)
 - #TODO hacer apuntes de kernels y del problema de optimización 
# Introducción
- La intuición detrás de este algoritmo es la de pensar que un hiperplano separador entre clases generalizará mejor si el margen que separa las clases es máximo.
- La tarea principal del algoritmo es maximizar la distancia entre el hiperplano separador y los puntos del espacio más cercanos, llamados [[vector|vectores]] de soporte.
- Es un algoritmo sensible a escalas.
- Las *SVMs* de margen duro o *hard margin* son muy sensibles a [[outliers]].
![[support vector machine-hyperplane-example.png]]
## Soft Margin
- Se introduce la llamada variable *slack*, haciendo referencia a la configuración más o menos permisiva del error que este parámetro configura.
- El *hard margin* asume que los datos son separables, pero en el caso de que no lo sean, las *SVMs* de margen duro no funcionan. Por ello se introduce una predicción más suave en las *SVMs* con el parámetro $C$. Este parámetro controla la penalización en predicciones incorrectas. Básicamente, es una forma de controlar el margen de forma que se admite cierto error en este.
- El objetivo es encontrar un buen equilibrio entre mantener la calle lo más grande posible y limitar las violaciones del margen -> [[svm optimization]]
![[support vector machine-slack-parameter.png]]
## Kernels
- **Linear**: $K(x,x^\prime) = x\cdot x^\prime$
- **Polinómico**: $K(x,x^{\prime}) = (x\cdot x^{\prime}+c)^d$
- **Radial/Gaussiana**: $K(x,x^{\prime}) = exp(\frac{||x-x^{\prime}||^2}{2\sigma^2})$

# Implementaciones
- Como regla general, siempre se debe probar primero el kernel lineal. La clase *LinearSVC* es mucho más rápida que *SVC(kernel="linear")*, especialmente si el conjunto de entrenamiento es muy grande. 
- Esto se debe a que *LinearSVC* utiliza optimizaciones específicas para la clasificación lineal que no son posibles con el uso de un kernel.
- La clase *LinearSVC* se basa en la biblioteca **liblinear**, que implementa un algoritmo optimizado para SVM lineales.
- No admite el truco del kernel, pero se escala casi linealmente con el número de instancias de entrenamiento y el número de características.
- Su complejidad en tiempo de entrenamiento es aproximadamente $O(m×n)$.

# Regresión
- La regresión es posible cambiando los objetivos del algoritmo. En vez de maximizar el margen y minimizar las violaciones de este (instancias dentro del margen), se intenta maximizar el número de instancias dentro del margen y minimizar el número de violaciones (instancias fuera del margen).
- La anchura del margen en este caso de controla mediante el uso de un hiperparámetro $\epsilon$.
- Reducir $\epsilon$ aumenta el número de vectores de soporte, lo que [[regularization|reguraliza]] el modelo.