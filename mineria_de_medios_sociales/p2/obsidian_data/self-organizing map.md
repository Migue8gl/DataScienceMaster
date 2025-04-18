- #neuralnetworks 

# Concepto
- Es un algoritmo de aprendizaje no supervisado usado para codificar datos multidimensionales en bajas dimensiones (normalmente dos) preservando la estructura topológica.
- Los mapas autoorganizativos de *Kohosen* o **SOM** no hacen suposiciones sobre las distribuciones de las variables ni requieren independencia. Manejan muy bien datos ruidosos, faltantes y muestras de tamaño limitado.
![[self-organizing map-example.png|500]]
- Cuando llega una nueva señal de entrada, las neuronas *compiten* entre ellas para representarla.
- La unidad que mejor se ajusta es la unidad que gana la competición y en conjunto con los vecinos de la rejilla aprenden la señal. Los vecindarios de neuronas terminan especializándose para representar señales parecidas.

# Funcionamiento
- Se calcula la unidad que mejor se ajusta a un vector de entrada.
$$||x-w_{c}||=min_{i}||x-w_{i}||$$
- Donde se utiliza la métrica de distancia pertinente, normalmente la [[distances|euclídea]].
- Se actualiza la unidad ganadora y los vecinos más cercanos. La magnitud de dicha actualización está regida por la tasa de aprendizaje:
$$W_{j}(t+1)=W_{j}(t)+\theta(k,j,t)\alpha(t)(X(t)-W_{j})$$
- Donde $\theta(k,j,t)$ es a vecindad de la neurona $k$ a la $k$. Si es próxima vale $1$, si no es así, vale $0$. $\alpha(t)$ es la razón de aprendizaje.
- Según se entrena, se decrece la tasa de aprendizaje y la vecindad.
![[self-organizing map-example2.png]]

# Usos
- *Clustering*, reducción de dimensionalidad, visualizaciones, etc.