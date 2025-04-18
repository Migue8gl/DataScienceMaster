- #multi-instancelearning | #ml

# Contexto
- Es aquel problema que asocia un conjunto de instancias a una salida. 
- La etiqueta única de un conjunto es una función de etiquetas individuales de las instancias.
![[non standard learning-multiple-instance-problem.png]]
## Aplicaciones
- Predicción de actividad en fármacos, clasificación de imágenes por segmentos o predicción de bancarrotas.

# Taxonomías
- **Paradigma del espacio de instancias**: se construye un clasificador a nivel de instancia para discriminar las instancias en bolsas. El clasificador final a nivel de bolsa se obtiene agregando puntuaciones a nivel de instancia. Se considera las características de las instancias individuales, no las características de toda la bolsa (ej. **miSVM**).
- **Paradigma del espacio de bolsas**: la clasificación de una nueva bolsa se basa en la información proporcionada para toda la bolsa, no para las instancias individuales (ej. Citation **kNN**).
- **Paradigma del espacio incrustado**: En este paradigma se realiza un mapeo del espacio de bolsas a un único espacio vectorial. A continuación, se entrena un clasificador de instancia única tradicional (ej. **SimpleMI**).

# Mecanismo de Agregación
- Calcular la agregación de las predicciones y establecer un cierto umbral positivo.
- Dos parámetros esenciales:
	- El umbral a partir del cual se asigna a una clase positiva.
	- El operador de agregación (media, media armónica, etc.).

# Citation K-NN
- El método de votación por mayoría en *K-NN* puede fallar debido a falsos positivos en bolsas positivas. El enfoque de citación mejora esto considerando no solo los vecinos más cercanos de una bolsa, sino también las bolsas que la tienen como vecina, usando la distancia de *Hausdorff*. Esto permite predecir etiquetas basándose en referencias y citadores. 
- Los citadores son bolsas que consideran a la bolsa $b$ como uno de sus vecinos más cercanos.
- El método *Citation-KNN* clasifica una bolsa desconocida considerando tanto sus $K$ vecinos más cercanos ($Ne$) como sus $C$ citadores más cercanos ($Ci$). Se calculan cuatro valores:
	- $Kp$: Número de bolsas positivas en $Ne$.
	- $Kn$: Número de bolsas negativas en $Ne$.
	-  $Cp$: Número de bolsas positivas en $Ci$.
	- $Cn$: Número de bolsas negativas en $Ci$.
- Si $Kp+Cp>Kn+Cn$, la bolsa es positiva.
## Distancia de Hausdorff
- Mide la discrepancia máxima entre dos conjuntos de puntos $A$ y $B$. Calcula la mayor distancia mínima entre un punto de un conjunto y el punto más cercano del otro conjunto. Aquí puede afectar la dirección del cálculo, puede no ser lo mismo de $A$ a $B$ que al revés, ya que si un conjunto es más denso puede dar resultados diferentes.
$$d_{H}(A,B)=max(max_{a\in A}min_{b\in B}|a-b|,max_{b\in B}min_{a\in A}|b-a|)$$