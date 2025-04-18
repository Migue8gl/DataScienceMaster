- #outliersdetection | #supervised

# Introducción
- Cuando se tienen un conjunto de datos, entonces, se pueden usar métodos de aprendizaje supervisado. Existe una diferencia con respecto a otros tipos de problemas supervisados, ya que este caso será, seguramente, un problema **desbalanceado**. Lo más normal es que no haya demasiadas anomalías. Se pueden dar que el clasificador tenga una **precisión** muy alta, pero porque la clase normal sea mayoritaria.
- Con dos clases (normal y **outlier**), con un porcentaje del $98\%$ para la clase normal y $2\%$ para la clase anómala, por tanto una precisión del $98\%$ no tiene por qué ser nada buena si no se detecta la clase minoritaria.
![[outlier detection-example-unbalanced-data.png|500]]

# Métodos de instancias
- Modifican los datos.
## Undersampling
- Esta clase de métodos elimina instancias de la clase mayoritaria:
	- [[tomek-links]]
	- [[cnn]]
	- Se pueden combinar varios métodos.
## Oversampling
- En este tipo de métodos se introducen datos artificiales dentro de la clase minoritaria. El ejemplo más famoso de este tipo de técnicas es **[[smote|SMOTE]]**.
- Este tipo de métodos son delicados, ya que generan datos artificiales bajo la suposición de que son correctos, por ello la supervisión de un experto es, si no necesaria, muy conveniente.

# Métodos de algoritmos
- Estos métodos no modifican los datos, sino el algoritmo de aprendizaje.
## Ponderación
- Cambiando la forma en la que la función de coste penaliza errores es posible dar más importancia a las instancias minoritarias, solo bastaría con asignar mayor coste a un error en una instancia minoritaria.
## Métodos de *ensembles*
- Utilizando métodos de [[ensembles]] existen diversos métodos:
	- Usando [[bagging]], se incluyen subconjuntos de datos con preferencia de presencia de instancias minoritarias en cada iteración del algoritmo.
	- Usando [[boosting]], se le da mayor peso en la operación de actualización de distribución a las instancias con clase minoritaria en cada paso del algoritmo.
	- Métodos híbridos.
![[outlier detection-algorithms-methods-ensembles.png|500]]

# Métricas
- Métricas como el *accuracy* no son las más deseables, por lo mencionado anteriormente.
## Recall
- *Recall*, también conocida como sensibilidad o tasa de verdaderos positivos, mide la proporción de positivos reales que se identifican correctamente.
- ¿Que porcentaje de anomalías has captados?
$$\text{recall} = \frac{\text{VP}}{\text{VP} + \text{FN}}$$
## F-Measure (f1-score)
- *F-Measure*, también conocida como **F1-score**, es la media armónica de la precisión (*accuracy*) y *recall*.
$$\text{F1-score} = 2 \cdot \frac{\text{Accuracy} \cdot \text{Recall}}{\text{Accuracy} + \text{Recall}}$$
## Curvas ROC
- Las curvas **ROC** (Característica Operativa del Receptor) representan la tasa de verdaderos positivos (*recall*) frente a la tasa de falsos positivos a varios umbrales de decisión.
- Las curvas **ROC** proporcionan una visión completa del rendimiento del modelo, lo que permite seleccionar el umbral de decisión óptimo.
![[supervised outlier detection-roc.png|340]]