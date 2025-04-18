- #smote | #oversampling | #outlier | #ml

# Concepto
- En **SMOTE** (*Synthetic Minority Over-sampling Technique*) para una instancia de una clase minoritaria, se tienen en cuenta algunos (no más de cinco) vecinos más cercanos para generar nuevas instancias que sean linealmente parecidas a los vecinos y la instancia seleccionada.
- Esto se logra interpolando valores entre las instancias originales y sus vecinos, creando puntos de datos sintéticos que pertenecen a la clase minoritaria.
![[outlier detection-smote.png]]

# Desventajas
- Se generan datos bajo la suposición de que las características de las instancias interpoladas son representativas y válidas, lo cual puede no ser cierto si clase minoritaria sigue una distribución compleja.
- Si las instancias de la clase minoritaria incluyen ruido, el método llevará a la amplificación del mismo.
- Si no se controla, puede hacer que se sobreajuste demasiado el modelo.