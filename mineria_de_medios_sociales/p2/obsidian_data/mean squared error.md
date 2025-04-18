- #mse | #ml

# Concepto
El **Error Cuadrático Medio** (MSE, por sus siglas en inglés *Mean Squared Error*) es una métrica ampliamente utilizada para evaluar la precisión de un modelo de regresión. Se calcula como el promedio de los cuadrados de los errores, donde cada error es la diferencia entre el valor real y el valor predicho.
## Definición Matemática
Sea $y_i$ el valor real y $\hat{y}_i$ el valor predicho para $i = 1, 2, \dots, n$, donde $n$ es el número total de observaciones. El MSE se define como:
$$ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$
## Interpretación
- **Valores bajos** de MSE indican que las predicciones están cerca de los valores reales.
- **Valores altos** de MSE indican mayores diferencias entre predicciones y valores reales.
El MSE penaliza los errores más grandes de forma cuadrática, lo que significa que los errores grandes tienen un impacto mucho mayor en la métrica que los errores pequeños.
## Propiedades
1. **No negativo:** El MSE siempre es $\geq 0$.
2. **Sensibilidad a outliers:** Debido a la cuadratura de los errores, el MSE es muy sensible a valores atípicos.
3. **Escalabilidad:** La magnitud del MSE depende de la escala de los datos.
## Ejemplo en Python

```python
import numpy as np

# Valores reales y predichos
y_real = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

# Cálculo del MSE
mse = np.mean((y_real - y_pred) ** 2)
print(f"Error Cuadrático Medio: {mse}")
```

# RSME
- El **RMSE** (por sus siglas en inglés *Root Mean Squared Error*) es una métrica utilizada para evaluar la precisión de los modelos de regresión. Representa la raíz cuadrada del **Error Cuadrático Medio (MSE)** y proporciona una medida de cuánto se desvían las predicciones de los valores reales.
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$