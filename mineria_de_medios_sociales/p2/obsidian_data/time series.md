- #timeseries | #ml

# Contexto
- Una serie temporal es una magnitud observada a lo largo del tiempo en intervalos (días, minutos, horas, semanas, etc.).
- Los datos de las series temporales son útiles cuando se está prediciendo algo que cambia con el tiempo.
- La predicción de series temporales pretende calcular cómo continúa la serie temporal en el futuro.

# Autocorrelación
- Esta métrica se refiere a la relación entre valores retrasados de una serie temporal.
- Correlación de una señal con respecto a una copia de ella misma retrasada en el tiempo.
- Es una herramienta para encontrar patrones repetitivos como la presencia de periodicidad (ocultada por el ruido).
![[time series-autocorrelation.png]]

# Cross validation
- En problemas de machine learning tradicionales, se puede dividir aleatoriamente los datos en conjuntos de entrenamiento y prueba. Sin embargo, en **series temporales**, los datos tienen una **estructura secuencial**, por lo que un muestreo aleatorio puede hacer que el modelo entrene con datos futuros y pruebe en datos pasados, lo que no refleja la realidad.
## Blocked Time Series Split
Este método divide los datos en bloques consecutivos y no mezcla información entre el pasado y el futuro.
![[time series-cross-validation.png|500]]

# Análisis de residuos
- Un buen método de predicción debe producir [[residuals|residuos]] que cumplan lo siguiente:
	- No están correlados.
	- La media es cero.
- Cualquier método que no verifique estas propiedades puede ser mejorado. Además, si la varianza de los residuos es constante y estos se distribuyen según una normal es muy buen indicativo.

# Errores de evaluación
- **Error:** 
	- $e_i = y_i - \hat{y}_i$
- **Error dependiente de escala:** 
	- $MAE = \text{mean}(|e_i|)$
	- $RMSE = \sqrt{\text{mean}(e_i^2)}$ ([[mean squared error]])
- **Error en %:**
	- $p_i = 100 \frac{e_i}{y_i}$
- **Errores escalados:**
	- $MAPE = \text{mean}(|p_i|)$
	- $sMAPE = \text{mean} \left( 200 \frac{|y_i - \hat{y}_i|}{y_i + \hat{y}_i} \right)$
	- $q_j = \frac{e_j}{\frac{1}{T-1} \sum_{t=2}^{T} |y_t - y_{t-1}|}$
	- $MASE = \text{mean}(|q_j|)$

# Intervalo de predicción
- Un intervalo de predicción es un intervalo dentro del cual está el valor esperado con una probabilidad especificada.
- Cuando la predicción es de un paso, la desviación estándar del predictor es casi la misma que la desviación estándar de los residuos.

# Descomposición
- Las series temporales pueden mostrar gran cantidad de patrones que puede categorizarse por varios comportamientos. Al descomponerlos es más fácil ajustarlos.
- Algunos patrones como tendencia, estacionalidad, irregularidad o cíclicidad.
## Descomposición aditiva
$$y_{t}= S_{t}+T_{t}+E_{t}$$
## Descomposición multiplicativa
$$y_{t}= S_{t}\cdot T_{t}\cdot E_{t}$$
![[time series-decomposition-example.png|500]]
## Media móvil
- La media móvil sirve para para obtener la tendencia de una serie temporal. Esta suaviza la serie. Es como una media deslizante sobre la serie.
- Calcula la media de los $k$ puntos vecinos.
$$\hat{T}_{t}=\frac{1}{m}\sum_{j=-k}^{k}y_{t}+j$$
## STL
- La **descomposición STL** (_Seasonal-Trend decomposition using LOESS_) es un método utilizado para descomponer una serie temporal en tres componentes principales:
	- **Tendencia ($T_{t}$​)**: Representa la evolución a largo plazo de la serie.
	- **Estacionalidad ($S_{t}$​)**: Captura los patrones repetitivos que ocurren en intervalos regulares (ej. mensual, anual).
	- **Residuo ($R_{t}$)**: Es la parte aleatoria o el "ruido" que queda después de extraer tendencia y estacionalidad.
- Método robusto y versátil.

# ARIMA
- **ARIMA** (_AutoRegressive Integrated Moving Average_) es un modelo estadístico utilizado para analizar y predecir series temporales. Se basa en tres componentes principales:
1. **Autoregresión (AR, AutoRegressive)**: Utiliza valores pasados de la serie para predecir el futuro. Se controla con el parámetro $p$.
2. **Diferenciación (I, Integrated)**: Transforma la serie en estacionaria restando los valores anteriores. Se controla con el parámetro $d$.
3. **Media Móvil (MA, Moving Average)**: Modela la relación entre el error de predicción y los errores pasados. Se controla con el parámetro $q$.