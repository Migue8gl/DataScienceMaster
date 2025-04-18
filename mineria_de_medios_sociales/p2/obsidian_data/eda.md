-  #analysis | #eda | #exploration | #data | #datascience 

#  Análisis Exploratorio de Datos (EDA): Guía Completa
## 1. Definición del Problema
### a. Identificación del Tipo de Problema
Determina si se trata de un problema de **clasificación** o **regresión**:
- **Clasificación**: La variable objetivo es categórica. Ejemplos: predicción de spam/no spam, diagnóstico médico.  
- **Regresión**: La variable objetivo es continua. Ejemplos: predicción de precios de viviendas, demanda de energía.
```r
# Ejemplo en R: Clasificación vs Regresión
# Cargar datos y revisar si la variable objetivo es categórica o continua
data(mtcars)
str(mtcars$mpg)  # Regresión: mpg es numérica
mtcars$cyl <- as.factor(mtcars$cyl)
str(mtcars$cyl)  # Clasificación: cyl es categórica
```
### b. Variables Dependientes e Independientes
- **Variable Dependiente** (objetivo o salida): Define qué variable se predice.
- **Variables Independientes** (predictoras o entradas): Determina qué variables influyen en la salida.
```r
# Ejemplo en R: Variables dependientes e independientes 
# mpg (dependiente) y hp, wt (independientes) 
model <- lm(mpg ~ hp + wt, data = mtcars) summary(model)
```
### c. Hipótesis Iniciales
Plantea hipótesis basadas en el problema:
- ¿Qué factores podrían influir en la variable objetivo?
- ¿Esperas correlaciones positivas o negativas específicas?
**Ejemplo de hipótesis para regresión:**
- "El precio de la vivienda aumenta con la superficie en metros cuadrados."
**Ejemplo de hipótesis para clasificación:**
- "Las personas con más de X años tienen más probabilidad de comprar cierto producto."
```r
# Ejemplo en R: Verificación de hipótesis
# Hipótesis: mpg disminuye al aumentar el peso del auto
plot(mtcars$wt, mtcars$mpg, main = "Relación entre peso y mpg", xlab = "Peso", ylab = "MPG")
abline(lm(mpg ~ wt, data = mtcars), col = "red")

```
## 2. Preparación de los Datos
### a. Descripción Inicial de los Datos
1. **Resumen Estadístico General**:
   - Media, mediana, moda, desviación estándar.
   - Valores mínimos y máximos.
2. **Tipo de Variables**:
   - Categóricas vs Numéricas.
   - Fechas, texto, o datos mixtos.
```r
# Resumen estadístico
summary(mtcars)
# Conversión de tipos
mtcars$cyl <- as.factor(mtcars$cyl)
```
### b. Procesamiento de Datos
#### i. Formateo de Datos
- **Conversión de tipos**: Asegura que las variables tienen el tipo correcto (e.g., fechas como `datetime`).
- **Estandarización de nombres**: Renombra columnas para consistencia y claridad.
#### ii. Limpieza de Datos
- **Valores Ausentes (Missing Values)**:
  - *Eliminar registros completos*: Si los datos faltantes son pocos y no cruciales.
  - *Imputación*: Usar métodos como:
    - Media/mediana para datos numéricos.
    - Moda para datos categóricos.
    - Modelos avanzados como [[k-nearest neighbors|KNN]] o [[linear regression|regresión]].
```r
# Imputación de valores ausentes con la media
mtcars$mpg[is.na(mtcars$mpg)] <- mean(mtcars$mpg, na.rm = TRUE)
```
- **Outliers**:
  - Detecta con métodos como:
    - **Z-score** (valor absoluto > $3$).
    - **IQR** (valores fuera de $[Q1 - 1.5\cdot IQR, Q3 + 1.5\cdot IQR]$).
  - Decide si eliminarlos o transformarlos según el contexto del dominio.
```r
  # Detectar outliers usando IQR
IQR_val <- IQR(mtcars$mpg)
outliers <- mtcars$mpg[mtcars$mpg < (quantile(mtcars$mpg, 0.25) - 1.5 * IQR_val) | mtcars$mpg > (quantile(mtcars$mpg, 0.75) + 1.5 * IQR_val)]
outliers
```
### c. Resumen Visual de los Datos
#### i. Análisis Univariado
1. **Histograma**: Distribución de cada variable numérica.
2. **Gráficos de barras**: Para variables categóricas.
3. **Boxplots**: Identificación visual de [[outliers]].
#### ii. Análisis Bivariado
1. **Gráficos de dispersión (scatter plots)**: Relaciones entre dos variables numéricas.
2. **Diagramas de caja y bigote (boxplots)** por categoría: Comparación de distribuciones.
3. **Matriz de correlación**: Para identificar relaciones entre variables numéricas.
#### iii. Verificación de Hipótesis
- Crear gráficos como:
  - Boxplots para clasificaciones categóricas frente a la variable objetivo.
  - Líneas de tendencia en gráficos de dispersión.
### d. Transformación y Generación de Nuevas Variables
#### i. Descomposición de Variables Complejas
- **Fechas**: Extraer año, mes, día, hora.
```r
# Extraer año, mes, día de una fecha
mtcars$date <- as.Date("2024-11-29")
mtcars$year <- format(mtcars$date, "%Y")
```
- **Texto**: Usar técnicas de tokenización, conteo de palabras, etc.
- **Variables categóricas**: Convertir a variables dummy (*one-hot encoding*).
#### ii. Identificación de Datos Redundantes
- **Correlación**:
  - Variables con correlación alta ($> 0.9$) pueden eliminarse si no aportan valor añadido. Aunque esto no tiene por qué ser siempre así, solo cuando haya información de que las variables son redundantes. Muchas veces correlación alta no es redundancia entre variables.
```r
# Matriz de correlación
cor_matrix <- cor(mtcars[, sapply(mtcars, is.numeric)])
cor_matrix
```
- **Reducción de Dimensionalidad**:
  - Usar **PCA** ([[pca|Análisis de Componentes Principales]]) si hay muchas variables correlacionadas.
#### iii. Transformaciones Necesarias
1. **Distribución Normal**:
   - Verificar con histograma o prueba de *Shapiro-Wilk*.
   - Aplicar transformaciones como:
     - Logaritmo.
     - Raíz cuadrada.
2. **Estandarización o Normalización**:
   - Estandarización (**z-score**): Útil si los datos tienen diferentes escalas.
   - Normalización (**min-max scaling**): Escalar datos en un rango de $0$ a $1$.
## 3. Conclusiones Finales
### a. Hipótesis Verificadas
- Indica cuáles de las hipótesis iniciales fueron comprobadas.
- Proporciona evidencia visual o estadística.
### b. Selección de Variables para el Modelo
1. **Variables clave seleccionadas**:
   - Explica por qué ciertas variables fueron seleccionadas.
   - Justifica eliminación o transformación de variables.
2. **Resumen del proceso**:
   - Enumera las transformaciones realizadas (imputación, eliminación de outliers, etc.).
3. **Próximos pasos**:
   - Preparación para modelado: Validación cruzada, selección de algoritmo.
