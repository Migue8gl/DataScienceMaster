- #r | #ggplot | #plots | #statistics

# Concepto
- Permite construir gráficos de forma coherente y estructurada. Los gráficos se construyen a partir de capas, donde cada capa puede añadir información y complejidad al gráfico.

```r
library(ggplot2)

# Crear un dataframe de ejemplo
data <- data.frame(x = rnorm(100), y = rnorm(100))

# Crear un gráfico de dispersión
ggplot(data, aes(x = x, y = y)) +
  geom_point() +
  labs(title = "Gráfico de dispersión", x = "Eje X", y = "Eje Y")
```

# Tipos (más importantes)
| Tipo de Gráfico               | Descripción                                                                                                   | Código en R                                                                            |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Gráfico de dispersión         | Muestra la relación entre dos variables numéricas.                                                            | `ggplot(data, aes(x = variable1, y = variable2)) + geom_point()`                       |
| Gráfico de líneas             | Representa datos a lo largo del tiempo o una secuencia.                                                       | `ggplot(data, aes(x = tiempo, y = valor)) + geom_line()`                               |
| Histograma                    | Muestra la distribución de una variable numérica dividiendo los datos en intervalos.                          | `ggplot(data, aes(x = variable)) + geom_histogram(bins = 30)`                          |
| Gráfico de caja               | Muestra la distribución de una variable numérica a través de cuartiles.                                       | `ggplot(data, aes(x = categoria, y = variable)) + geom_boxplot()`                      |
| Gráfico de barras             | Representa la cantidad o frecuencia de categorías.                                                            | `ggplot(data, aes(x = categoria)) + geom_bar()`                                        |
| Gráfico de barras apiladas    | Muestra la composición de diferentes grupos en cada categoría.                                                | `ggplot(data, aes(x = categoria, fill = subcategoria)) + geom_bar(position = "stack")` |
| Gráfico de áreas              | Similar a un gráfico de líneas, pero el área debajo de la línea se llena.                                     | `ggplot(data, aes(x = tiempo, y = valor)) + geom_area()`                               |
| Gráfico de violín             | Combina un gráfico de caja con un histograma, mostrando la densidad de la distribución.                       | `ggplot(data, aes(x = categoria, y = variable)) + geom_violin()`                       |
| Gráfico de puntos             | Muestra la relación entre dos variables o la frecuencia de un solo grupo.                                     | `ggplot(data, aes(x = variable1, y = variable2)) + geom_dotplot()`                     |
| Gráfico de mosaico            | Representa la relación entre dos o más variables categóricas.                                                 | `ggplot(data, aes(x = categoria1, fill = categoria2)) + geom_mosaic()`                 |
| Gráfico de cuantiles (qqplot) | Representa la distribución de una variable utilizando los cuantiles y en referencia a la distribución normal. | `ggplot(data, aes(sample = value)) + stat_qq() + stat_qq_line(color = "red")`          |
# Funciones adicionales
- `labs(title, subtitle, x, y, size)` -> Permite añadir títulos, subtítulos, etiquetas a los ejes.
- `scale_color_brewer(), scale_color_gradient(), ...` -> Personalizan los colores de un gráfico.
- `scale_fill_*` -> Igual que los `scale_color_*` pero para rellenar colores de áreas.
- `scale_x_continuous(limits, breaks, labels)` -> Personalizan las escalas de los ejes (también existe para y).  `limits` define los límites de la escala, `breaks` los puntos donde mostrar las marcas en el eje, `labels` personaliza las etiquetas en los puntos de ruptura.
- `scale_x_discrete(limits, labels)` -> Se utilizan para las variables categóricas, donde `limits` define el orden de las categorías y `labels` personaliza las etiquetas de las categorías.
- `facet_wrap()` -> Permite dividir un gráfico en múltiples paneles (o facetas) según los valores de una o más variables categóricas. Esto es especialmente útil para comparar distribuciones o relaciones entre variables en diferentes grupos dentro del mismo conjunto de datos. Ejemplo: `face_wrap(~ var1)`.