- #statistics | #regression

# Concepto
- Una interacción ocurre cuando el efecto de una variable independiente sobre la variable dependiente cambia según el nivel de otra variable independiente.

# Interacciones en Modelos de Regresión
- Las interacciones en modelos de regresión pueden aplicarse tanto a variables cuantitativas como cualitativas, o a una combinación de ambas. Un ejemplo interesante es la interacción entre una variable cualitativa y una cuantitativa.
## Ejemplo: Predicción del Balance de Crédito
- Consideremos un modelo para predecir el balance de crédito usando el ingreso (cuantitativa) y el estatus de estudiante (cualitativa). Sin un término de interacción, el modelo sería:
$$
\text{balance}_i \approx \beta_0 + \beta_1 \times \text{income}_i + 
\begin{cases} 
\beta_2 & \text{si la i-ésima persona es estudiante} \\
0 & \text{si la i-ésima persona no es estudiante}
\end{cases}
$$

$$
= \beta_1 \times \text{income}_i + 
\begin{cases}
\beta_0 + \beta_2 & \text{si la i-ésima persona es estudiante} \\
\beta_0 & \text{si la i-ésima persona no es estudiante}
\end{cases}
$$
- Este modelo ajusta dos líneas paralelas a los datos: una para estudiantes y otra para no estudiantes. Las líneas tienen diferentes interceptos ($\beta_0 + \beta_2$ vs $\beta_0$) pero la misma pendiente ($\beta_1$).
### Limitación del Modelo
- Este enfoque implica que el efecto promedio en el balance por un cambio de una unidad en el ingreso no depende de si el individuo es estudiante o no. Esto puede ser una limitación seria, ya que un cambio en el ingreso podría tener un efecto muy diferente en el balance de la tarjeta de crédito de un estudiante versus un no estudiante.
### Modelo con Interacción
- Para abordar esta limitación, podemos agregar un término de interacción multiplicando el ingreso con la variable dummy para estudiante:
$$
\text{balance}_i \approx \beta_0 + \beta_1 \times \text{income}_i + 
\begin{cases}
\beta_2 + \beta_3 \times \text{income}_i & \text{si es estudiante} \\
0 & \text{si no es estudiante}
\end{cases}
$$

$$
= 
\begin{cases}
(\beta_0 + \beta_2) + (\beta_1 + \beta_3) \times \text{income}_i & \text{si es estudiante} \\
\beta_0 + \beta_1 \times \text{income}_i & \text{si no es estudiante}
\end{cases}
$$
- Ahora tenemos dos líneas de regresión diferentes para estudiantes y no estudiantes, con diferentes interceptos ($\beta_0 + \beta_2$ vs $\beta_0$) y diferentes pendientes ($\beta_1 + \beta_3$ vs $\beta_1$). Esto permite la posibilidad de que los cambios en el ingreso afecten los balances de las tarjetas de crédito de estudiantes y no estudiantes de manera diferente.

# Principio de jerarquía
- Este principio establece que si se incluye un término de interacción en un modelo, también se deben incluir los efectos principales correspondientes (es decir, las variables individuales), incluso si estos efectos principales no son estadísticamente significativos por sí solos.
- Razones para incluirlos:
	- **Interpretabilidad**: Incluir los efectos principales hace que la interpretación del modelo sea más clara y completa. 
	- **Estabilidad del modelo**: Omitir los efectos principales puede llevar a estimaciones inestables de los coeficientes de interacción.
	- **Coherencia teórica**: En muchos casos, tiene sentido conceptual incluir los efectos principales si se está considerando su interacción.