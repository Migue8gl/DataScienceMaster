- #datamining | #datascience | #outlier

# Introducción
- La detección de anomalías aglomera todos los métodos que se utilizan para la detección de sucesos u observaciones extrañas en los conjuntos de datos analizados.
- Los [[outliers]] o anomalías son esas observaciones extrañas y poco frecuentes, las cuales pueden ser:
	- Rarezas dentro de los datos reales.
	- Datos incorrectos debido a fallos en el proceso de obtención de datos.
- En el primer caso se necesita de un análisis exhaustivo del conjunto de datos para entender estos y poder identificarlos.
- Existen algoritmos basados en firmas, como el software **NIDS**. En esos casos la anomalía es conocida y se sabe qué es lo que se busca. La detección de anomalías, por lo general, trata con datos extraños, pero no conocidos, por lo que se deben utilizar técnicas más complejas.

# Métodos supervisados
- [[supervised outlier detection]]

# Métodos semisupervisados
- [[semi supervised learning]]
- En estos métodos, lo normal es encontrar un conjunto de datos con datos etiquetados para las clases normales, pero ningún ejemplo de valor anómalo. De esta forma, a partir de los datos comúnes, se intenta extraer los patrones que hacen que esos valores sean normales.
- Una anomalía sería una nueva instancia que no encaje con este perfil de normalidad.
- Se suelen utilizar métodos probabilísticos como cadenas de Markov o redes bayesianas. También se utilizan bastante las [[support vector machine|máquinas de vectores de soporte]] con una sola clase (la clase normal).

# Métodos no supervisados
- [[unsupervised outlier detection]]

# Evaluación
- Asumamos que tenemos un proceso por el cual somos capaces de decir si una predicción para nuevas entradas es correcta o no, es decir, un conjunto de test.

| Confusion Matrix    | Predicted Class NC  | Predicted Class A   |
| ------------------- | ------------------- | ------------------- |
| **Actual Class NC** | True Negative (TN)  | False Positive (FP) |
| **Actual Class A**  | False Negative (FN) | True Positive (TP)  |

## Notación:
- **NC**: Clase normal
- **A**: Clase de anomalía
- **P**: Positivo (Predicción es A)
- **N**: Negativo (Predicción es NC)
- **T**: True (Predicción es correcta)
- **F**: False (Predicción es incorrecta)
## Global Accuracy
$$acc = \frac{TN+TP}{TN+TP+FN+FP}$$
- No nos sirve, pues en el caso de las anomalías suele haber desbalance extremo y un predictor que siempre prediga la clase mayoritaria tendrá un *accuracy* muy bueno.
## Medidas de un solo valor
- Además del *accuracy* (que no es buena métrica) existen [[supervised outlier detection|recall y f1-score]].
- Otras:
	- **Precision of A** -> $\frac{TP}{TP+FP}$ -> ¿Qué porcentaje de predicciones de anomalías eran corectas?
	- **Specifity** (true negative rate) -> $\frac{TN}{TP+FN}$ -> ¿Que porcentaje de los casos normales has captado?
	- **False Positive Rate** -> $\frac{FP}{FP+TN}=P(Predict=A|Real=NC)$ -> ¿Qué porcentaje de los casos normales fueron clasificados erroneamente como anomalías?
![[outlier detection-evaluation-1.png|500]]
![[outlier detection-evaluation-2.png|500]]
- Conseguir una precisión alta suele ser a costa de un *recall* bajo, de igual forma ocurre al revés. Un balance es necesario. Por ello se suele usar directamente el [[f-score]], que es la media armónica de ambas métricas.
## Problemas
 - Este tipo de medidas funcionan bien y son útiles con *datasets* desbalanceados, pero ocurren nuevos problemas cuando se trabaja con conjuntos de datos **extremadamente** desbalanceados.