- #naivebayes | #bayesiannetwork 

# Concepto
- El **aprendizaje bayesiano semi-ingenuo** (_Semi-Naive Bayesian Learning_) es una variante del [[naive bayes]] que relaja la suposición de independencia condicional entre los atributos.
- El método de **semiNB** introduce dependencias parciales entre algunos atributos para mejorar la precisión del modelo.

# TAN (Tree Augmented Naive Bayes)
- Emplea una estructura en árbol donde cada atributo solo depende de la clase $C$ y otro atributo.
- Se utiliza la medida de información condicional:
$$I(A_{i},A_{j},C)=\sum_{A_{i},A_{j},C}P(A_{i},A_{j},C)\cdot log(\frac{P(A_{i},A_{j}|C)}{P(A_{i}|C)\cdot P(A_{j}|C)})$$
- Explica qué tan dependientes son $A_{i}$ y $A_{j}$ entre sí cuando se conoce $C$.
- Esta métrica es el peso que pondera las arista de relación entre $A_{i}$ y $A_{j}$.
- Se construye el árbol expandido de máximo peso mediante el algoritmo de **Kruskal**.
- Se introduce $C$ y todos los enlaces originales de [[naive bayes]].
![[semi naive nayes-ex.png|300]]
## Complejidad
|                | Entrenamiento              | Clasificación            |
|----------------|----------------------------|--------------------------|
| **Tiempo**     | O(n²t + nt) = O(n²t)       | O(kn)                    |
| **Espacio**    | O(knv)                      | O(knv)                   |
### **Explicación de los términos**
- **n**: Número de variables (atributos) $A_1, A_2, \dots, A_n$.
- **k**: Número de posibles valores de la variable de clase $C$ (es decir, el número de clases).
- **v**: Número promedio de valores distintos que puede tomar cada atributo $A_i$.
- **t**: Número total de muestras (ejemplos de entrenamiento).

# BAN (Bayesian Network Augmented Naive Bayes Bayesian Classifier)
- Es un clasificador bayesiano de $n$ dependencias.
- Se fija la estructura de [[naive bayes]] y se buscan los arcos entre atributos mediante cualquier algoritmo de aprendizaje de [[bayesian networks|redes bayesianas]].
![[semi naive nayes-ban.png|300]]