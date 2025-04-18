- #ml | #statistics | #bayes | #naivebayes 

# Concepto
- Es un caso particular de una [[bayesian networks|red bayesiana]] en el que todas las variables son condicionalmente independientes dado $C$, donde $C$ es una variable aleatoria unidimensional.
- En clasificación supervisada $C$ es una clase de una observación y en no supervisada es un clúster.
- Se fija directamente la estructura que representa las restricciones impuestas.
- Parte de unas suposiciones muy restrictivas y poco realistas.
![[naive bayes-ex.png|300]]
- En la imagen se muestra *naive bayes* como una red bayesiana con variables $a\in A$, donde todas ellas son independientes entre sí, como muestra la estructura.
$$P(C|A_{1},A_{2},...,A_{n})=\alpha P(C)\cdot\prod_{i=1}^{n}P(A_{i}|C)$$
- Donde $\alpha$ es un factor de normalización que hace que las probabilidades sumen $1$ entre todas y $P(C)$ es la probabilidad a priori de $C$.
- Hace falta estimar $P(A_{i}|C)\forall A_{i}$ mediante el cálculo de frecuencias.
- Se calcula la clase $C^*$ más probable dada una serie de características $A_{i}$ de una observación.

# Complejidad 
|             | Entrenamiento | Clasificación |
| ----------- | ------------- | ------------- |
| **Tiempo**  | O(nt)         | O(kn)         |
| **Espacio** | O(knv)        | O(knv)        |

## **Explicación de los términos**
- **n**: Número de variables (atributos) $A_1, A_2, \dots, A_n$.
- **k**: Número de posibles valores de la variable de clase $C$ (es decir, el número de clases).
- **v**: Número promedio de valores distintos que puede tomar cada atributo $A_i$.
- **t**: Número total de muestras (ejemplos de entrenamiento).

# Semi NB
- [[semi naive nayes]]