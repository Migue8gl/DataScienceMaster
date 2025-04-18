- #statistics | #ml | #bayes 

# Estructura
- Se deben identificar eventos de interés o hipótesis. Estos eventos se agrupan en conjuntos mutuamente exclusivos para generar **variables hipótesis**.
- Identificar las fuentes que proporcionan información sobre las variables hipótesis: **variables informativas**.
- Establecer relaciones entre las variables identificadas. Este paso puede requerir el uso de **variables mediadoras**.
## Variables mediadoras
- Permiten expresar de forma más simple las relaciones entre las variables hipótesis e informativas.
- Su introducción agrega relaciones de independencia que simplifican el modelo.

# Ejemplos
![[bayesian networks-structure-example.png]]
- La primera aproximación seria considerar el modelo de la izquierda. Pero en caso de conocer el estado de embarazo, entonces las variables asociadas a los test se hacen independientes. Y ocurre que el resultado de estos test puede depender de otras causas, ademas del embarazo. Por ejemplo, del nivel hormonal (**NH**). Esto justifica la introduccién de esta variable mediadora.
- En el modelo de la derecha se utiliza la variable **mediadora** **NH**. Si se conoce el valor de **E** no se hacen independientes las variables de los test.
![[bayesian networks-structure-example2.png|300]]
- La leche de una vaca puede estar infectada. Para detectar esta situación se puede realizar un test con resultados positivo - negativo. 
- El test no es perfecto y puede producir falsos positivos y negativos.
* Hipótesis: variable I(Infección), con valores posibles si, no.
* Variable informativa: T(test), valores posibles positivo, negativo.
- Relación entre ellas: de infección a test.
![[bayesian networks-structure-example3.png]]
- El estado de la leche puede cambiar de un dia a otro: las vacas enfermas sanan y al revés. El granjero realiza el test todos los dias. Tras una semana dispone de los resultados de todos los tests. Considerando toda la información disponible podemos modelizar el paso del tiempo y usar la informacion del pasado para tomar nuevas decisiones.
- Este tipo de modelo que consideran el paso del tiempo se denominan **redes bayesianas dinámicas**.
- Los supuestos de este modelo son:
	- **Propiedad de Markov**: si conocemos el presente, entonces el pasado no tiene influencia en el futuro. Formalmente: $I_{i-1}\perp I_{i+1}|I_{i}$ donde $\perp$ indica la [[d-separation|d-separación]].
	- Dos nodos de test están $d$-separados si se dispne de información de alguno de los nodos de infección entre ellos: $T_{i}\perp T_{i+1}|I_{k},k=i\text{ o }k=i+1$ 
![[bayesian networks construction-structure-example4.png]]
- La primera condición, $I_{i-1}\perp I_{i+1}|I_{i}$, no siempre se cumple en la realidad. Las enfermedades suelen tener un trascurso natural a lo largo de varios dias. 
- Por ejemplo, si hoy tengo gripe y estaba sano ayer, entonces lo normal es que mañana también esté enfermo (es decir, el pasado sirve para saber el estado futuro). Por esta razón, el modelo anterior se ajusta más a la realidad.
![[bayesian networks construction-structure-example5.png]]
- La segunda condición, $T_{i}\perp T_{i+1}|I_{k}, k=i\text{ o }k=i+1$, implica que el fallo en un test no depende del funcionamiento del test previo. Sin embargo, si existe la posibilidad de haber lotes defectuosos de pruebas, esto podria no ser asi. 

# Técnicas de modelización
- [[bayesian networks modelization techniques]]

# Teorema de Descomposición
- [[decomposition theorem]]

# Modelo lineal Gaussiano
- [[linear gaussian model]]