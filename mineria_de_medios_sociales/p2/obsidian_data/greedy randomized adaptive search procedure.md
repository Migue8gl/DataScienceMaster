- #grasp | #metaheuristic 

- Es una metaheurística comunmente aplicada a problemas de optimización combinatoria.
- Normalmente, **GRASP** consiste en iteraciones hechas a partir de construcciones de soluciones aleatorias del algoritmo **greedy** y posteriormente mejoradas de manera iterativa a partir de una [[local search|búsqueda local]].
- Las soluciones aleatorias greedy son generadas añadiendo elementos al conjunto de soluciones del problema a partir de un conjunto ordenado por una función greedy, la cual evalua la calidad de esta.
- Para obtener **variabilidad** en la solución, las mejores soluciones dentro del conjunto greedy suelen introducirse en una *lista restringida de candidatos* y despues al escoger elementos para la solución, se hace de manera aleatoria.