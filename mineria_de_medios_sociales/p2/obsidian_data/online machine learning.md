- #online | #secuencia

# Concepto
- Es un método de aprendizaje automático en el que los datos están disponibles en un orden secuencial y se utilizan para actualizar el mejor predictor para datos futuros en cada paso, de manera casi inmediata, a diferencia de las técnicas de aprendizaje por lotes, que generan el mejor predictor aprendiendo sobre todo el conjunto de datos de entrenamiento a la vez. 
- El aprendizaje en línea es una técnica comúnmente utilizada en áreas del aprendizaje automático en las que es inviable desde el punto de vista computacional entrenar sobre todo el conjunto de datos, lo que requiere la necesidad de algoritmos *out-of-core*.
- También se utiliza en situaciones en las que es necesario que el algoritmo se adapte dinámicamente a nuevos patrones en los datos, o cuando los propios datos se generan en función del tiempo, por ejemplo, la predicción del precio de las acciones. 
- Los algoritmos de aprendizaje en línea pueden ser propensos a interferencias catastróficas, un problema que puede resolverse con enfoques de aprendizaje incremental. Esto se refiere al modelo tendiendo a olvidar información previamente aprendida al aprender nueva información

# CVFDT (Concept-adapting VFDT)
- Es un algoritmo de árbol diseñado para el problema del [[concept drift]] que mantiene al árbol consistente con una ventana deslizante.
- Mantiene estadísticas en todos los nodos del árbol para las distintas divisiones candidatas. Comienza a construir ramas del árbol alternativas cuando otras divisiones son mejores.
- Periódicamente evalúa la validez del modelo y sustituye un nodo por otro si la alternativa mejora.
- No manejan atributos continuos y son mucho más lentos que un *VFDT*