#word2vec | #embedding | #nlp 

# Concepto
- Es una técnica de procesamiento del lenguaje que sirve para obtener un vector como representación de una palabra, lo que se conoce como [[embedding]].
- El algoritmo *word2vec* estima este vector dado un **corpus** de palabras.
- *Word2Vec* se basa en el **contexto** de las palabras, bajo la hipótesis de que "una palabra se conoce por el contexto que la rodea" (la hipótesis distribucional). Usa redes neuronales simples para aprender estos vectores.
- Limitación: Si la palabra nunca se ha visto en el conjunto de datos de entrenamiento, no tiene representación.

# Arquitecturas
![[word2vec-skipgramvscbow.png]]
## CBOW (Continuous Bag of Words)
- Predice una palabra dado su contexto.
- Ejemplo: Si tenemos "El gato ___ en el tejado", el modelo intenta predecir la palabra que falta ("está") a partir de las palabras vecinas.
- Es eficiente con corpus pequeños.
## **Skip-gram**
- Predice el contexto dado una palabra.
- Ejemplo: Si tienes la palabra "está", intenta predecir que las palabras vecinas son "gato", "en", etc.
- Funciona mejor con corpus grandes y palabras raras.