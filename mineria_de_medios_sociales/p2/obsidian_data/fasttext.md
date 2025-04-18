- #fasttext | #embedding | #nlp

# Concepto
- **FastText** es un algoritmo de *embedding*. Este mejora [[word2vec]] al descomponer las palabras en subunidades llamadas **n-gramas**. Estas son secuencias de caracteres dentro de las palabras, lo que permite que *FastText* genere representaciones no solo para palabras completas, sino también para los fragmentos dentro de esas palabras.
- *FastText* genera un vector para cada **n-grama** (subcadena de la palabra) y luego, para cada palabra, la representación se calcula como la suma de los vectores de sus **n-gramas**. Esto permite que *FastText* tenga representaciones más robustas para palabras que nunca han aparecido en el conjunto de datos de entrenamiento, ya que puede generalizar a partir de los **n-gramas** que componen una palabra.
- Mejor rendimiento en el manejo de palabras raras o desconocidas (*out-of-vocabulary*).
- Mejora en lenguajes morfológicamente ricos (como el alemán o el finlandés) porque entiende la estructura interna de las palabras.
- FastText es particularmente útil para lenguajes que combinan muchas raíces y sufijos, ya que los n-gramas pueden capturar estas variaciones de manera más eficiente.
![[fasttext-vs-word2vec.png]]