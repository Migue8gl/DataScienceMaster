#tf-idf | #nlp | #termfrecuency | #inversefrecuency

# Concepto
- El **TF-IDF** (*Term Frequency-Inverse Document Frequency*) es un algoritmo de procesado de texto que sirve para evaluar la importancia de una palabra en un documento dentro de un conjunto de documentos (**corpus**).
- Se basa en dos conceptos principales:
	- **TF (Term Frecuency)**: MIde cuantas veces aparece una palabra en un documento. Cuando más aparece más importante podría ser:
	$$TF(t,d)=\frac{\text{Número deveces que aparece t en d}}{\text{Número total de términos en d}}$$
	- **IDF (Inverse Document Frecuency)**: Mide cuán común o rara es una palabra en el **corpus**. Palabras que aparecen mucho tienen un valor más bajo. $N$ es el número total de documentos y $|\{d\in D:t\in d\}|$ el número de documentos donde aparece $t$.
	$$IDF(t,D)=log(\frac{N}{1+|\{d\in D:t\in d\}|})$$
- Se combinan ambas métricas multiplicándolas entre sí.