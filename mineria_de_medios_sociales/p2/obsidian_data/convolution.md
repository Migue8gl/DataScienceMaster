- #convolution | #images | #cv

# Introducción
- Matemáticamente la convolución es un operador de dos funciones que producen una tercera función.
- En visión por computador, una convolución se traduce como el [[dot product]] deslizante (ocurre en una ventana deslizante) entre el **kernel** y el *ROI* (*region of interest*). 
- El **kernel** es una matriz pequeña que actua como filtro o extractor.
- El **kernel** se desplaza por la imagen de entrada en un proceso de ventana deslizante que va aplicando operaciones de convolución.

# Proceso
- El kernel se desplaza por la imagen en un proceso de ventana deslizante.
- En cada pasada genera un valor que actua como resumen de ese **ROI** dada una matriz de **kernel**.
$$O(i,j)=\sum_{m=0}^{M-1}\sum_{n=0}^{N-1}I(i+m,j+n)\cdot K(m,n)+b$$
- Donde $M\times N$ es el tamaño del kernel, $b$ es el *bias*, $I$ es una matriz de entrada y $K$ es el kernel.
![[convolution-example.png]]
![[convolution-example2.png|550]]