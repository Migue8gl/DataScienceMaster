- #dl | #transformers 

- Un **logit** es la puntuación final del token antes de pasar por la función [[softmax]].
- Un logit, en general, es expresado de la siguiente forma:
	- $logit(p)=log(\frac{p}{1-p})$ 
- Es decir, un logit es el logaritmo natural de los [[probability#^3291b4|odds]]. De esta forma, se transforma una probabilidad entre $[0,1]$ al rango de todos los números [[number theory#^4aab70|reales]].
- La función [[softmax]] por tanto, vuelve a convertir este valor en un número en el rango $[0,1]$ para interpretarlo como una probabilidad.