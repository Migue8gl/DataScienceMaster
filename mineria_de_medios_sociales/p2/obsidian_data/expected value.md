- #probability | #expectedvalue

# Concepto
- El *expected value* o **esperanza** es el valor promedio o valor esperadode una función $f(x)$ teniendo en cuenta la distribución de $x$.

# Fórmula
- Para funciones discretas:
$$E(f(x))= \sum_{x}f(x)P(x)$$
- Para funciones continuas:
$$E(f(x))=\intop f(x)P(x)\quad dx$$
- **Nota:** Se usa $P(x)$ para notar tanto [[normal distribution|función de densidad]] como función masa de probabilidad (lo mismo, pero discreta).

# Reglas
- El valor esperado de una constante es una constante: $E(k)=k$.
- El valor esperado de una constante por una función es el valor esperado de esa función multiplicado por la constante: $E(kf(x))=kE(f(x))$.
- La esperanza de una suma de funciones es la suma de la espereza de cada función por separado: $E(f(x)+g(x))=E(f(x)) + E(g(x))$.
- La esperanza del producto de funciones en las variables $x,y$ es el producto de las esperanzas de las funciones en las variables $x,y$, **SIEMPRE Y CUANDO** $x, y$ sean variables independientes: $E(f(x)\cdot g(y))=E(f(x))\cdot E(g(y))$. 