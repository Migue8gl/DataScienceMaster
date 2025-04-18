#doc | #product | #algebra

- El *dot product*, también conocido como producto escalar, es una operación fundamental en álgebra lineal.
- Puede interpretarse el escalar resultante como una medida de similaridad entre [[vector|vectores]]. Si es positivo tienen direcciones parecidas, cero sin son perpendiculares y negativo si tienen direcciones opuestas.
## Definición
Dado dos vectores **a** y **b** en un espacio euclidiano, el producto punto entre ellos se define como:
$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i \cdot b_i$$
Donde:
- $\mathbf{a} = [a_1, a_2, ..., a_n]$ es el primer vector.
- $\mathbf{b} = [b_1, b_2, ..., b_n]$ es el segundo vector.
- $n$ es la dimensión de los vectores.
## Propiedades
El producto punto tiene varias propiedades útiles:
1. **Conmutatividad**: $\mathbf{a} \cdot \mathbf{b} = \mathbf{b} \cdot \mathbf{a}$
2. **Distributividad sobre la suma**: $\mathbf{a} \cdot (\mathbf{b} + \mathbf{c}) = \mathbf{a} \cdot \mathbf{b} + \mathbf{a} \cdot \mathbf{c}$
3. **Asociatividad con escalares**: $(k \mathbf{a}) \cdot \mathbf{b} = k (\mathbf{a} \cdot \mathbf{b})$
4. **Producto escalar de un vector consigo mismo**: $\mathbf{a} \cdot \mathbf{a} = \| \mathbf{a} \|^2$, donde $\| \mathbf{a} \|$ es la norma euclidiana de $\mathbf{a}$.
## Aplicaciones
- Cálculo de ángulos entre vectores.
- Proyección de un vector sobre otro.
- Resolución de sistemas de ecuaciones lineales.