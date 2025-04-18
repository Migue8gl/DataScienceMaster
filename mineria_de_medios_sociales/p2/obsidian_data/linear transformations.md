- #linealtransformations | #transformacioneslineales

### Vectores base
- Los [[vector|vectores]] base se utilizan para describir transformaciones de otros vectores. 
- Estos se escriben como $\hat{i}$ y $\hat{j}$. Al aumentar dimensiones también aumentas las variables, pero para dos dimensiones tenemos estos dos vectores.
- Los vectores base se expresan como una matrix $2x2$ donde la primera columna es $\hat{i}$ y la segunda $\hat{j}$.
	- $\hat{i}=\begin{bmatrix}1\\ 0 \\\end{bmatrix}$
	- $\hat{j} = \begin{bmatrix}0 \\ 1 \\ \end{bmatrix}$
	- $basis =\begin{bmatrix}1 \ 0\\ 0 \ 1\\\end{bmatrix}$
- Una **matriz** es una forma de empaquetar varios vectores en una sola colección, esta tiene varias filas y varias columnas.
- Podemos usar los vectores base de apoyo para crear cualquier vector solamente escalandolos y sumándolos.

![[basis-vectors-example.png|400]]

- Una **transformación lineal** es aquella que se consigue por medio de transformar un vector con *estiramiento*, *aplastamiento*, *recorte* o *rotación* mediante el seguimiento de los movimientos del vector base.

![[lineal-transformations.png|600]]

- Las rotaciones giran el espacio del vector.
- Estirar o aplastar un vector puede conseguirse mediante una transformación de escalado.
- El "espejo" o inversión cambia los ejes de forma que $\hat{i}$ y $\hat{j}$ cambian sus sitios.

### Multiplicación matriz - vector
- Podemos usar matrices de transformación para aplicar esa transformación a un vector dado.
	- $\begin{bmatrix} x_{\text{new}} \\ y_{\text{new}} \end{bmatrix} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}$
	- $\begin{bmatrix} x_{\text{new}} \\ y_{\text{new}} \end{bmatrix} = \begin{bmatrix} ax + by \\ cx + dy \end{bmatrix}$
- Las dos primeras columnas de nuestra matriz de transformación son nuestros vectores base.
- Es una manera sencilla de empaquetar vectores base y aplicar transformaciones lineales a vectores.