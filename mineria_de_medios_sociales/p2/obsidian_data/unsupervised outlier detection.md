- #outliersdetection | #unsupervised

# Métodos gráficos
- *Plottear* la base de datos $D$ y encontrar el punto alejado que catagorizaremos de anomalía.
- Usando [[biplot]] es posible resumir las variables de un conjunto de datos de forma que la pérdida de información sea mínima (usando [[pca]]) y además detectar visualmente puntos anómalos, gracias a valores extremos en algunas variables.
- Si algún punto anómalo está cerca de la nube principal generada por el biplot, es posible que sea por una combinación de valores extraños en varias variables (los más interesantes).
# Métodos estadísticos
- Hacen [[hypothesis testing|tests de hipótesis]] para determinar si un punto 
  $x$ en un conjunto $D$ es una anomalía con una probabilidad $p$.
- [[grubbs test]] (una anomalía)
- [[tietjen-moore test]] ($k$ anomalías exactas)
- [[rosner test]] (menos de $k$ anomalías)
## Mahalanobis
- La distancia [[distances|Mahalanobis]] es una versión multidimensional del [[z-score]]. Esta mide la distancia de un punto con respecto al centroide de una distribución [[normal distribution|normal]]. Considerando esta distancia se plantea el siguiente test:
	- $H_{0}=\text{No hay outliers}=x_{highest}\sim N(\mu,\sum)$ (el más alto usando Mah.distancia)
	- $H_{1}=\text{Ese valor no sigue la distribución, es un outlier}$ 
- En el caso de usar varios test de hipótesis sobre varios puntos para ver si siguen la distribución, entonces hay que realizar una penalización para controlar [[fwer]].
- No es un método muy robusto, costaría mucho rechazarlo. Se suele hacer para mirar si hay $k$ o menos.
## Distancia intercuartil
- Se aplica cuando los datos siguen una distribución normal u de otro tipo con colas más largas. Nunca con distribuciones de varios picos.
- Un valor $p$ se dice *outlier* si está por encima o por debajo de la distancia intercuartil multiplicada por $1.5$-> $IQR = Q3-Q1$. Si está por encima o por debajo de esta distancia multiplicada por $3$, es un *outlier* extremo.
![[outlier detection-idr.png|200]]
- Todos los métodos anteriores se basan en métricas poco robustas como la media o varianzas, por lo que hay que identificar alguna métrica más robusta.
## Métricas robustas
- Algunas son la [[median|mediana]] o la media recortada, es decir, quitando un $x\%$ de aquellos valores más extremos.
- Algunas limitaciones usando métodos de distancias es el [[curse of dimensionality|problema de la dimensionalidad]].  
- Un estimador robusto es **MCD** (*minimum covariance determinant*), en el cual se tienen en cuenta la [[covariance|covarianza]] de la [[sample|muestra]] de los puntos "buenos".
- Con Mahalanobis además, al crear elipses, no se capturan relaciones no lineales. También puede haber una mezcla de distribuciones, por las cuales se discriminan muchos puntos que no deberían ser outliers al estar teniendose en cuenta solo una distribución.
![[outlier detection-limitation-mahalanobis.png|500]]
- Las principales limitaciones de los métodos estadísticos son que los datos no suelen seguir una distribución normal y que los datos multivariantes en muchas dimensiones no suelen seguir niguna distribución conocida.

# Métodos de distancias
## Vecinos más cercanos
- Se pueden utilizar los [[k-nearest neighbors]] para asignar grados de anomalías a los puntos en el espacio. De esta forma se pueden utilizar algún umbral para detectar anomalías basandose en es grado.
- El grado de un punto es la distancia de ese punto con su $k$-ésimo vecino más cercano. También se puede hacer la media de las $k$ distancias.
- Si se usase la media de las $k$ distancias de vecinos para calcular el grado de anomalía, entonces puede ocurrir que este valor medio no sea significativo de por si, habría que comparar ese valor con las medias de los otros $k$ vecinos -> **densidad**.
- El valor de $k$ es problemático.
## Densidad
- Se defina la densidad de un punto como la inversa de la media de la distancias a sus vecinos más cercanos.
- Se define la densidad relativa de un punto como el ratio entre la densidad del punto y la media de las densidades de los $k$ vecinos -> Grado de *outlier* (**LOF**).
- Es un algoritmo muy costoso.

# Métodos de clustering
- Una posibilidad es clasificar aquellos objetos que por medio de [[clustering]] no caen en ningún *cluster*. Lo malo es que se puede etiquetar muchos puntos como *outlier*.
- Se usan pues, métodos que crean clusters para todos los puntos. El grado de anomalía es la distancia al **centroide**.
- Como en **LOF** también es posible corregir de forma local este valor de anomalía usando la mediana (para que valores extremos no afecten a la métrica) de las distancias de todos los puntos con respecto al centroide para corregir el grado de cada punto -> $g_{i}=\frac{d(x_{i})}{median(D_{i})}$, donde $d$ es la distancia de $x_i$ al centroide y $D_{i}$ son todos los puntos que conforman el cluster.
- Trabajar con $k$ es problemático y normalmente suele ser mejor trabajar con un número mayor de $k$. Es más fiable un *outlier* en múltiples grupos que un *outlier* en pocos grupos.