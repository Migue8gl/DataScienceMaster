- #variational | #autoencoder | #dl

# Concepto
- Los [[autoencoder]] variacionales se diferencian de los tradicionales en la forma en la que codifican las variables latentes. Los **autoencoders** tradicionales son determinísticos y codifican las variables en [[vector|vectores]] de valores discretos.
- Los variacionales en lugar de usar valores fijos discretos $z$, utilizan un rango continuo de posibilidades expresado como una [[probability|distribución de probabilidad]] $p(z)$.
- En estadística [[bayes theorem|Bayesiana]], esta distribución aprendida para una [[latent space|variable latente]] se conoce como *prior distribution*.
- En *inferencia variacional*, los procesos generativos utilizan *prior* para sintetizar nuevos puntos de datos, es decir, se calcula una distribución *posterior* $p(z|x)$.
- Para cada **variable latente** se codifica un vector de [[mean and weighted mean|medias]] $\mu$ y un vector de [[variance and standard deviation|desviaciones estándar]] $\sigma$.
- El problema fundamental con los autoencoders, cuando queremos generar muestras con ellos, es que el espacio latente, en quien convertimos los *inputs* y donde están los vectores codificados, puede no ser continuo y no permitir una interpolación fácil.
- El objetivo de los autoencoders variacionales, sin embargo, no es el de reconstruir un *input*. El objetivo es generar nuevas [[muestras|sample]] que se parezcan a las originales.
- El problema fundamental con los *autoencoders*, cuando queremos generar muestras con ellos, es que el espacio latente, en quien convertimos los *inputs* y donde están los vectores codificados, puede no ser continuo y no permitir una interpolación fácil. En los variacionales, el espacio latente es por diseño continuo (probabilístico, cada normal es una normal con media y desviación).

# Función de pérdida
## MSE
- Se suelen usar funciones de pérdida como [[mean squared error|MSE]] para el objetivo de minimizar el error de reconstrucción.
## Kullback-Leibler divergence (KL)
- Para generar nuevos datos, el descodificador toma muestras del espacio latente. El muestreo de los puntos específicos del espacio latente que representan las entradas originales en los datos de entrenamiento replicaría esas entradas originales. Para generar nuevos datos, la **VAE** debe ser capaz de tomar muestras de cualquier punto del espacio latente entre los puntos de datos originales. Para que esto sea posible, el espacio latente debe presentar dos tipos de regularidad:
	- **Continuidad**: puntos cercanos en el espacio latente deben generar contenido similar cuando son descodificados.
	- **Completitud:** cualquier punto muestreado del espacio latente debe generar contenido con significado al ser descodificado.
- [[distances|KL]] es una métrica de divergencia que se usa para comparar dos distribuciones. Minimizar esta distancia entre las distribuciones de las variables latentes y la [[normal distribution|distribución Gaussiana]] cuyos valores caen entre $[0,1]$ hace que las distribuciones de las variables latentes puedan interpolarse cómodamente.
![[variational autoencoder-using-different-losses.png]]
### ELBO (Evidence lower bound)
- El denominador de **KL** es intratable, es decir, se tardaría infinito tiempo en calcularse. Por ello se trabaja maximizando **ELBO**.
- La evidencia en **ELBO** se refiere a $P(X)$, los datos de entrada observables. El límite inferior se refiere a el peor caso de la estimación de la log-verosimilitud de una distribución dada.
- Se está optimizando una cota inferior de la [[likelihood function|verosimilitud]], no la propia verosimilitud.

# Matemáticas
- Suponemos una serie de datos $X=\{x_{i}\}_{i=1,...,N}$. Cada uno de esos datos son generados por un proceso aleatorio asociado a una variable continua no observable $z$.
	- $z_{i}$ es generado por una distribución a priori $P_{\theta}(z)$.
	- $x_{i}$ es generado por la distribución condicionada $P_\theta(x|z)$.
- Suponemos por simplicidad que la distribución $P_\theta(z)$ no tienen $\theta$ (parámetros) a estimar, por lo que se escribe $P(z)$.
- Normalmente:
$$P(z)=N(z|0,I)\quad P_{\theta}(x|z)=N(x|f_{\theta}(z),\sigma^2I)$$
- Donde $f$ es una red neuronal.
## Encoder y autoencoder
- Se introduce un **modelo de reconocimiento** $Q_{\phi}(z|x)$ que aproximará la verdadera a posteriori $P_{\theta}(z|x)$. Este se conoce como **ENCODER**.
	- Trata de aproximar la distribución de las variables latentes dada la distribución de los datos observados.
- De igual manera, $P_{\theta}(x|z)$ se denomina **DECODER**.
	- Este genera nuevos datos en la dimensión de los datos observados dadas las variables latentes.
- El modelo $P(z)P_{\theta}(x|z)$ es un modelo generativo.
	- Se generan las variables latentes y a partir de esas variables se crean nuevos datos condicionados a la distribución de las variables latentes.
- Los parámetros $\theta$ son parámetros de una red neuronal, como por ejemplo los pesos.
## Verosimilitud marginal
- La verosimilitud marginal de un conjunto de datos expresa como de bien un modelo explica un conjunto de datos.
$$logP_{\theta}(x_1,...,x_{n})=\sum_{i=1}^{N}logP_{\theta}(x_i)$$
- Como calcular este término es intratable, se intenta calcular una cota inferior (**ELBO** o  *Evidence Lower Bound*).
$$logP_{\theta}(x_{i})\ge L(\theta,\phi; x_{i})$$
- Al maximiar la **ELBO** sabemos que siempre estamos mejorando la verosimilitud marginal.
$$\large{L(\theta,\phi;x_{i})=E_{Q_{\phi(z|x)}}}[-logQ_{\phi}(z|x)+logP_{\theta}(x,z)]$$
- Es decir, la cota inferior **ELBO** es la esperanza (bajo la distribución del modelo *encoder*, es decir, considerando que los valores posibles están distribuidos según esa distribución) de la entropia del modelo *encoder* más la probabilidad conjunta del dato observado $x$ y la variable latente $z$ según el modelo generativo.
- Esta cota se puede reescribir como:
$$\large{L(\theta,\phi;x_{i})=-D_{KL}(Q_{\phi}(z|x_{i})||P_{\theta}(z))+E_{Q_\phi(z|x_{i})}[logP{\theta}(x_i|z)]}$$
- Es decir, la distancia [[distances|KL]] negativa más la esperanza bajo la distribución del modelo de reconocimiento (*encoder*) de la verosimilitud condicional de $x_{i}$ condicionada por $z$. Es decir,  la distancia de **KL** en negativa, ya que maximizamos **ELBO** y con ello minimizamos **KL**, más lo bien que se explica el dato observado dada la variable latente $z$.
## Trampa de la reparametrización
- Sea $z$ una variable continua y $z\sim Q_{\phi}(z|x)$. Al optimizar **ELBO**, el proceso de muestreo introduce ruido no derivable.
- Para hacer derivable el muestreo de $z$ respecto a los parámetros $\phi$, se muestrea directamente de una distribución determinística simple como $\epsilon$ (que sigue una normal) y se transforma mediante $G_{\phi}(x,\epsilon)$, que introduce una dependencia en $\phi$:
$$z=G_{\phi}(x, \epsilon)$$
- De forma que:
$$z=\mu_{\phi}(x)+\sigma_{\phi}(x)\cdot \epsilon$$
- Y ahora si es posible diferenciar, ya que $z$ depende de los parámetros $\phi$ en una función derivable.
- **RESUMEN:** El truco de la reparametrización es una forma de reescribir la esperanza de forma que la distribución con respecto a la cual tomamos el gradiente sea independiente del parámetro $\phi$. Esto se debe a que diferenciar la esperanza con respecto a los parámetros de la distribución es un problema **no trivial**.
![[variational autoencoder-reparametrization-trick.png]]