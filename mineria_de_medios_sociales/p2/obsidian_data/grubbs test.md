- #test | #datascience | #univariate | #outliersdetection | #hypothesis 

# Concepto
- El test de **Grubb** es un test estadístico univariable.
- Es usado para detectar [[outliers]].
- Se asume que los datos provienen de una [[normal distribution|distribución normal]].

# Proceso
- Se hace el siguiente planteamiento de [[hypothesis testing|hipótesis]]:
	- $H_{0}: \text{No hay outliers en los datos}$.
	- $H_{1}: \text{Hay exactamente un outlier}$.
## Estadístico de Grubb
$$G=\frac{max_{i=1,..N}|X_{i}-\overline{X}|}{S}$$
- El *outlier* participa en el cálculo de la media y de la desviación estándar $S$.
- Se rechaza $H_{0}$ si:
	- $\large{G > \frac{(N-1)}{\sqrt{N}}\sqrt{\frac{t^2_{\alpha/(2N),N-2}}{N-2+t^2_{\alpha/(2N),N-2}}}}$
- El test de **Grubb** es el valor de desviación absoluto de la media muestral en unidades de la desviación estándar.
- Este test se utiliza para un solo *outlier* en una distribución normal de una sola variable. Si se requiere detectar **múltiples** se hace el test a un punto, si sale *outlier* se elimina y se hace el test al siguiente punto.
- Ocurre el problema del **Masking**, donde un *outlier* hace que falle el test para otros, estos se *esconden* entre ellos (ya que intervienen en el cálculo del estadístico).
![[grubbs test-masking-problem.png|500]]