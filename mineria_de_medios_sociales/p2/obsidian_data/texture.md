- #featureextraction | #texture

# Concepto
- Vista una imagen como una superficie, la textura es una medida de rugosidad. En muchos casos, una región no se caracteriza como un solo nivel de gris, sino por **variaciones locales en la intensidad** de gris.
- También se puede definir la textura como una repetición de elementos llamados *texels*.
- Una textura suele basarse en una función de *mapeo* entre la imagen original y un mapa de texturas, donde cada pixel refleja información sobre la textura de su entorno.
![[texture-map-function.png]]
- A partir del mapa de texturas se extraen los descriptores.
- Algunas opciones a la hora de crear el mapa de texturas son:
	- Rango, [[variance and standard deviation|varianza]], autocorrelación, etc.