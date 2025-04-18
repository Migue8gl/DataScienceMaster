- #localbinarypattern | #featureextraction | #computervision 

# Concepto
- Las etiquetas **LBP** dan información del patrón o disposición de los píxeles del vecindario de cada píxel.
- Al comparar cada vecino con el píxel central en un orden preestablecido se obtiene un código binario cuyo valor en decimal corresponde a la etiqueta.
- Por cada píxel se comparan los valores de celda con sus vecinos (de izquierda arriba a derecha arriba, abajo derecha, izquierda derecha, es decir, *clockwise*).
- Si el valor del píxel central es mayor que el del vecino, se pone 
- $0$. En caso contrario, $1$. Esto da un número binario de $8$ dígitos.
![[local binary patterns-example.png]]
- Repitiendo este proceso en todos los píxeles se obtiene una imagen de códigos **LBP** que será la base del descriptor.
![[local binary patterns-lsb-image.png|550]]
- La imagen **LBP** no se utiliza directamente, se usa el histograma de estos códigos.
- Se obtienen histogramas de $256$ bins porque los códigos tienen valores en el rango $[0,255]$.

# Interpretación
- También se puede entender como que un píxel tiene un vecindario de $8$ píxeles equiespaciados angularmente que están a distancia $1$ (radio).
- Por tanto, dado un radio y un número de vecinos se pueden obtener características a distintas escalas. 
- En algunos casos será necesario interpolar valores o elegir vecino más cercano.
![[local binary patterns-different-radius.png]]

# Variantes
- Una variante del **LBP** consiste en asignar la misma etiqueta a patrones que resulten de un desplazamiento cíclico. De esta forma se introduce invariancia a rotaciones.
![[local binary patterns-rotation-invariance.png]]
- El **LBP uniforme** tiene en cuenta transiciones $0-1$ a la hora de asignar el código al patrón. Si notamos por $U$ al número de transiciones:
	- Se asigna etiqueta distinta a las códigos con $U=0, U=2$ (uniformes).
	- Si tiene $U>2$ son no uniformes.
![[local binary patterns-uniforms.png]]
- Los patrones uniformes son frecuentemente los más comunes en imágenes naturales, como bordes, esquinas y superficies homogéneas.
- Al enfocarse en estos patrones, los **LBP** uniformes proporcionan una representación más **robusta y menos sensible al ruido** en las texturas, lo que los hace más adecuados para distinguir características relevantes en aplicaciones de clasificación.
- Además reducen la longitud del [[vector]] de características e implementan un descriptor invariante a rotaciones.