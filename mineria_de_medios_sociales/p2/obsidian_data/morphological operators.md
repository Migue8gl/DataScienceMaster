- #cv | #morphological

# Contexto
- Las operaciones morfológicas son operaciones que suelen realizarse sobre imágenes binarias y se basan en la forma de la imagen. 
- La operación viene determinada por lo que se conoce como un **elemento estructural**. 
- Todas las operaciones morfológicas se derivan de las operaciones base, que son erosión y dilatación.
# Erosión
- Esta operación erosiona las fronteras del objeto en primer plano. Para ello el elemento estructural se desplaza por la imagen. Si se encuentra un $1$ este será considerado si y solo si todos los píxeles bajo el kernel están a $1$, en cualquier otro caso se ponen a $0$.
- Los píxeles de la frontera del objeto son puestos a cero, recortando la forma del objeto.

# Dilatación
- Es la operación contraria a la erosión. En la imagen se salida hay un $1$ si al menos un píxel bajo el elemento estructural es $1$. Esta operación extiende las fronteras de un objeto.
- Se puede obtener una frontera exterior si a la dilatación se le resta la imagen original. También se puede obtener una frontera interior si a la imagen original se le resta la imgen erosionada.
![[morphological operators-edges.png|540]]

# Apertura
- Resultado de hacer una erosión y a continuación una dilatación. Es un proceso muy simple que ayuda a eliminar ruido de la imagen.
![[morphological operators-opening.png|540]]
- Si se tienen dos regiones conectadas por un puente débil la apertura obtendrá la separación de las dos regiones.

# Clausura
- Si se concatena una dilatación con una erosión, se obtiene la clausura. Es útil cuando se pretende tapar agujeros.

# Black Hat y Top Hat
- *Black Hat* es la diferencia entre la clausura y la imagen, mientras que *Top Hat* es la diferencia entre la imagen y la apertura.
- *Black Hat* obtiene una imagen con objetos más pequeños y más oscuros que el entorno. *Top Hat* obtiene una imagen con objetos más pequeños y más claros que el entorno.
- Es decir, *Black Hat* enfatiza los detalles más pequeños y oscuros, mientras que *Top Hat* enfatiza los detalles más pequeños y claros.
![[morphological operators-black-top-hat.png]]

# Esqueleto
- Es la representación minimal de un objeto. Un punto de la imagen original estará en el esqueleto si se pierde al realizar una apertura. Es decir si erosionamos y luego al dilatar no recuperamos ese punto. El número de veces que se aplica este proceso es el numero de veces que podemos erosionar la imagen original y no obtenemos la imagen negra (sin objetos).