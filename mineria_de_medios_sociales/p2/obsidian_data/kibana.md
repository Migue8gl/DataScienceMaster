- #kibana 

- Podemos descargarlo desde: [download kibana](https://www.elastic.co/downloads/kibana).

### Inicio
- Primero debemos correr el servicio `bin/kibana`. Acto seguido podremos acceder desde la url de `localhost:5601`.
- Nos aparecerá una pantalla en la que podremos introducir el token generado por [[elastic]] o bien usuario y contraseña.
- Una vez cargado y completado podremos acceder a todas las opciones de kibana.

### Dev tools
- Dentro de dev tool podremos crear peticiones a elastic de manera fácil y sencilla, por ejemplo, si queremos acceder a nuestros [[index|índices]] podemos hacer una petición get del estilo:

![[kibana-get-index.png]]

- Las peticiones del estilo `get /` atacan el [[endpoint]] de elastic. Dentro de esa funcionalidad podremos acceder a multitud de APIs, como por ejemplo #cat (ejemplo anterior).
- Otro ejemplo de petición en el que mostramos los nodos disponibles.

![[kibana-get-nodes.png]]