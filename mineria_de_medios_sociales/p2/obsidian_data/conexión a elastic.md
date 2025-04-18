### Mediante API de Python
- Creamos la instancia de cliente de #elasticsearch mediante el método `ElasticSearch()` (documentación: [elasticSearch client](https://elasticsearch-py.readthedocs.io/en/v8.4.3/api.html#module-elasticsearch)) con los siguientes parámetros:
	- Dirección url del servidor de ElasticSearch.
	- Localización local del certificado http_ca.ctr del servidor.
	- Usuario y contraseña.
- A partir de aquí, podemos realizar cualquier operación con esa instancia del cliente.

### Mediante el navegador web
- Comprobamos que el servicio está activo, para ello accedemos a la url `localhost` en el puerto 9200, que es donde escucha el servidor de ElasticSearch.
	- Introducimos contraseña y usuario.
- Nos muestra un json con la información del servidor.
- En segundo lugar debemos conectarnos a #kibana, que es donde realmente podemos trabajar con el servidor de ElasticSearch. Este escucha en el puerto 5601, de nuevo introducimos contraseña y usuario.