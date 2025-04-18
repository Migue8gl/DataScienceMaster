- #pythonelastic 

## Conexión
- Para conectarnos a Elasticsearch debemos [[conexión a elastic|crear una instancia del cliente]]. Después podemos hacer ping para comprobar que todo funciona correctamente.
- La contraseña y el usuario se generan al iniciar elastic por primera vez.
- Podemos resetear la contraseña con `bin/elastic-search-reset-password -u "usuario"`.
- También se nos da un token de acceso a [[kibana]] válido para los próximos 30 minutos (si no lo aplicamos habrá que generar otro).

![[python-elastic-conection.png]]

## Indices
### Primeros pasos
- Para crear nuestro primer [[index|índice]] tendremos que introducir la siguiente instrucción:
	- `es.indices.create(index="nombre_indice")`
- Para poder visualizar el índice creado iremos a [[kibana#Dev tools|kibana]] y ejecutaremos una petición para ver el listado de índices disponibles:

![[elastic-index-created-example-kibana.png]]

- También podemos mostrarlos por consola con Python:

![[show-indices-python-elastic.png]]
![[indices-example-terminal.png]]

### Creación de índices en secuencia
- Podemos crear varios índices en masa:

![[python-index-bulk-creation.png]]
![[index-bulk-creation-result.png]]

### Creación de índices a través de un input
- Otra forma de crear índices en elastic podría ser a través de un fichero de entrada, con los datos necesarios para ser almacenados. Un ejemplo en Python bastante simple podría ser el siguiente:

![[yaml-example-data.png]]
![[python-yaml-parser-function.png]]
![[data-yaml-create-index-elastic.png]]

- En este ejemplo además hemos creado lo que se conoce como documentos, su equivalente en una base de datos relacional sería una fila. Hemos insertado la fila del empleado Miguel y la empleada Martita.

### Búsqueda de índices
- Podemos hacer búsqueda de índices con la función `search`.

![[seach-function-elastic-search.png]]
![[terminal-output-elastic-index-search.png]]

- Nos devuelve varios datos, como el tiempo de ejecución de búsqueda, si ha dado time-out, el total de [[shards]], aquellas que han fallado en recuperarse y aquellas exitosas. 
- Después de muchos datos de la consulta, obtenemos el índice en cuestión y todos sus datos asociados en el formato asociado.
- Usando cURL habríamos tenido que hacer una [[http requests|petición]] del siguiente estilo:
	- `PUT indexname/_doc/docname`.
- Donde doc es el tipo del documento (solo existe el tipo doc desde la versión 5 de elastic) y indexname y docname los nombres del índice y el documento a crear. Se abrirían llaves y se introducirían los datos.

## Búsquedas
### Búsqueda texto completo
- Para hacer una búsqueda por matching debemos crear una query, que tendrá un formato de diccionario. Esta contendrá el tipo de búsqueda, en este ejemplo será match, y el campo donde queremos buscar.
- En la instancia de ElasticSearch solo deberemos indicar el índice y pasarle la query.

![[query-text-based-elastic.png]]
![[terminal-queryelastic-result.png]]

- Obtenemos los resultados de aquellos campos en los que aparezca la palabra buscada.

## Salud
- [[elastic-health]]
- Para obtener la salud del cluster con Python podemos escribir el siguiente código:

![[get-cluster-health-python-elastic.png]]

- En nuestro caso, al existir un único nodo y tener la restricción de que un shard primario no puede estar en el mismo nodo que su secundario ([[shards#Tipos|restricciones shards]]), obtendremos que la salud del clúster es amarilla o **Yellow**.
- Si obtenemos los datos de nuestros shards, veremos que la réplica del shard creado para `employee` tiene la siguiente información:

 ![[info-example-shards.png]]
- El shard réplica no se asigna, eso es porque tenemos solo un nodo y en ese se guarda el principal. Automáticamente se reasignará a otro nodo cuando se cree.

## Nodos
- Dentro de nuestro cluster, en elastic tenemos varios nodos (instancias de elastic que se ejecutan en una máquina).
- Para poder ver cuantos nodos tenemos en nuestro cluster podemos hacer lo siguiente.

![[node-stats-python-elastic.png]]
- De esta forma podemos obtener los datos de nuestros nodos. 
- Nosotros para las pruebas realizadas solo tenemos un nodo.

![[node-info-output-terminal.png]]

- Vamos a correr otra instancia de Elastic para crear un nuevo nodo. Le pondremos como nombre "otroNodo" y estará bajo el mismo clúster.

![[node-info-elastic-9200-port.png]]

- Obtengamos la información de los nodos con Python:

