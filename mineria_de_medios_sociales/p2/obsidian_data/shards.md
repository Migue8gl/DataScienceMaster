- #shards

- En Elasticsearch, un shard es una unidad básica de indexación y búsqueda. Cada shard es esencialmente una parte del [[index]] completo que contiene una porción de los datos indexados.

- Los shards permiten a Elasticsearch dividir un índice en partes más pequeñas y distribuir los datos de manera eficiente en un clúster. Esto mejora el rendimiento y la escalabilidad de Elasticsearch, ya que los datos pueden ser indexados y buscados en paralelo en varios nodos.

### Tipos
- Elasticsearch tiene dos tipos de shards: shards primarios y shards secundarios. Cada índice tiene al menos un shard primario, que es responsable de manejar todas las operaciones de indexación y búsqueda. Los shards secundarios son copias de los shards primarios y se utilizan para la redundancia y la replicación de datos, lo que mejora la disponibilidad y la tolerancia a fallos.
- Un shard réplica y su primario no pueden/deben estar en el mismo nodo, por razones de disponibilidad.

- Cada shard es un índice de Apache Lucene.

![[shards-index-elastic.png]]