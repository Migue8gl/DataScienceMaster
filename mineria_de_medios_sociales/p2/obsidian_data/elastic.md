- #elasticsearch | #elastic

- [[elastic errores]]

### ¿Qué es elastic?
- Elasticsearch es un motor de búsqueda y analítica distribuido, gratuito y abierto para todos los tipos de datos, incluidos textuales, numéricos, geoespaciales, estructurados y no estructurados. Elasticsearch está desarrollado a partir de Apache Lucene.
- Elastic funciona mediante [[http requests]] que buscan en documentos según un [[index]] establecido. Cada índice se decompone a su vez en [[shards]], los cuales son internamente índices de [[lucene]]. Estos se guardan en nodos, varios nodos establecen a su vez un cluster. Cada shard puede tener sus réplicas, de forma que existen shards primarios y secundarios, siendo los primarios los shards originales y los secundarios las réplicas. Obviamente, una réplica no puede estar en el mismo nodo que su shard primario.

![[elastic-search-cluster-image.png]]
### Elastic Stack
- Los componentes de este conjunto son el propio #elasticsearch, #logstash y #kibana:
	- #elasticsearch: Motor de búsqueda de texto y análisis, núcleo de elasticstack.
	- #logstash: Herramienta de extracción, transformación y cargado de información.
	- #beats: Pequeños agentes que permiten leer información de métricas, logs etc. Se puede hacer a partir de #logstash.
	- [[kibana]]: Nos permite representar de forma visual la información dentro de #elasticsearch.
 ![[elasticstack.webp]]
### ¿Cómo conectarse a la API?
- [[conexión a elastic]]
### Documentación
-  [documentación](https://www.elastic.co/guide/en/elasticsearch/reference/8.4/docs.html)
### Xpack
- [[xpack]] -> aporta multitud de funcionalidades a elastic.
### Elastic Search
- #elasticsearch 
- Motor de análisis de datos en tiempo real, basado en #apachelucene. Se trabaja en documentos JSON (sin esquema rígido). Se realizan búsquedas mediante elasticDSL
- No tiene esquema: es decir, no estructura sus datos ni estricta sus datos.
- Full-text: Realiza búsquedas en texto completo, es decir, busca en todos los términos ( #term ) de todos los documentos de la base de datos.
- Análisis en tiempo real.
- Múltiples clients y uso de [[rest api]]. 
- Escalable: Puede ejecutarse en un solo nodo e ir escalando a cientos de ellos. Funciona "out of the box", que se refiere a la funcionalidad o características predefinidas que vienen incluidas en la instalación predeterminada del software de ElasticSearch, sin necesidad de realizar ninguna configuración adicional.
- #elasticsearch escucha en el puerto **9200**.
### Lucene
- [[lucene]] | #lucene
### Integración Python
- [[elastic python]] | #pythonelastic 
### Otros
- [[policies]] 
- [[snapshots]]
- [[elastic queries]]