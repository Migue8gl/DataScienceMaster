- #lucene

- Apache Lucene es una biblioteca de búsqueda de texto de código abierto, escrita en Java, que proporciona capacidades de búsqueda de texto completas y de alta calidad. Fue desarrollada originalmente por Doug Cutting en 1999 como un proyecto independiente y luego se incorporó a la Apache Software Foundation en 2001.

- Lucene se utiliza comúnmente como un motor de búsqueda de texto completo en aplicaciones de búsqueda en línea, sistemas de recuperación de información, motores de búsqueda de sitios web y otras aplicaciones que requieren búsquedas rápidas y precisas de texto. Lucene también es utilizado como base para otras herramientas de búsqueda y análisis de texto, como #elasticsearch, Solr y Amazon CloudSearch.

- Lucene utiliza un índice invertido para proporcionar búsquedas de texto completo. El índice invertido es una estructura de datos que mapea cada término en el texto de los documentos indexados a una lista de documentos que contienen ese término. Esto permite a Lucene buscar rápidamente documentos que contienen términos específicos.

### Index
- #index
- En el contexto de la búsqueda de información, un índice es una estructura de datos que se utiliza para acelerar la recuperación de información de un conjunto de documentos. En particular, en el caso de Lucene, un índice es una colección de archivos que contienen información de los documentos que se han indexado para realizar búsquedas posteriores.

### Shards
- #shard
- Un shard es una parte del #index que se divide y almacena en diferentes servidores o nodos. En otras palabras, un índice de Lucene puede estar compuesto por varios shards, donde cada shard es una porción independiente del índice que se almacena en un servidor o nodo separado.

- La división del índice en múltiples shards permite la distribución del índice en diferentes servidores, lo que puede mejorar la escalabilidad y el rendimiento del sistema de búsqueda. Además, permite que el índice completo se pueda almacenar en múltiples discos duros, lo que permite una mayor capacidad de almacenamiento.