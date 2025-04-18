- #googlecloud | #bigquery | #dataanalyst

-  **BigQuery** es un depósito/almacen de datos. Esto significa que organiza y almacena datos estructurados que pueden ser usados para análisis y consultas avanzadas.
- Los **Data Sources** vuelcan los datos en un **Data Lake**, el cual almacena esos datos *crudos*, sin procesar ni transformar. Después, tras un proceso de extracción, transformación y carga (**ETL**), estos son almacenados en BigQuery.
![[bigquery-pipline-example.png|600]]
- BigQuery está optimizado para correr consultas enormes, de forma que se pueda analizar información en tiempo real.
- Provee almacenaje y análisis de datos, no necesita de recursos (**serverless**).
- Provee encriptación de datos y herramientas para machine learning.
### Almacenaje
-  El servicio de almacenaje de BigQuery organiza las tablas de datos en unidades llamadas *datasets*, los cuáles están limitados dentro de un proyecto de Google Cloud. Para referenciar a una tabla debe hacer -> $project.dataset.table$. 
- El servicio soporta ingesta de datos en masa (**bulk**) o ingesta de datos en tiempo real (**streaming**).
- Se pueden hacer consultas a través de la [[rest api|REST API]], a través del comando de interfaz **bq** o a través de la UI de la web dentro del proyecto.
- Cada columna de una tabla se guarda por separado, comprimida, encriptada y replicada. No es necesario el uso de keys, índices etc.
- BigQuery, de forma parecida a [[elastic|Elastic]], divide las tablas en [[shards|fragmentos]] para poder hacer consultas paralelamente.
### BigQuery level schema
![[big-query-schema.png|600]]
### Table level schema
![[table-gcloud-schema.png]]
### Consultas
- El servicio de consultas puede hacer consultas a otros archivos (csv, google sheets) que se encuentren en otros tipos de almacénes externos.
### Google cloud data pipeline
![[data-google-pipeline.png|600]]

