- #gcp | #googlecloud | #functions 
- **Links**:
	- [Dependencies python](https://cloud.google.com/functions/docs/writing/specifying-dependencies-python?hl=es-419)
	- [Operadores de airflow](https://airflow.apache.org/docs/apache-airflow-providers-google/stable/_api/airflow/providers/google/cloud/operators/functions/index.html#airflow.providers.google.cloud.operators.functions.CloudFunctionDeployFunctionOperator)
	- [SimpleHttpOperator](https://airflow.apache.org/docs/apache-airflow-providers-http/stable/_api/airflow/providers/http/operators/http/index.html#airflow.providers.http.operators.http.SimpleHttpOperator)
	- [Paso de datos de SimpleHttpOperator a GCF](https://stackoverflow.com/questions/74063284/pass-data-with-simplehttpoperator-to-trigger-cloud-function-2nd-gen)

# Concepto
- *Google Cloud Functions* es un servicio de computación sin servidor que permite ejecutar código en respuesta a eventos específicos sin necesidad de gestionar la infraestructura subyacente.
## Beneficios
- **Sin servidor**: No es necesario aprovisionar ni administrar servidores. Solo se paga por el tiempo de ejecución del código.
- **Escalabilidad automática**: Se escala automáticamente en función de la demanda. Si hay más eventos, se pueden ejecutar más instancias de la función simultáneamente.
- **Integración con otros servicios**: Se puede integrar fácilmente con otros productos de *Google Cloud*, como *Google Cloud Storage*, *Pub/Sub* y más.
- **Desarrollo basado en eventos**: Se puede configurar para que ejecute código en respuesta a eventos, como la carga de archivos en *Cloud Storage*, mensajes en *Pub/Sub*, o solicitudes *HTTP*.
- **Lenguajes compatibles**: Soporta varios lenguajes de programación, incluyendo Node.js, Python, Go, y Java.
## Dependencias
- Para empaquetar las dependencias en Python se utiliza el fichero `requirements.txt`. Si existen dependencias locales se pueden empaquetar en un directorio concreto que sea tratado como un paquete.
```python
myfunction/
├── main.py
└── localpackage/
    ├── __init__.py
    └── script.py
```

# Airflow 
- *Google Cloud Function* tiene soporte en [[airflow]] por medio de la [[rest api|API]] que implementan. Para instalar lo necesario se requiere correr: `pip install 'apache-airflow[google]'`.
- `CloudFunctionInvokeFunctionOperator` es el operador por defecto para invocar una funcionalidad de *GCF*. El tráfico para esta función está muy restringido, por lo que solo se admite para propósitos de *testing*.
- Por ello es mejor usar el operador `SimpleHttpOperator` y llamar a la función por la **API**.
```python
# Fetch the project ID from environment variables
project_id = env_vars["project_id"]

# Define parameters for GCP connection and Cloud Function
gcp_conn_id = "sa-med-ml"  # Service account ID for GCP connection
location = "us-central1"    # Location of the Cloud Function
function_id = "cf_create_eta_automl_dataset"  # Cloud Function name

# Create an operator to invoke the Cloud Function
invoke_cf = CloudFunctionInvokeFunctionOperator(
    task_id="invoke_cf",              # Unique task ID
    project_id=project_id,            # GCP project ID
    location=location,                # Location of the Cloud Function
    gcp_conn_id=gcp_conn_id,          # GCP connection ID
    input_data={"data": payload},      # Input data for the Cloud Function
    function_id=function_id,           # Cloud Function ID
    dag=dag                            # DAG reference
)
```

# ¿Múltiple endpoints?
- [Pregunta al respecto en stackoverflow](https://stackoverflow.com/questions/75877643/how-to-implement-more-than-one-endpoint-in-a-google-cloud-function)
-  La recomendación contra el uso de Google Cloud Functions para múltiples *endpoints* se basa en su diseño como función de propósito único. En el [artículo de Guillaume Blaquiere](https://medium.com/google-cloud/hack-use-cloud-functions-as-a-webserver-with-golang-42edc7935247), se menciona que Cloud Functions, especialmente en su tiempo de ejecución V1, está estructurada para gestionar un único punto de entrada por función, lo que limita su idoneidad para gestionar múltiples rutas o puntos finales de forma eficiente. 
- El tiempo de ejecución V1 impone el procesamiento de una única instancia por solicitud, lo que provoca posibles problemas de rendimiento y costes cuando se gestionan solicitudes concurrentes. Esto puede provocar un aumento de los tiempos de arranque en frío, mayores costes y dificultades para gestionar recursos estáticos. Aunque V2 (impulsado por Cloud Run) mejora la gestión de la concurrencia, Cloud Run en sí se recomienda generalmente sobre Cloud Functions para múltiples puntos finales, ya que proporciona una mayor flexibilidad, pruebas locales, y la productividad del desarrollo sin las limitaciones o soluciones "hacky" necesarios para emular múltiples puntos finales en Cloud Functions.
- Aun así, en este [artículo](https://medium.com/google-cloud/use-multiple-paths-in-cloud-functions-python-and-flask-fc6780e560d3) se enseña como hacerlo.