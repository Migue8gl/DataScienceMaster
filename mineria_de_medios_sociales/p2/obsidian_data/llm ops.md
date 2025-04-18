- #ml | #llmops

# Concepto
- Es una especificación de [[ml ops]]. 
- *Prompt desing*.
- *Supervised tuning*.
- Monitoreo y función de evaluación del *output* de la *llm*.

![[llm ops-driven-app.png]]
- LLM ops se centra en la parte de *customization* y el *deploy*.
- En el entorno de GCP podemos usar como repositorio de datos el *DataWarehouse* de **BigQuery**, como fichero de archivos **Bucket**.
	
# Automatización y orquestación
- Como orquestador y automatizador de un **flujo de trabajo** (*workflow*) **Kubeflow Pipelines**.
- **Kubeflow Pipelines** consiste en dos conceptos clave, los componentes y las *pipelines*.
	- Las *pipelines* son conjuntos de código que realizan varios pasos como preprocesado y preentrenamiento.
	- Un componente es una parte aislada del *pipeline* (como en un *container*).
	- Se especifican con decoradores.
	- Se puede compilar la *pipeline* en un archivo de configuración *yaml*.
- Servicios como **VertexAI** pueden ejecutar *pipelines* definidas en *yaml*:
```python
from google.cloud.aiplatform import PipelineJob

job = PipelineJob(
        ### path of the yaml file to execute
        template_path="pipeline.yaml",
        ### name of the pipeline
        display_name=f"deep_learning_ai_pipeline",
        ### pipeline arguments (inputs)
        ### {"recipient": "World!"} for this example
        parameter_values=pipeline_arguments,
        ### region of execution
        location="us-central1",
        ### root is where temporary files are being 
        ### stored by the execution engine
        pipeline_root="./",
)

### submit for execution
job.submit()

### check to see the status of the job
job.state
```

- Balanceo de carga entre modelos y usuarios.
- **VertexAI** tiene atributos de seguridad en las respuestas del modelo.