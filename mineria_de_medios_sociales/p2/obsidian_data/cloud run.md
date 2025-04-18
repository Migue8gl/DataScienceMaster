- #googlecloud | #gcp 

# Concepto
- *Cloud Run* es una plataforma de procesamiento administrada que te permite ejecutar contenedores directamente sobre la infraestructura escalable de *Google*.
- En *Cloud Run* el código [[docker|dockerizado]] se ejecuta como un servicio o como un trabajo:
	- Servicio es cuando se usa para responder a solicitudes *web* o eventos.
	![[cloud run-service.png|500]]
	- Trabajo es cuando el código ejecuta un trabajo específico y este se cierra cuando dicho trabajo es finalizado.
	![[cloud run-work.png|500]]
# Servicios
- Un servicio de *Cloud Run* te proporciona la infraestructura necesaria para ejecutar un extremo [[https|HTTPS]] confiable. Tu responsabilidad es asegurarte de que tu código escuche en un puerto **TCP** y maneje las solicitudes **HTTP**.
- Ofrece escalado automático basado en solicitudes, es decir, puede escalar horizontalmente en base a la solicitudes.
- Cada **implementación** crea una nueva revisión inmutable. Puedes enrutar el tráfico entrante a la revisión más reciente, revertir a una revisión anterior o dividir el tráfico a varias revisiones al mismo tiempo [para realizar un lanzamiento gradual](https://cloud.google.com/run/docs/rollouts-rollbacks-traffic-migration?hl=es-419). Esto es útil si deseas reducir el riesgo de implementar una revisión nueva. Puedes comenzar con el envío del $1\%$ de las solicitudes a una revisión nueva y aumentar ese porcentaje mientras supervisas la telemetría.
- Ofrece posibilidad de gestionar el servicio como privado o público especificando una política de acceso con [[cloud iam]].
# CI/CD
- Se puede configurar un *Cloud Run* para la [implementación continua](https://cloud.google.com/run/docs/continuous-deployment-with-cloud-build?hl=es-419#new-service) si el código del proyecto está alojado en un repositorio de [[git]]. De esta forma, con cada *release* se activa una nueva **implementación**.
# Deployment
- Lo primero que debemos hacer es habilitad **google cloud build API**. Una vez hecho nos vamos a *settings* y habilitamos permisos para **google cloud run** y para **service account** (esta última para poder administrar permisos).
- Habilitados **google cloud run**.
- Nos vamos a **IAM & Admin** y creamos una nueva cuenta de servicio. Después le damos permiso de *Cloud Build Service Agent* y *Cloud Build Editor*. Creamos la cuenta y añadimos una clave nueva, esa clave debemos guardarla y evitar que se filtre. **OPCIONAL SI TU PROJECTO EN GCP YA ESTÁ CONFIGURADO.**
- Lo siguiente es configurar el archivo de configuración de *cloud build*:
```yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    entrypoint: 'bash'
    args: [ '-c', 'docker pull europe-southwest1-docker.pkg.dev/inteligencia-artificial-aicore/elastic-indexer/elastic-indexer:latest || exit 0' ]
  # Build the container image
  - name: 'gcr.io/cloud-builders/docker'
    args: [
      'build',
      # Try to reuse the cached layers from previous builds
      '--cache-from', 'europe-southwest1-docker.pkg.dev/inteligencia-artificial-aicore/elastic-indexer/elastic-indexer:latest',
      '-t', 'europe-southwest1-docker.pkg.dev/inteligencia-artificial-aicore/elastic-indexer/elastic-indexer:latest',
      '.'
    ]
  # Push the container image to Artifact Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'europe-southwest1-docker.pkg.dev/inteligencia-artificial-aicore/elastic-indexer/elastic-indexer:latest']
  # Deploy container image to Cloud Run
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - 'elastic-indexer'
      - '--image'
      - 'europe-southwest1-docker.pkg.dev/inteligencia-artificial-aicore/elastic-indexer/elastic-indexer:latest'
      - '--region'
      - 'europe-southwest1'
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'services'
      - 'update-traffic'
      - 'elastic-indexer'
      - '--to-latest'
      - '--region'
      - 'europe-southwest1'
images:
  - 'europe-southwest1-docker.pkg.dev/inteligencia-artificial-aicore/elastic-indexer/elastic-indexer:latest'
```
- Después en *cloud build* podemos configurar un repositorio (de **Gitlab** por ejemplo) y vincularlo a un activador. Si todo está correcto el activador montará el *cloud run* a partir del activador.
- Algunas cosas necesitarán de configuración extra, como las variables del entorno. En el *cloud run* del proyecto entramos en configuración y añadimos un volúmen con el secreto del *secret manager* para poder vincularlo. 
- Con las bases de datos igual, en el *cloud run* debe configurarse para poder tener acceso.