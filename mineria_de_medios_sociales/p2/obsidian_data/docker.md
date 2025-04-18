- #docker

- [Tutorial](https://www.youtube.com/watch?v=4Dko5W96WHg)

# Instalación linux
- [docker linux](https://docs.docker.com/engine/install/ubuntu/)
- [docker desktop linux](https://docs.docker.com/desktop/install/ubuntu/)
# Teoría
- Un contenedor es una forma de empaquetar aplicaciones junto con sus dependencias y archivos de configuración.
- Son modos de hacer portable un proyecto.
- Se almacenan en repositorios de contenedores.
- Se descarga una imagen basada en Linux. Corriendo un comando, ya se puede usar la aplicación.
- Una imagen contiene las dependencias, el código, las configuraciones. El contenedor son capas de imágenes, siendo la primera y más profunda una distribución de Linux y la más elevada nuestra aplicación. Los contenedores suelen ser muy livianos.
## Docker Desktop
- Virtual machine:
	- Corre Linux y ejecuta containers.
- Permite acceder al sistema de archivos y a la red.
- Docker Compose y otras herramientas.
## Docker Hub
- [docker hub](https://hub.docker.com/)
- Tiene imágenes oficiales de muchas herramientas.
## Comandos
- `docker images` -> lista de imágenes.
- `docker pull <img-name>` -> descargar imágen de docker hub.
- `docker compose build` -> sirve para construir imágenes de múltiples servicios definidos en un archivo `docker-compose.yml`.
- `docker build` -> para construir una sola imagen a partir de un archivos `Dockerfile`. Usar `--no-cache` para volver a recompilar.
- `docker compose up -d` -> levanta la imagen de docker en segundo plano.
- `docker compose down` -> para la imagen de docker.

# Pasos
- Se debe definir un archivo `Dockerfile` en el cual se especifiquen los comandos necesarios para levantar un contenedor.
- Se especifica por ejemplo:
	- **Imagen base**:
		- `FROM python:3.10-slim-bullseye` ->Este comando indica que la imagen base será `python:3.10-slim-bullseye`, una versión ligera de Debian (Bullseye) con Python $3.10$ preinstalado. La versión "slim" ayuda a reducir el tamaño de la imagen. 
	- **Variables de entorno**:
		- `ENV PYTHONWRITEBYTECODE=1` -> Activa la escritura de bytecode (.pyc), acelerando las ejecuciones posteriores.
		- `ENV PYTHONBUFFERED=1` -> Evita que el buffer de salida de Python se guarde en el caché, lo que facilita la depuración.
		- `ENV PYTHONOPTIMIZE=1` -> Activa optimizaciones en Python, como la eliminación de `assert`.
		- `ENV ACCEPT_EULA=Y` -> Acepta el *EULA* para componentes de *Microsoft* que se instalarán más adelante (como el controlador *ODBC* para *SQL Server*).
	- **Instalación de paquetes del sistema**:
		- `RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add. 
		- `RUN curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list RUN apt-get -y update; apt-get install -y msodbcsql17`.
	- **Definiciones de directorios**:
		- `WORKIR /code` -> Creación del directorio de trabajo.
		- `COPY . /code/` -> Copiar todo el código y directorios.
	- **Instalación de dependencias del proyecto**:
		- `COPY ./requirements.txt /code/`.
		- `RUN pip install --no-cache-dir --upgrade pip`.
		- `RUN pip install --no-cache-dir -r requirements.txt`.
- Se construye la imagen con todas las instrucciones (las anteriores eran de ejemplo, se pueden definir muchos tipos de pasos). Para construir la imagen: `docker build -t <your-image-name>:<tag> .` y para lanzar la imagen: `docker run -p 80:80 <your-image-name>:<tag>`.

# Tips
- [Arreglar timezone en docker](https://hoa.ro/blog/2020-12-08-draft-docker-time-timezone/)