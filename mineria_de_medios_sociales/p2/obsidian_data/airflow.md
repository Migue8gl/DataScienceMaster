- #airflow 

#  Introducción
- **Airflow** es la herramienta estándar y _open-source_ para la orquestración de datos. Cuando hablamos de orquestración de datos nos referimos a la automatización, coordinación y gestión de flujos de trabajo que involucran tareas con datos, como es el caso de las **ETL** (ingesta, transformación y carga de datos).
- Caen bajo el paraguas de la orquestración tareas como el monitoreo, manejo de errores, **ETL** y demás tipos de procesos.
- **Airflow** es una herramienta totalmente agnostica, lo que quiere decir que puede usarse con cualquier servicio externo sin cambiar la capa de orquestración.
- **Apache Airflow** nos ayuda a controlar mediante programación nuestros flujos de trabajo en Python estableciendo dependencias de tareas y monitorizando tareas dentro de cada **DAG** en una interfaz web.

# Componentes principales
## DAG
- Un **DAG** es un grafo dirigido acíclico. Es un grafo porque presenta una estructura que representa relaciones entre sus nodos, es dirigido ya que las aristas que conectan los nodos tienen una dirección definida, no se puede recorrer en ambas direcciones, y es acíclico porque dado un nodo no se puede volver a él por ningún camino, no hay ciclos dentro de la estructura.
- Es la estructura principal de **Airflow**. Este grafo representa las tareas o **Tasks** como nodos, donde las aristas dirigidas definen las dependencias entre tareas.
![[airflow-dag1.png]]
![[airflow-dag2.png]]
- Básicamente, el **DAG** actua como una configuración de las tareas a ejecutarse. A continuación un ejemplo muy básico de **DAG**.
```python
 import datetime

 from airflow import DAG
 from airflow.operators.empty import EmptyOperator

 with DAG(
     dag_id="my_dag_name",
     start_date=datetime.datetime(2021, 1, 1),
     schedule="@daily",
 ):
     EmptyOperator(task_id="task")
```
- Existen tres formas de declarar un **DAG**, usando una declaración con with (gestor de contexto), declarando un objeto o con un decorador.
- Las dependencias entre tareas se declaran se las siguientes formas:
```python
# PRIMERA FORMA
first_task >> [second_task, third_task]
third_task << fourth_task

# SEGUNDA FORMA
first_task.set_downstream([second_task, third_task])
third_task.set_upstream(fourth_task)
```
- Existen por supuesto otras formas con las funciones de chain, cross_downstream, etc. Como opinión personal, la mejor manera y más limpia es con los operadores `<<` y `>>`.
- **Airflow** carga los **DAGs** desde la carpeta configurada para ello, la _DAG_FOLDER_. Esto puede configurarse desde el archivo de configuración _airflow.cfg_.
- También puede definirse dentro de los argumentos del **DAG** el parámetro de default_args. Este sirve para definir un diccionario de argumentos básicos que son comunes a las tareas, tales como retries.
###  Control Flow
- El **DAG** solo correrá una tarea si las tareas de las que depende han sido ejecutadas exitosamente.
- Existen formas de cambiar este comportamiento por defecto:
    - **Branching** → Se usa para decirle al **DAG** que en vez de no correr todas las tareas dependientes, que escoja uno o varios caminos alternativos en caso de de no ejecución de una tarea padre. Se usa el decorador de @task.branch. Este decora una función cuyo propósito es retornar el _ID_ de otra función (o lista de _IDs_).
    - **Depends On Past** → Una tarea solo correrá si el estado previo de esa misma tarea fué exitoso, de esta forma se asegura que la tarea solo se ejecuta si anteriormente se ejecutó bien. Para ello solo hay que pasarle a la tarea el argumento depends_on_past a **TRUE**.
- Para más información sobre **DAGs**, consultar la [docu (parte de DAGs).](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html#dags)
![[airflow-dag3.png]]
### Task Group
- Los _task group_ son elementos simplemente visuales y organizativos. Para usarlo es posible con el decorador `@task_group`.
```python
 from airflow.decorators import task_group

 @task_group()
 def group1():
     task1 = EmptyOperator(task_id="task1")
     task2 = EmptyOperator(task_id="task2")

 task3 = EmptyOperator(task_id="task3")

 group1() >> task3
```
### Edge Labels
- Al igual que con _task groups_ se pueden etiquetar las aristas de dependencias. De esta forma es visualmente más claro y sencillo de entender. Es un puro elemento de _GUI_.
![[airflow-dag4.png]]
```python
ingest >> analyse >> check
check >> Label("No errors") >> save >> report
check >> Label("Errors found") >> describe >> error >> report
```
## DAG Runs
- Cada vez que un **DAG** se ejecuta, un **Run DAG** se crea y todas las tareas dentro se ejecutan. Cada **DAG** se ejecuta de forma independiente a otras instancias, de esta forma, se pueden tener varias instancias del mismo **DAG** corriendo simultáneamente.
## Task
- La unidad básica de proceso en **Airflow** son las tareas. Existen diferentes tipos de tareas:
    - **Operadores** → son tareas predefinidas en plantillas que pueden usarse directamente. Algunos de ellos tiene relación con tecnologías tales como _http, docker, mysql,_ etc.
    - **Sensores** → son tareas especiales que esperan a un _trigger_ para continuar con la tarea.
    - **TaskFlow** → tareas definidas totalmente en _Python_ con el decorador @task. Son los más sencillos y versátiles, ya que pueden definir cualquier tipo de proceso escrito en _Python_, incluidos los procesos de los **operadores.**