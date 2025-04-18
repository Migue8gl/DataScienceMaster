- #googlecloud | #lookml | #dataanalyst 

- LookML (Looker Modeling Languaje) es el lenguaje con el que se crean los modelos de [[looker|Looker]]. 
- Es un lenguaje que permite generalizar consultas SQL de una manera sencilla dentro de una interfaz de usuario cómoda e intuitiva. Es una capa de abstracción por encima de las consultas a base de datos, de forma que encapsula relaciones entre tablas, atributos, cálculos sobre estos atributos, etc.

## LookML Jerarquía
### Project
- Un projecto es una librería de código LookML que contiene uno o varios modelos. Cada proyecto debe tener asociado un solo repositorio de [[git|Git]].

### Model
- Un modelo establece una conexión con la base de datos que se vaya a utilizar y establece las vistas de datos que vayan a utilizar esa conexión. 
- Estos definen los **explores** que se vayan utilizar y las relaciones entre estas vistas. 

### Explores
- Los **explores** son agrupaciones de vistas que tienen que ver entre sí. Un explore (cojunto de **views** ) responde dudas dentro del negocio.
- Su uso incluye análisis de los datos, visualización y exploración. 
- Se utilizan para crear visualizaciones y reportes.

### Views
- Representan las tablas dentro de base de datos, cada una con sus atributos y datos.
- Las vistas contienen definiciones de las tablas utilizadas, las dimensiones y las medidas.

## Definición de dimensiones
![[lookml_dimension_definition.png|500]]
![[lookml_dimension_definition_2.png|500]]

- Hay varios tipos de dimensiones -> **string, number, yesno (boolean with yes or no answers), tier (clasifica en rangos o valores), timeframe (varios formatos de tiempo), duración (tiempo en intervalos)**.

 ![[look-ml-dimension-tier.png|500]]