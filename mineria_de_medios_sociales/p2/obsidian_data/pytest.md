- #python | #test | #unittest

# Conceptos
## Fixture
- Un *fixture* es una función de configuración cuya función es ejecutarse antes o después de ciertos *tests* para preparar el entorno en el que los *tests* van a correr.
- Se utilizan los decoradores: `@pystest.fixture`
- Ejemplos de estos son:
	![[pytest-example-fixture.png|350]]
## Mock
- Un *mock* es un tipo de función que simula el comportamiento de objetos, sistemas o componentes.
- Gracias a ello, podemos diseñar *tests* de forma que otros componente no interfieran y así probar de forma aislada lo que se plantea *testear*.
## Class-Based tests
- Los *tests* basados en clases tienen dos métodos especiales:
	- **setup_method**: ejecuta código el principio de cada test, de forma que todos los *tests* de la clase tendrán ese código en común. Sirve para establecer un contexto común. Equivalente a *fixture* en *testing* funcional.
	 - **teardown_method**: de igual forma que el **setup**, pero al finalizar los *tests*. Su uso suele ser el de limpiar el entorno tras finalizar los *tests*.
## Marks
- Las *marks* funcionan como marcadores de *tests*. De esta forma se pueden tratar de forma distinta algunos *tests* debido a la naturaleza de estos. Por ejemplo:
	- **skip**: se le puede pasar una *reason* al decorador. Sirve para saltar un *test*.
	- **slow**: los *tests* marcados con *slow* deben ser aquellos que van a tardar mucho tiempo en ejecutarse. Si es necesario, se pueden saltar los *tests* que tengan esta marca.
## Parametrize
- Este tipo de decoradores sirven para especificar múltiples parámetros que han de probarse para una misma función de test. Por ejemplo:
	![[pytest-parametrize-example.png|500]]