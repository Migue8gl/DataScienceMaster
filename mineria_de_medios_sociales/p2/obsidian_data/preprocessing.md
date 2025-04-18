- #ml | #preprocessing | #data

## Variables cualitativas
- Cuando se tiene una variable categórica o cualitativa, una forma de codificarla es añadiendo una variable *dummy* que indique si pertenece o no a esa categoría. Es decir:
	- Hay una variable **etnias** con los valores **asiático** y **africano**. Se añaden dos variables $x_1$ y $x_2$. 
	- Si $x_1$ es $1$ es asiático, si es $0$ no es asiático.
	- Con la variable **sexo** añadiríamos $x_1$. Si es $1$ es hombre, si es $0$ es mujer.
- El número de variables *dummys* siempre será menor o igual al número de variables cualitativas originales.
### Interacción de cuantitativas con cualitativas
- [[interactions between variables]]