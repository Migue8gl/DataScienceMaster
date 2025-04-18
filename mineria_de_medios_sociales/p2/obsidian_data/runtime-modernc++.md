- #runtime

### Expresiones Lambda
- Las expresiones #lambda proporcionan la característica de las funciones anónimas. La expresión general de las expresiones lambda es la siguiente:

```
[capture list] (parameter list) mutable(optional) exception attribute -> return type {
// function body
}
```

- La parte de `capture list` puede ser entendida como un tipo de parámetro. Esta puede servir para la transferencia de datos entre la función y el exterior, debido a que la función lambda solo puede usar variables locales a su cuerpo. Existen estos distintos tipos:
	- #value: El valor captado es copiado cuando la función  lambda es creada, no cuando se llama.

	```c++
	void lambda_value_capture() {
	    int value = 1;
	    auto copy_value = [value] {
	        return value;
	    };
	    value = 100;
	    auto stored_value = copy_value();
	    std::cout << "stored_value = " << stored_value << std::endl;
	    // At this moment, stored_value == 1, and value == 100.
	    // Because copy_value has copied when its was created.
	}
	```

	- #reference: Se guarda la referencia, por lo que el valor cambia.

	```c++
	void lambda_reference_capture() {
	    int value = 1;
	    auto copy_value = [&value] {
	        return value;
	    };
	    value = 100;
	    auto stored_value = copy_value();
	    std::cout << "stored_value = " << stored_value << std::endl;
	    // At this moment, stored_value == 100, value == 100.
	    // Because copy_value stores reference
	}
	```

	- #implicit: Podemos dejar al compilador que maneje el valor captado, en vez de escribirlo nosotros solo le dejamos una guía de si el valor es por referencia o por valor (& o =).
	- #expression:  Los valores capturados en los anteriores tipos eran variables declaradas fuera del ámbito de la función, por lo que estos métodos capturan el [[lvalue]] y no el [[rvalue]]. C++14 nos permite capturar valores inicializados con expresiones arbitrarias, por lo que serían #rvalue .

	```c++
	#include <iostream>
	#include <memory>  // std::make_unique
	#include <utility> // std::move
	
	void lambda_expression_capture() {
	    auto important = std::make_unique<int>(1);
	    auto add = [v1 = 1, v2 = std::move(important)](int x, int y) -> int {
	        return x+y+v1+(*v2);
	    };
	    std::cout << add(3,4) << std::endl;
	}
	```

- `important` es un puntero exclusivo que no puede ser capturado por valor usando el operador `=` en la sintaxis de inicialización de captura. En cambio, es necesario mover el puntero a la expresión lambda utilizando `std::move()`.

##### Lambda genérico
- En C++14 las funciones lambda pueden utilizar la `keyword` `auto` para hacerlas genéricas.

### Wrapper de objetos de función
- Esta es una característica que nos permite crear objetos que se comportan como funciones. Esto es útil cuando se quiera pasar una función a otra por parámetro o cuando se necesita que una fución tenga un estado interno.
- `std::function` es un envoltorio o #wrapper, genérico y polimórfico. Sus instancias pueden guardar, copiar y llamar cualquier entidad que pueda ser llamada. Un contenedor de funciones.

```c++
#include <functional>
#include <iostream>

int foo(int para) {
    return para;
}

int main() {
    // std::function wraps a function that take int paremeter and returns int value
    std::function<int(int)> func = foo;

    int important = 10;
    std::function<int(int)> func2 = [&](int value) -> int {
        return 1+value+important;
    };
    std::cout << func(10) << std::endl;
    std::cout << func2(10) << std::endl;
}
```

- La función func toma un valor `int` como parámetro y devuelve un `int` como salida. También podemos envolver funciones lambda.
- #bind: Con `std::bin` podemos vincular los parámetros de llamada de la función al objeto. De esta forma, podemos ir recogiendo estos parámetros según vayan siendo generados. Esto es útil cuando no podemos acceder a todos los parámetros de una función a la vez.
- #placeholder: Se utiliza junto con `std::bin`. Los placeholders nos sirven para marcar qué argumento es, se marca su posición. Ej:

```c++
auto f = std::bind(foo, std::placeholders::_1, 42);
```

- Se utiliza el placeholder `_1` para indicar que el primer argumento de la función `foo` se inicializará con el valor `42`.

- Los placeholders se numeran a partir de `_1`, que representa el primer argumento, `_2` representa el segundo argumento, y así sucesivamente. También hay un placeholder especial llamado `_n` que se utiliza para especificar el máximo número