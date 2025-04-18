- #usability

- A la usabilidad del lenguaje nos referimos cuando hablamos del comportamiento del lenguaje antes de tiempo de ejecución, es decir, el diseño de nuestro código de cara a la facilidad de mantenerlo, entenderlo, escalarlo etc.

### Constantes
- #nullptr:
	- El propósito de **nullptr** aparece cuando intentamos reemplazar una variable con el valor NULL. Antes de la introducción de **nullptr** en C++11, se usaba **NULL** o $0$ para representar un puntero nulo, lo que a menudo generaba confusión en el código.
	- **nullptr** es un valor constante que se puede asignar a cualquier puntero o puntero a miembro. Su tipo es `std::nullptr_t`, que se define en la biblioteca estándar de C++. Cuando se asigna nullptr a un puntero, se indica explícitamente que el puntero no apunta a ningún objeto o función.
- #constexpr:
	- Las expresiones constantes son aquellas que siempren producen el mismo resultado sin efectos secundarios. Como por ejemplo $1+2$ o $3·4$ . El compilador puede optimizar estas expresiones en tiempo de compilación. Para definir una función como tal debemos añadir la palabra clave (siempre y cuando su comportamiento sea siempre determinístico). Ej:
	
	```c++
	constexpr int fibonacci(const int n) {
	    return n == 1 || n == 2 ? 1 : fibonacci(n-1) + fibonacci(n-2);
	}
	```

### Variables e inicialización
- Podemos definir variables locales dentro de un if o un switch, antes solo de podía en los bucles for.

### Inicializador de lista
- #initializerlist
- Se usa para inicializar los atributos de una clase. En el C++ clásico, los métodos en los que se usaba un inicializador de lista  no eran genéricos. Ahora pueden utilizarse los símbolos `{}` como cuando inicializamos un struct o un array.

### Unión estructurada 
- #structuredbinding
- Podemos retornar múltiples valores de las funciones y despaquetarlos usando la función `auto()`. Un ejemplo es el siguiente:

```c++
#include <iostream>
#include <tuple>

std::tuple<int, double, std::string> f() {
    return std::make_tuple(1, 2.3, "456");
}

int main() {
    auto [x, y, z] = f();
    std::cout << x << ", " << y << ", " << z << std::endl;
    return 0;
}
```

### Inferencia de tipos
-  Se introducen los palabras claves `auto` y `decltype` para implementar la derivación del tipo de la variable, de forma que el compilador se encargue de ello, como otros lenguajes modernos.
- #auto: 
	- Se encarga de darle un tipo genérico a la variable que vamos a utilizar (no se puede inferir todavía el tipo de un array). Ej:
	
	```c++
	int add(auto x, auto y) {
	    return x+y;
	}
	
	auto i = 5; // type int
	auto j = 6; // type int
	std::cout << add(i, j) << std::endl;
	```
	
	 - Se puede utilizar a partir de C++14 como tipo de retorno de una función.

- #decltype:
	- Su uso es muy parecido al de `typeof`. Se usa para inferir el tipo de una expresión o variable, normalmente para comprobaciones.

### Flujo de control
- Bucles basados en rangos:
	- Son bucles basados en rango, al igual que en Python, de forma que son muy concisos:
	
```c++
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> vec = {1, 2, 3, 4};
    if (auto itr = std::find(vec.begin(), vec.end(), 3); itr != vec.end()) *itr = 4;
    for (auto element : vec)
        std::cout << element << std::endl; // read only
    for (auto &element : vec) {
        element += 1;                      // writeable
    }
    for (auto element : vec)
        std::cout << element << std::endl; // read only
}
```

- constexpr en condiciones if:

```c++
#include <iostream>

template<typename T>
auto print_type_info(const T& t) {
    if constexpr (std::is_integral<T>::value) {
        return t + 1;
    } else {
        return t + 0.001;
    }
}
int main() {
    std::cout << print_type_info(5) << std::endl;
    std::cout << print_type_info(3.14) << std::endl;
}
```


### Plantillas 
- #externtemplates:
	- En el C++ tradicional las plantillas eran instanciadas solo cuando estas eran usadas. Esto se traduce como un tiempo de compilación más prolongado debido a instancias repetidas
	- En C++11 pues se introduce una plantilla externa que amplia la sintaxis del compilador para instanciar una plantilla en una ubicación concreta de memoria, de forma que podemos decirle de forma explícita al compilador cuando instanciar la plantilla.
	
	```c++
	template class std::vector<bool>;          // force instantiation
	extern template class std::vector<double>; // should not instantiation in current file
	```

- #typealiastemplates:
	- Las plantillas o `templates` son usadas para generar tipos en C++. En el C++ tradicional usando #typedef podíamos cambiar el nombre de un tipo (que no de una plantilla). En C++11 podemos usar #using para definir la forma de escritura a partir de ahora y al mismo tiempo tiene el mismo efecto que `typedef`.

	```c++
	typedef int (*process)(void *);
	using NewProcess = int(*)(void *);
	template<typename T>
	using TrueDarkMagic = MagicType<std::vector<T>, std::string>;
	
	int main() {
	    TrueDarkMagic<bool> you;
	}
	```

- #variadictemplates:
	- A partir de C++11 podemos definir en una plantilla, ya sea de clase o de función, un número indeterminado de parámetros con la siguiente sintaxis (podemos también obligar a que haya cierto número de parámetros obligatorios y el resto libres):

	```c++
	template<typename Require, typename... Args> class Magic;
	```

	- Para despaquetar los parámetros podemos usar la función `sizeof` para saber que número de paŕametros le llega a la plantilla.
	-  Después de conocer el número podemos usar varias formas de desempaquetamiento:
		1) Plantilla de función recursiva:  Se pasa de forma recursiva los parámtros a la función hasta que no queden más.

		```c++
		#include <iostream>
		template<typename T0>
		void printf1(T0 value) {
		    std::cout << value << std::endl;
		}
		template<typename T, typename... Ts>
		void printf1(T value, Ts... args) {
		    std::cout << value << std::endl;
		    printf1(args...);
		}
		int main() {
		    printf1(1, 2, "123", 1.1);
		    return 0;
		}
		```

		2) Ampliación de la plantilla de parámetros variables:

		```c++
		template<typename T0, typename... T>
		void printf2(T0 t0, T... t) {
		    std::cout << t0 << std::endl;
		    if constexpr (sizeof...(t) > 0) printf2(t...);
		}
		```

		3) Ampliación de lista de inicialización: El problema de la solución recursiva es que debes crear una función que termine la recursión, con la inicialización de lista podemos expandir los parámetros de número indefinido e ir recoriendolos.
		
		La expresión `(void) std::initializer_list<T>{([&args] { std::cout << args << std::endl; }(), value)...};` utiliza una expresión lambda para imprimir cada elemento de la lista de argumentos "args" en la consola.

		```c++
		template<typename T, typename... Ts>
		auto printf3(T value, Ts... args) {
		    std::cout << value << std::endl;
		    (void) std::initializer_list<T>{([&args] {
		        std::cout << args << std::endl;
		    }(), value)...};
		}
		```

- #foldexpression: 
	- La expresión utilizada para definir variables variádicas, que es `...`, también se utiliza para definir expresiones:

	```c++
	#include <iostream>
		template<typename ... T>
	auto sum(T ... t) {
	    return (t + ...);
	}
	int main() {
	    std::cout << sum(1, 2, 3, 4, 5, 6, 7, 8, 9, 10) << std::endl;
	}
	```

- #nontypetemplateparameterdeduction:
	- Los parámetros de una plantilla solían ser tipos específicos. Ahora, usando la palabra clave `auto` podemos dejar la deducción de los tipos al compilador, de forma que ya no es necesario declararlos. Ej: 

	```c++
	template <auto value> void foo() {
	    std::cout << value << std::endl;
	    return;
	}
	
	int main() {
	    foo<10>();  // value as int
	}
	```

### Orientado a objetos
- #delegateconstructor:
	- A partir de C++11 se introduce el concepto de constructor de delegación, de forma que un constructor puede llamar a otro constructor en la misma clase.
	
```c++
#include <iostream>
class Base {
public:
    int value1;
    int value2;
    Base() {
        value1 = 1;
    }
    Base(int value) : Base() { // delegate Base() constructor
        value2 = value;
    }
};

int main() {
    Base b(2);
    std::cout << b.value1 << std::endl;
    std::cout << b.value2 << std::endl;
}
```

- #inheritanceconstructor:
	- En el C++ tradicional, los constructores necesitan pasar argumentos uno a uno si necesitan herencia, lo que conduce a la ineficiencia. C++11 introduce el concepto de constructores de herencia mediante la palabra clave using.
- #explicitvirtualfunctionoverride:
	- En el C++ tradicional es corriente tender a sobrescribir [[virtual functions]]  (funciones virtuales).  Al crear dos métodos con el mismo nombre en la clase padre y en la subclase, la intención quizá es crear un método con el mismo nombre y no sobrescribir el método padre. Otro posible escenario es que cuando la función virtual del padre se borre, la subclase se haga propietaria de la antigua función, ya no sobrecarga la función virtual y se convierta en un método normal de la clase.

	```c++
	struct Base {
	    virtual void foo() { std::cout << "Base::foo\n"; }
	};
	
	struct SubClass : Base {
	    void foo() { std::cout << "SubClass::foo\n"; }
	    void bar() { std::cout << "SubClass::bar\n"; }
	};
	
	int main() {
	    SubClass obj;
	    obj.foo(); // Imprime "SubClass::foo"
	    obj.bar(); // Imprime "SubClass::bar"
	
	    // Versión posterior de la biblioteca elimina o renombra la función virtual
	    // de la clase Base
	    // virtual void foo() -> virtual void newFoo()
	    
	    // Debido al cambio en la biblioteca, la función foo en la subclase ya no
	    // sobrescribe la función virtual foo de la clase Base
	    // Se convierte en una función normal de la subclase
	    // void foo() -> void newFoo()
	    
	    // Esto significa que el siguiente código invoca la función SubClass::newFoo en lugar
	    // de la función sobrescrita SubClass::foo, lo que puede causar errores inesperados.
	    obj.foo(); // Ahora imprime "SubClass::newFoo" en lugar de "SubClass::foo"
	
	    return 0;
	}
	```

	- Para ello se introducen las siguientes palabras claves:
		- #override: Le dice explícitamente al compilador que sobrecargue la función, si es que existe en la clase padre.
		- #final: Si una clase y/o una función virtual lleva `final`, sus clases hijas dejan  de heredar y la función virtual deja de ser sobrecargada.

- #explicitdeletedefaultfunction: 
	- En C++11, se puede utilizar la palabra clave "default" para permitir que el compilador genere una función por defecto de manera explícita, y "delete" para eliminar una función generada por defecto de manera explícita. Esto permite un control más preciso sobre la generación de funciones y evita problemas como tener dos constructores al mismo tiempo.

### Ejercicios
- [[exercises|#Usabilidad del lenguaje]] 
