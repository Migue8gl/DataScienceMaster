- Una función virtual es una función miembro que se declara dentro de una clase base y que es redefinida (sobrescrita) por una clase derivada. Cuando se hace referencia a un objeto de una clase derivada utilizando un puntero o una referencia a la clase base, se puede llamar a una función virtual para ese objeto y ejecutar la versión de la función de la clase derivada.

- Las funciones virtuales aseguran que se llame a la función correcta para un objeto, independientemente del tipo de referencia (o puntero) utilizado para la llamada de función. Se utilizan principalmente para lograr el polimorfismo en tiempo de ejecución. Las funciones se declaran con una palabra clave "virtual" en la clase base. La resolución de la llamada a la función se realiza en tiempo de ejecución.

- Las reglas para las funciones virtuales son:
	- Las funciones virtuales no pueden ser estáticas.
	- Una función virtual puede ser una función amiga de otra clase.
	- Las funciones virtuales deben ser accedidas utilizando puntero o referencia de tipo de clase base para lograr el polimorfismo en tiempo de ejecución.
	- El prototipo de las funciones virtuales debe ser el mismo en la clase base y en la clase derivada.
	- Las funciones virtuales siempre se definen en la clase base y se sobrescriben en una clase derivada. No es obligatorio que la clase derivada sobrescriba (o redefina) la función virtual, en ese caso, se utiliza la versión de la función de la clase base.
	- Una clase puede tener un destructor virtual, pero no puede tener un constructor virtual.
- Ej:

```c++
// CPP program to illustrate
// concept of Virtual Functions

#include<iostream>
using namespace std;

class base {
public:
	virtual void print()
	{
		cout << "print base class\n";
	}

	void show()
	{
		cout << "show base class\n";
	}
};

class derived : public base {
public:
	void print()
	{
		cout << "print derived class\n";
	}

	void show()
	{
		cout << "show derived class\n";
	}
};

int main()
{
	base *bptr;
	derived d;
	bptr = &d;

	// Virtual function, binded at runtime
	bptr->print();

	// Non-virtual function, binded at compile time
	bptr->show();
	
	return 0;
}
```

- Output: 

```
print derived class
show base class
```