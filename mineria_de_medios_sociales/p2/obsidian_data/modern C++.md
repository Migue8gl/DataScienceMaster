- #modercplusplus

## Github:
- [modernC++](https://github.com/changkun/modern-cpp-tutorial/tree/master)

## Compilador 
- Vamos a usar [[clang++]] en vez de otros compiladores (como g++). Utilizaremos el flag *-std=c++2a*, que es una opción de compilación que indica que se debe compilar el código utilizando el estándar C++2a (también conocido como C++20).

## Uso de C
- Cuando hagamos uso del lenguaje C, debemos usar `extern "C"`, separar el código de C del código en C++ y después linkearlos ambos.
- Ej:

```c++
// foo.h
#ifdef __cplusplus
extern "C" {
#endif

int add(int x, int y);

#ifdef __cplusplus
}
#endif

// foo.c
int add(int x, int y) {
    return x+y;
}

// 1.1.cpp
#include "foo.h"
#include <iostream>
#include <functional>

int main() {
    [out = std::ref(std::cout << "Result from C code: " << add(1, 2))](){
        out.get() << ".\n";
    }();
    return 0;
}
```

## Usabilidad del lenguaje
- [[usability-modernc++]]

## Mejoras en el tiempo de ejecución
- [[runtime-modernc++]]

