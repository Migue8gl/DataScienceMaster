# Usabilidad del lenguaje

### Ejercicio 1
- Usando #structuredbinding, implementa las siguientes funciones con solo una línea de código:

```c++
#include <string>
#include <map>
#include <iostream>

template <typename Key, typename Value, typename F>
void update(std::map<Key, Value>& m, F foo) {
    // TODO:
}
int main() {
    std::map<std::string, long long int> m {
        {"a", 1},
        {"b", 2},
        {"c", 3}
    };
    update(m, [](std::string key) {
        return std::hash<std::string>{}(key);
    });
    for (auto&& [key, value] : m)
        std::cout << key << ":" << value << std::endl;
}
```

### Solución

```c++
template <typename Key, typename Value, typename F>
void update(std::map<Key, Value>& m, F foo) {
	for (auto&& [key, value] : m) value = foo(key);
}
```

### Ejercicio 2
- Implementa una función que calcule la media usando #foldexpression , de manera que acepte todos los argumentos que se le pase.

### Solución

```c++
template <typename ... T>
auto mean(T ... t) {
    return (t + ...) / sizeof...(t);
}
```