- #apachebenchmark | #ab 
- [Documentación](https://httpd.apache.org/docs/2.2/programs/ab.html)

- Apache Benchmark o AB es un programa de hebra única que se usa para medir el rendimiento de servidores web HTTP.
- Es una herramienta sencilla, pues simula la petición del sitio web repetidas veces, pero esto no es lo que suelen hacer los usuarios, por lo que su uso es limitado en cuanto a customización.

### Ejemplo de uso
- `ab -n 10000 -c 10 "http://localhost/index.html"`
- Esta instrucción ejecutará 10000 [[http requests]] GET, procesando hasta 10 peticiones concurrentemente.

![[ab-ejemplo.png]]
