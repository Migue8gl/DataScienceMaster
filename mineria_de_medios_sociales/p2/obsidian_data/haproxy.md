- #haproxy

- HaProxy es un balanceador de carga ([[load balancer]]) y también un proxy, de forma que puede balancear cualquier tipo de tráfico.

### Instalación
- Para instalar haproxy en ubuntu solo deberemos introducir el siguiente comando: `sudo apt-get install haproxy`.

### Configuración
- Debemos modificar el archivo **/etc/haproxy/haproxy.cfg** para indicarle nuestros servidores y qué [[http requests]] balancear.
- Con la siguiente configuración haremos que escuche en el puerto 80 y lo redirigiremos a alguna de las máquinas finales:

```
frontend http-in  
	bind *:80  
	default_backend balanceo_usuarioUGR  
backend balanceo_usuarioUGR  
	server m1 ip_maquinaM1:80 maxconn 32  
	server m2 ip_maquinaM2:80 maxconn 32
```

- En las máquinas servidores finales deberemos tener Apache instalado escuchando en el puerto 80.

##### Habilitado de estadísticas
- Añadiendo lo siguiente al archivo de configuración podremos acceder a un dashboard de estadísticas del balanceador:

```
global  
	stats socket /var/lib/haproxy/stats  
listen stats  
	bind *:9999  
	mode http  
	stats enable  
	stats uri /stats  
	stats realm HAProxy\ Statistics  
	stats auth usuario_UGR:usuario_UGR
```

![[haproxy-stats.png]]

- Podremos acceder a través de `http://ip_maquina:9999/stats`.

### Enlaces de interés
- [link1](http://code.google.com/p/haproxy-docs/)  
- [link2](http://haproxy.1wt.eu/download/1.4/doc/configuration.txt)