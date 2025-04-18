- #iptables

- Iptables es una herramienta de cortafuegos, de espacio de usuario, con la que el superusuario define reglas de filtrado de paquetes, de traducción de direcciones de red y que mantiene registros logs. Esta herramienta está construida sobre [[netfilter]], una parte del núcleo de Linux que permite interceptar y manipular paquetes de red.

### Información instrucciones
- Consultar información:

```
man iptables  
iptables –h
```

- Comprobar estado del cortafuegos:

```
iptables –L –n -v
```

- Parar el cortafuegos y eliminar todas sus reglas:

```
iptables -F  
iptables -X  
iptables –t nat -F  
iptables –t nat -X  
iptables –t mangle -F  
iptables –t mangle -X  
iptables -P INPUT ACCEPT  
iptables -P OUTPUT ACCEPT
```

- Denegar cualquier tráfico:

```
iptables -P INPUT DROP  
iptables -P OUTPUT DROP  
iptables -P FORWARD DROP
```

- Abrir puertos HTTP/HTTPS (80 y 443):

```
iptables -A INPUT -m state --state NEW -p tcp --dport 80 -j ACCEPT  
iptables -A INPUT -m state --state NEW -p tcp --dport 443 -j ACCEPT
```

- Estos son solo algunas de las muchas configuraciones posibles de firewall.