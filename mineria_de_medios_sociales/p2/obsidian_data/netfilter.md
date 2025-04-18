- #netfilter

- Netfilter es un proyecto gratis y open-source mantenido por la comunidad que provee filtrado de software para las versiones de kernel de Linux desde la 2.4.x hasta la más nueva.
- Los ganchos de netfilter son un framework dentro del núcleo Linux que permite a los módulos del núcleo registrar funciones de devolución de llamada en diferentes ubicaciones de la pila de red de Linux. La función de llamada de retorno registrada es llamada por cada paquete que atraviesa el gancho respectivo dentro de la pila de red Linux.

### Usos
- Construir firewalss basados en filtrado de paquetes [[stateless]] y [[stateful]].
- Desplegado de clusters de firewall de alta disponibilidad stateless y stateful.
- Uso de NAT y enmascaramiento para compartir el acceso a Internet si no se tienen suficientes direcciones IP públicas.
- Uso de NAT para crear proxies transparentes.
- Etc.