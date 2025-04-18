- #mailcommandinjection

- Este tipo de ataques pueden aprovechar inputs mal validados para atacar servidores web #smtps o #imap.
- Para explotar un servidor SMTP, los atacantes necesitan una cuenta de correo electrónico válida para enviar mensajes con comandos inyectados. Si el servidor es vulnerable, responderá a las peticiones de los atacantes, permitiéndoles, por ejemplo, anular las restricciones del servidor y utilizar sus servicios para enviar spam.
- La inyección IMAP podría realizarse principalmente en aplicaciones de correo web, explotando la funcionalidad de lectura de mensajes. En estos casos, el ataque puede realizarse simplemente introduciendo, en la barra de direcciones de un navegador web, una URL con los comandos inyectados.