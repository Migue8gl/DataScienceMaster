- #ssl

- SSL (Secure Sockets Layer) es un protocolo de seguridade. SSL se utiliza para cifrar los datos que se transfieren entre el servidor y el navegador. Es el protocolo que utiliza [[https]].

- Técnicamente hablando, SSL utiliza un cifrado asimétrico de clave pública para autenticar el servidor y establecer una conexión cifrada entre el servidor y el cliente. Cuando un cliente se conecta a un servidor que utiliza SSL, el servidor envía su certificado SSL al cliente. El certificado SSL contiene la clave pública del servidor, que el cliente utiliza para cifrar los datos antes de enviarlos al servidor.

- Una vez que se ha establecido una conexión segura, los datos se encriptan con un cifrado simétrico de clave secreta. Este cifrado utiliza una clave compartida que se genera durante la conexión y se utiliza para cifrar y descifrar los datos. Debido a que la clave compartida se genera durante la conexión, es única para esa conexión en particular y no puede ser utilizada para descifrar otras conexiones SSL.