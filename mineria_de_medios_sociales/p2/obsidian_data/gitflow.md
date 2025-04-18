#releases | #gitflow | #git | #feature

## Base
- **Gitflow** es un flujo de trabajo basado en [[git]]. Otras prácticas para **CI/CD** usan el sistema de [trunk-based-workflow](https://www.atlassian.com/continuous-delivery/continuous-integration/trunk-based-development).
- Se basa en el desarrollo de un proyecto siguiendo dos tipos de ramas:
	- **main**: la cual almacena el historial de versiones.
	- **develop**: sirve como una rama de integración de características.
- Normalmente, todas las versiones o *releases* se marcan con un número de versión. Un tag, como por ejemplo `v0.1`.
![[gitflow-branches.png]]
## Comandos básicos
### Init
- Para crear la rama de **develop** de forma automática y comenzar a configurar aspectos básicos, se puede ejecutar:
	- `git flow init`
- Si **develop** ya existe, entonces creará la configuración necesaria.
### Feature
- Siempre hacer `pull` de todos los cambios de **develop** y **main** antes de comenzar o terminar algo.
- Para crear una rama de **feature** en develop:
	- `git flow feature start feature/<feature-name>`
- Para terminarla:
	- `git flow feature finish feature<feature-name>`
- Si ya se está en la rama de **feature**, entonces no hay que especificar el nombre.
![[gitflow-feature-develop.png]]
### Release
- Siempre hacer `pull` de todos los cambios de develop y main antes de comenzar o terminar algo.
- Para hacer un **release** se requiere haber terminado suficientes **features** o que una fecha para la **release** se acerque.
- El **release** es un fork de **develop**. Esta se crea con un nuevo tag que indique el número de versión.
	- `git flow release start release/<tag>`
- En este punto del desarrollo no se añaden más características, solo **bug-fixes**, **documentación** y otras tareas del estilo.
	- Alguna que otra comprobación: `git diff origin/main`
- Una vez la **release** se ha terminado:
	- `git flow release finish release/<tag>`
- La creación de *tags* son automáticos.
- Después se hace `push` tanto en la rama de **main** y **develop**. Si se quieren añadir *tags* adicionales se crean y se *pushean* (en algunos proyectos es un *trigger* para los despliegues en *cloud*).
### Hotfix
- Siempre hacer `pull` de todos los cambios de develop y main antes de comenzar o terminar algo.
- Para crear un **hotfix**, que es además la única rama que se puede crear directamente desde **main** se hace lo siguiente:
	- `git flow hotfix start hotfix/<hotfix-name>` (el nombre suele ser el *tag*)
- Un **hotfix** se hace cuando es necesario crear un parche lo más rápido posible y urgentemente a producción.
- Para terminarlo:
	- `git flow hotfix finish hotfix/<hotfix-name>` (el nombre suele ser el *tag*)
- La creación de *tags* son automáticos.
- Después se hace `push` tanto en la rama de **main** y **develop**. Si se quieren añadir *tags* adicionales se crean y se *pushean* (en algunos proyectos es un *trigger* para los despliegues en *cloud*).