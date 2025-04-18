- #git | #github
- links:
	- [gitflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)

## Comenzar proyecto
- Iniciamos el proyecto en el directorio base:
	- **Inicio proyecto** -> `git init`
- Una vez iniciado y creados todos los archivos de git, hacemos el primer commit. Para ello añadimos los cambios a git y luego commiteamos:
	- **Añadir cambios** ->  `git add .`
	- **Commit** -> `git commit -m "Commit inicial"`

## Commits mensajes
- [Convención](https://www.conventionalcommits.org/en/v1.0.0/)
- `(feat | fix | refactor | docs | chore)\(<scope>\ ): "Mi mensajito"`

## Sincronización con Github
- Tenemos que crear un remoto, para ello hacemos lo siguiente:
	- **Añadir remoto** -> `git remote add origin https://github.com/USER/REPONAME.git`
- Lo siguiente será crear un token para nuestro proyecto y así poder hacer operaciones sin necesidad de introducir credenciales cada vez:
	- Nos vamos a Github arriba a la derecha, luego *Settings > Developer Settings > Personal access tokens > Tokens*.
	- Añadimos el nombre que queramos y marcamos la opción **repo**.
	- Copiamos el token.
- Ahora añadimos el token a nuestro remoto previamente definido:
	- **Añadir url con token** -> `git remote set-url origin https://USER:TOKEN@github.com/USER/REPONAME.git`
- Ahora deberíamos poder hacer un `git push origin master` sin necesidad de autenticarnos.

## Actualizar URL remote
- Para ello debemos settear la nueva URL en el remote que ya teníamos. Si el remote se llamaba *origin*, entonces:
	- **Actualizar remote** -> `git remote set-url origin NEWURL`

## Opciones varias
- Modificar un solo fichero a commits anteriores -> `git checkout <commit-hash> <path-to-file>`.
- Modificar último commit -> `git reset --soft HEAD^1`.
- Cambiar al último commit (rollback) -> `git reset --hard HEAD^1`.

## Arreglar una cabeza independiente
- `git switch -c tmp` Para guardar los cambios en una nueva rama.
- `git switch master` Para cambiar a la rama principal.
- `git merge tmp` Para incorporar los cambios a la rama principal.

## Cambiar commits de rama
- `git checkout -b new-branch-name` -> Crear rama y cambiarte a ella.
- `git cherry-pich <commit-hash>` -> Coger ese commit y pasarlo a rama actual.
- `git checkout old-branch-name` -> Cambiar a antigua rama.
- `git reset --hard HEAD^1` -> Borrar commit.

## Borrar ramas
- `git branch -d <branch-name>` **local**
- `git push <remote_name> --delete <branch_name>` **remote**

## Arreglar un reset hard erróneo
- Si se ha hecho commit antes del reset hard se puede arreglar, si no es imposible.
- `git reflog` -> saca los commits hechos en el repo.
- `git reset --hard <hash>` -> volver al commit antes del reset.

## Borrar ramas en remoto
- `git push origin --delete <branch-name>`.

# TODO
╰─ git rm --cached vision_por_computador/p1/mivideo.avi                                          ─╯
