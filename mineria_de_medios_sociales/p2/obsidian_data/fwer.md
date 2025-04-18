- #statistics | #fwer

- El *Family-wise error rate* es la probabilidad de cometer un [[hypothesis errors|falso descubrimiento]] o error **tipo I** cuando se hacen $m$ hipótesis.
- La probabilidad de no cometer un error tipo I un una hipótesis es de $1-\alpha$. Cuando tenemos $m$ hipótesis, esta probabilidad es de $(1-\alpha)^m$. La probabilidad de cometer al menos un error de tipo I es de $1-(1-\alpha)^m$.
- $FWER=1-(1-\alpha)^m$ (probabilidad de cometer AL MENOS un error tipo I).
- Se pueden plantear [[post hoc]] para paliar el error tipo I cometido en aquellas familias de hipótesis, pero es preferible agrupar esas hipótesis en una sola:
	- En vez de comprobar si no hay diferencias entre los grupos $ij$ siendo $i,j\in\{0,m\}$, en el cual estaríamos planteando $m(m-1)/2$ hipótesis, es mejor plantear si todos los grupos son iguales. Con ello pasamos de varias hipótesis a solo una más general.
	- ¿Por qué hacer esto en evz de un **post-hoc**? Porque hacer un post-hoc tiene [[hypothesis errors#^3e0b9f|menos potencia]] (error tipo II).
- Se suele hacer lo primero y si se rechaza $H_0$, entonces seguimos con la familia de hipótesis + **post-hoc**.