- #overfitting | #variance | #tradeoffvariancebias

- El **overfitting** o sobreajuste se conoce como el fenómeno por el cual ajustamos demasiado nuestra función a los casos de entrenamiento, de forma que reducimos el error de entrenamiento o **error-in-sample**. 
- Esto se hace aumentando el orden del polinomio utilizado, de manera que se usan los [[degrees of freedom|grados de libertad]] adicionales para modelar el ruido de nuestras muestras. Al ajustar el **ruido**, nuestro modelo está aprendiendo patrones inexistentes o erróneos.
- Este proceso da como lugar a un modelo cuya capacidad de generalizar será atroz. Nuestras predicciones tendrán una [[variance and standard deviation|varianza]] muy elevada.
- Se suele decir que cuando nuestro modelo está sobreajustando, en realidad memoriza, de forma que no tendrá capacidad de predicción dadas nuevas muestras.
- Si mientras entrenamos, el $E_{in}$ disminuye, pero el $E_{out}$ aumenta, entonces estamos sobreajustando.
- Sesgar un modelo significa no ajustar tanto los datos, añadir un [[bias#Sesgo en un modelo|sesgo]], para poder adaptarnos mejor a nuevos datos. Se requiere ajustar una función, pero no demasiado, ya que necesitamos que generalice.

![[bias-variance-example.png]]

![[variance-bias-example2.png]]