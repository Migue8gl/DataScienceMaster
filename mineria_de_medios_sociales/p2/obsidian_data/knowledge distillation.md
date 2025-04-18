- #ml | #modelcompression | #knowledgedistillation

# Concepto
- El destilamiento de conocimiento es una técnica que tiene como objetivo reducir el tamaño del modelo original, manteniendo la máxima cantidad de conocimiento posible sin reducir las métricas de calidad.
- Se basa en entrenar un modelo **estudiante** de forma que mimetice el modelo **profesor**.
![[knowledge distillation.png|500]]

# Divergencia KL
- La idea principal detrás de esta técnica es la de cuantizar cuanta información se pierde cuando se utiliza la distribución del modelo profesor para aproximar la distribución del modelo estudiante.
- La métrica principal que cuantiza si una distribución se aleja de otra es [[distances|la distancia KL]]. La idea es entrenar minimizando esta distancia.
- Para ello se entrena al modelo *teacher* o profesor. Después se define la función de perdida $KL$.
```python
def KL_loss(student_logits, teacher_logits):
    
    # convert teacher model outputs to probabilities 
    p_teacher = F.softmax(teacher_logits, dim=1)
    
    # convert student model outputs to probabilities 
    p_student = F.log_softmax(student_logits, dim=1)
    
    # compute KL divergence loss (PyTorch's method)
    loss = F.kl_div(p_student, p_teacher, reduction='batchmean')
    
    return loss
```
- La función [[softmax]] acepta los [[logits]] (*outputs*) para transformarlos en probabilidades. Después se aplica $KL$ a las distribuciones generadas.
- Al entrenar, simplemente se utiliza esta pérdida para optimizar.