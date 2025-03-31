#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 17:37:42 2023

@author: manupc
"""

import gymnasium as gym
import numpy as np

# Nombre del entorno
envName= 'CartPole-v1'


# Función de selección de una acción a partir de la observación obs de entrada
def ActionSelection(obs):
    action= np.random.randint(low=0, high=2) # Selección de acción 0/1 aleatoria
    return action


# Creación del entorno
env= gym.make(envName, render_mode='human')
    

# Jugar un episodio
obs, _= env.reset() # Inicialización del entorno
env.render() # Renderizar entorno para visualización
R= 0 # Recompensa total obtenida
truncated, done= False, False
while not (truncated or done):
    
    # Selección de acción
    action= ActionSelection(obs)
    
    # Ejecución de la acción y obtención de recompensa y siguiente estado
    obs, r, done, truncated, _= env.step(action)
    env.render() # Renderizar entorno para visualización
    R+= r # Actualización de recompensa total (performance)
    
print('El episodio (partida) ha terminado con recompensa (performance)= {}'.format(R))
    




print('Presiona [ENTER] para terminar')
_= input()

env.close() # Cerrar entorno