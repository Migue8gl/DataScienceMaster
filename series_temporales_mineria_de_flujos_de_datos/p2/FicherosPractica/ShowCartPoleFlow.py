#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 17:37:42 2023

@author: manupc
"""

from river import stream


# Nombre del fichero de datos
datafile= 'CartPoleInstances.csv'

    
# Función que devuelve una lista con los nombres de los atributos de entrada
def AttributeNames():
    return ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Angular Velocity']

# Función que devuelve el nombre del atributo objetivo (clase)
def ClassName():
    return 'action'


# Conversores de datos para el flujo con  iter_csv
converters= {}
for i in AttributeNames():
    converters[i]= float
converters[ClassName()]= lambda x:int(float(x))



flujo= stream.iter_csv(datafile, converters= converters, target=ClassName())
for x,y in flujo:
    print('---------------------')
    print('Entradas:\n', x)
    print('Salidas: ', y)

