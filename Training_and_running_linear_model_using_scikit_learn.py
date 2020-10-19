# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 09:15:37 2020

@author: CAMILO HURTADO
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model

def prepare_country_stats(oecd_bli, gdp_per_capita):
    #Filtramos y dejamos solo las filas de "INEQUALITY" con valor TOT
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    #Ponemos como índice la columna "Country". Los valores de la columna
    #Indicator se van a convertir en columnas
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    #Cambiamos el nombre de una de las columnas de gpd_per_capita
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    #Eliminamos el índice numérico generado automáticamente
    gdp_per_capita.set_index("Country", inplace=True)
    #Unimos ambos dataframes de manera horizontal (uno a la izquierda y el otro a
    #la derecha)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    #Ordenamos el nuevo dataframe de acuerdo al valor de la columna
    #GDP per capita
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    #Se van a eliminar las siguientes filas
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    #Solo se retornan dos columnas
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]

#Cargamos los archivos
oecd_bli = pd.read_csv("C:/Users/CAMILO HURTADO/Documents/PERSONAL/\
MACHINE LEARNING/HANDS-ON MACHINE LEARNING WITH SCIKIT-LEARN KERAS TF/\
handson-ml2-master/datasets/lifesat/oecd_bli_2015.csv", thousands=',')

gdp_per_capita = pd.read_csv("C:/Users/CAMILO HURTADO/Documents/PERSONAL/\
MACHINE LEARNING/HANDS-ON MACHINE LEARNING WITH SCIKIT-LEARN KERAS TF/\
handson-ml2-master/datasets/lifesat/gdp_per_capita.csv", thousands = ',',
delimiter='\t', encoding='latin1',na_values='n/a')


#Preparamos los datos
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]


#Visualizamos los datos
country_stats.plot(kind='scatter',x="GDP per capita", y='Life satisfaction')


#Seleccionamos un modelo lineal
model = sklearn.linear_model.LinearRegression()

#Entrenamos el modelo
model.fit(X,y)

#Se pueden obtener los parámetros del modelo (que son dos por ser un modelo líneal de una sola variable)
#usando las siguientes instruccions
model.intercept_ #array([4.8530528]) Un país con 0 GDP tendría una satisfacción de vida de 4.85
model.coef_ #array([[4.91154459e-05]]) La línea es casi plana, casi cero.

#Hacemos una predicción para Chipre
X_new = [[22587]] #GDP per capita de Chipre
print(model.predict(X_new)) #[[5.96242338]]
