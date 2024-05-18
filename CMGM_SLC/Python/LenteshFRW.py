#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 13:53:32 2022

@author: MARIO
"""
#MODULOS
from __future__ import division
import corner
import getdist
import numpy as np
from getdist import loadMCSamples
from numpy.linalg import inv
from sys import *
import emcee
from scipy.misc import derivative
from math import log10
from math import pi
from matplotlib.ticker import MaxNLocator
from matplotlib.pylab import hist, show
import time
import sys
from numpy import *
import matplotlib.pyplot as plt
import math
#import uncertainties as u
#from uncertainties import ufloat
from scipy.integrate import quad
from scipy.optimize import differential_evolution

######################################################################

#LECTURA DE ARCHIVOS



#Aqui leemos nuestra Nueva muestra de Lentes de 204 sistemas, se puede cambiar de muestra por alguna en especifico
#ver DOI: 10.1093/mnras/staa2760 (publication) para mas detalles

lentes3=open("Nuevos_lentes-FINAL.txt")
sistema=[]
surveysis=[]
zlente=[]
zfuente=[]
radE=[]
eradE=[]
dispvel=[]
edispvel=[]
model=[]

for linea in lentes3:
    vectora=linea.split()
    sistema.append(vectora[0])
    surveysis.append(vectora[1])
    zlente.append(float(vectora[2]))
    zfuente.append(float(vectora[3]))
    radE.append(float(vectora[4]))
    eradE.append(float(vectora[5]))
    dispvel.append(float(vectora[6]))
    edispvel.append(float(vectora[7]))
    model.append(vectora[8])
    
  
prior="Om_flat"
data="SLS_204_hfrw"
#data=aqui se puede especificar otra muestra observcional de alguna referencia antigua

########################################################
#Aqui creo un archivo para los datos 
arlen = open('datoslentes.txt', 'w')

fo = open("bestfits_{0}_{1}.txt".format(prior,data), 'w') 

#Aqui defino los valores de h para Riess y su error
h0=0.7324
dh0=0.0174

#Aqui defino lso valores para omega de materia segun planck y su error
#Ya esta actualizado a Planck2018
Om0Pl=0.3111
eOm0Pl=0.0056

################################################################################
#F    U    N    C    I    O    N    E    S  ######################################

#Aqui se define el modelo cosmologico hFRW 


# hFRW model 
def EzhFRW(om0, omL, z):
    omI=(1-omL-2*om0 + om0**2.0)/omL
    Ex = pow(omL,2)/4 + pow(1.0+z,3)*omL*om0 + pow(1.0+z,4)*omL*omI
    #a = omL/2 + pow(1.0+z,3)*omM + pow(Ex,1/2)
    if Ex > 0: #this is the constraint, where you would say a+b+c <=1000
        return np.sqrt(omL/2 + pow(1.0+z,3)*om0 + pow(Ex,1/2))
    else:
        return 10000 #some high value


print 'La funcion Ez para hFRW con valores de om0=0.3 y omL=0.7 es,',EzhFRW(0.2, 0.7, 0.3)
#Aqui defino Ez para una energia oscura constante
#def EzLCDM(om0,z):
#	return math.sqrt(om0*pow(1.0+z,3.0) + (1.0-om0))


#aqui defino el inverso de la funcion Ez del modelo estandar
#def InvEzLCDM(om0,z):
#	return 1.0/EzLCDM(om0,z)
	

#Aqui defino Ez para una energia oscura constante general
#def EzwCDM(om0,z,w):
#	return math.sqrt(om0*pow(1.0+z,3.0) + (1.0-om0)*pow(1.0+z,3.0*(1.0+w)))

#este es el modelo CPL
#def EzCPL(om0,w0,w1,z):
#    aa=(-3.0*w1*z)/(1.0+z)
#    vv= om0*pow(1.0+z,3.0) + (1.0-om0)*pow(1.0+z,3.0*(1.0+w0+w1))*np.exp(aa)
#    return math.sqrt(vv)

#Aqui defino las funciones para wCDM
def u(om0,omL,z):
	return 1.0/EzhFRW(om0,omL,z)

#################################
#La definicion de arriba la uso para la DA de los lentes mas abajo
	

	

# Aqui defino la ecuacion de los lentes

#Definicion de D segun Cao
c2=pow(299792.4580,2.0) #en km/s
def D(rEi,veldpn):
	rerad=rEi*4.84814e-6
	return (c2*rerad)/(4.0*math.pi*pow(veldpn,2.0))#las variables de D las nombre de diferente manera


#Aqui calculo la ecuación de propagacion del error
def eD(rEi,errorrEi,veldpn,eveldpn):
    www=pow(eveldpn,2.0)
    qqq=pow(www,0.5)
    rerad=rEi*4.84814e-6
    erEi=4.84814e-6*errorrEi
    xx=pow(erEi/rerad,2.0)
    tt=qqq/veldpn
    yy=4.0*pow(tt,2.0) 
    zz=D(rEi,veldpn)*math.sqrt(xx+yy)
    return zz #Recordar que los terminos estan al reves del calculo que hice a mano
    
#Aqui defino la integral del DA del observador a la fuente
def DAol(om0,omL,z2):
    return quad(lambda x:u(om0,omL,x),0,z2)[0]


#Aqui defino la integral del DA del lente a la fuente
def DAls(om0,omL,z,z2):
    return quad(lambda x:u(om0,omL,x),z,z2)[0]


#Aqui defino la ecuacion teorica de los lentes como cociente de dos integrales 
def Dth(om0,omL,z,z2):
    return DAls(om0,omL,z,z2)/DAol(om0,omL,z2)
#print('------------------------------------')


def ChiSL(om0,omL):
	a=0.0
	for i in range(len(model)):
		x=(D(radE[i],dispvel[i]) - Dth(om0,omL,zlente[i],zfuente[i]))/eD(radE[i],eradE[i],dispvel[i],edispvel[i])
		a +=pow(x,2.0)
	return a  


#print('La chicuadrada  de wcdm con 0m=0.31 y w=-1 es',ChiSL(0.31,-1.0))

#print('La funcion CHi2 con valores de H(z) es',ChiSL(0.2146,0.5738))

#print('La funcion CHi2 con valores de Panteon es',ChiSL(0.0344,0.685))

#############################################################################################
#############################################################################################
###########    Aqui uso una paqueteria de scipy.optimize differential evolution  ##############################################################################################
##############################################################################################
fun = lambda par: ChiSL(par[0],par[1]) 
bounds = [(0.0001,0.2), (0.21,0.9)]
result = differential_evolution(fun, bounds)
result.x, result.fun

print 'El best fit usando differential evolution es',result
print 'best-fit de om0 usando dif evolution es',result.x[0]
print 'best-fit de omL usando dif evolution es',result.x[1]
#print 'best-fit de w1 usando dif evolution es',result.x[2]


#resultado=result.x
resultado=[0.15,0.55]

print 'bfchi con differential evolution',ChiSL(result.x[0],result.x[1])

################################################################################################################################
######################################################################################################
#############################################################################################################################
###############################  MONTECARLO #############################
##############################################################################################################################
#Aqui se puede utilizar el método Montecarlo de tu elección para constreñir los parámetros cosmológicos dados por la función ChiSL(om0,omL)

        
print '\n----- end ... Se acabo el codigo  -----'
