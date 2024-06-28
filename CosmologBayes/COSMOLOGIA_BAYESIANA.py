#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:04:30 2024

@author: mario
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 13:25:31 2024

@author: mario
"""
#from __future__ import division

#MODULOS
#from lnprob_function import lnprob


import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import math
from scipy.integrate import quad
from scipy.optimize import differential_evolution
from scipy.stats import norm


#LECTURA DE ARCHIVOS


#AQUI LEEMOS LOS ARCHIVOS DE CRONOMETROS
Hdecronometros=open("HzDA.txt")
rdeh=[]
hdez=[]
error=[]

for linea in Hdecronometros:
    vectora = linea.split()
    hdez.append(float(vectora[1]))
    rdeh.append(float(vectora[0]))
    error.append(float(vectora[2]))


print('el redshift de las mediciones de Hz es',rdeh)


#Aqui leemos nuestra Nueva muestra de Lentes de 143 sistemas


lentes3=open("Nuevos_lentes-FINAL_Dobs_entre05y1.txt")
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


#print('todos los sistemas son',sistema)
    
#aass=sorted(dispvel)
#print('Los rangos de sigma van entre',aass)


#Aqui pongo el valor de omega de materia oscura para JOINT de BAO, CMB y HZ
Om0Pl=0.279
eOm0Pl=0.021

#Aqui pongo el valor de omateria barionica segun el joint Hz_CMB_BAO para CPL
Ob0CPL=0.0513
eOb0CPL=0.0056

#DEFINIMOS LOS MODELOS COSMOLÓGICOS

#Aqui defino Ez para una energia oscura para el modelo estandar de cosmologia
def EzLCDM(ocdm,ob0,or0,z):
	return math.sqrt(ocdm*pow(1.0+z,3.0) + ob0*pow(1.0+z,3.0) + or0*pow(1.0+z,4.0) +(1.0-ocdm - ob0 - or0))


#Aqui defino Ez para una energia oscura constante general
def EzwCDM(om0,z,w):
	return math.sqrt(om0*pow(1.0+z,3.0) + (1.0-om0)*pow(1.0+z,3.0*(1.0+w)))

neff0=3.04

#Define parametro de densidad de radiacion
def omr(h):
	return 2.469e-5*pow(h,-2.0)*(1.0+0.2271*neff0)


def EzCPL(om0,ob0,h,w0,w1,z):
    aa=(-3.0*w1*z)/(1.0+z)
    aas= om0*pow(1.0+z,3.0) + ob0*pow(1.0+z,3.0)  + omr(h)*pow(1.0+z,4.0)  + (1 - om0 -ob0 - omr(h) )*pow(1.0+z,3.0*(1.0+w0+w1))*np.exp(aa)
    #return pow(aas,0.5)
    return pow(max(aas, 1e-4), 0.5)

def HzCPL(om0,ob0,h,w0,w1,z):
    xxx= EzCPL(om0,ob0,h,w0,w1,z)*h*100
    return xxx

#Aqui defino lel inverso de Ez
def u(om0,ob0,h,w0,w1,z):
	return 1.0/EzCPL(om0,ob0,h,w0,w1,z)


#################################
#La definicion de arriba la uso para la DA de los lentes mas abajo
	

	

# Aqui defino la ecuacion de los lentes

#Definicion de ecuacion de la lente
c2=pow(299792.4580,2.0) #en km/s
def D(rEi,veldpn):
	rerad=rEi*4.84814e-6
	return (c2*rerad)/(4.0*math.pi*pow(veldpn,2.0))


def eD(rEi,errorrEi,veldpn,eveldpn):
    www=pow(eveldpn,2.0)
    qqq=pow(www,0.5)
    rerad=rEi*4.84814e-6
    erEi=4.84814e-6*errorrEi
    xx=pow(erEi/rerad,2.0)
    tt=qqq/veldpn
    yy=4.0*pow(tt,2.0) 
    zz=D(rEi,veldpn)*math.sqrt(xx+yy)
    return zz 


#print('La funcion de propagacion del error para el primer valor de la muestra es',eD(radE[0],eradE[0],dispvel[0],edispvel[0]))

#Aqui defino la integral del DA del observador a la fuente
def DAol(om0,ob0,h, w0,w1, z2):
    #return quad(lambda x:u(om0,or0, ok0, oml0, n, x),0,z2)[0]
    return quad(lambda x: u(om0,ob0,h, w0,w1, x), 0, z2, epsabs=1e-9)[0] #el comando epsabs fue sugerido por CHatgpt

#Aqui defino la integral del DA del lente a la fuente
def DAls(om0,ob0,h, w0,w1, z, z2):
    return quad(lambda x:u(om0,ob0,h, w0,w1, x),z,z2, epsabs=1e-9)[0]


#Aqui defino la ecuacion teorica de los lentes como cociente de dos integrales 
def Dth(om0,ob0,h, w0,w1, z, z2):
    return DAls(om0,ob0,h, w0,w1, z, z2)/DAol(om0,ob0,h, w0,w1, z2)
#print('------------------------------------')




def ChiSL(om0,ob0,h, w0,w1):
	a=0.0
	for i in range(len(model)):
		x=(D(radE[i],dispvel[i]) - Dth(om0,ob0,h,w0,w1,zlente[i],zfuente[i]))/eD(radE[i],eradE[i],dispvel[i],edispvel[i])
		a +=pow(x,2.0)
	return a    


print('La chicuadrada  de Lentes para CPL con ciertos valores de omega es',ChiSL(0.25,0.05,0.7,-1.0,0.0))


###########################################################################################################################################################################################################################################################################
##################################################################################################################################################################################
########################################  H(z)   ##########################################################################################
##################################################################################################################################################################################
##################################################################################################################################################################################


#tambien se puede usar la muestra de cronometros
Hdecronometros=open("HzDA.txt")
rdeh=[]
hdez=[]
error=[]

for linea in Hdecronometros:
    vectora = linea.split()
    hdez.append(float(vectora[1]))
    rdeh.append(float(vectora[0]))
    error.append(float(vectora[2]))


def ChiCron(om0,ob0,h, w0,w1):
    a=0.0
    for i in range(len(rdeh)):
        x=(hdez[i] - HzCPL(om0,ob0,h,w0,w1,rdeh[i]))/error[i]
        a +=pow(x,2.0)
    return a


print('La chi de cronometros para Cotton es',ChiCron(0.25,0.05,0.7,-1.0,0.0))



####################################################
########## C H I   T O T A L #######################
####################################################


def ChiT(om0,ob0,h, w0,w1):
    return ChiCron(om0,ob0,h, w0,w1)
    #return ChiSL(om0,ob0,h, w0,w1)



#############################################################################################
#############################################################################################
###########    Aqui uso differential evolution  ##############################################
##############################################################################################
fun = lambda par: ChiT(par[0],par[1],par[2],par[3],par[4]) 
bounds = [(0.2,0.35), (0.000001,0.15),(0.6,0.8), (-4.0,1.0), (-4.0,4.0)]
result = differential_evolution(fun, bounds)
result.x, result.fun

print('El best fit usando differential evolution es',result)
print('best-fit de om0 usando dif evolution es',result.x[0])
print('best-fit de ob0 usando dif evolution es',result.x[1])
print('best-fit de h usando dif evolution es',result.x[2])
print('best-fit de w0 usando dif evolution es',result.x[3])
print('best-fit de w1 usando dif evolution es',result.x[4])


#resultado=result.x
resultado=[0.25,0.05,0.7,-1.0,0.0]

print('bfchi con differential evolution en Hz',ChiT(result.x[0],result.x[1],result.x[2],result.x[3],result.x[4]))



################################################################################################
#################################################################################################
#######################  D  A  T  O  S  ###########  CPL ############# H z ###########################
######################################################################################################
############################################################################################

#leyendo el archivo que contiene el muestreo de los parametros del MCMC de CPL
OOoc=[]
HHoc=[]
NNoc=[]
MMoc=[]
PPoc=[]




with open('Chains_CPL_Hz_CC.txt') as ochzdat:
    print('reading MCMC chains of CPL...wait')
    next(ochzdat)
    for line in ochzdat:
        cols = [float(x) for x in line.split()]
        om0, ob0, h, w0, w1 = cols[2], cols[3], cols[4], cols[5], cols[6]
        OOoc.append(om0)
        HHoc.append(ob0)
        NNoc.append(h)
        MMoc.append(w0)
        PPoc.append(w1)
    print('terminamos de leer las cadenas')

#P R O P A G A C I O N   D E   E r r or r para Ez y qz para Original Cardassian
E1sgoc=[]
E1sgnoc=[]
E3sgoc=[]
E3sgnoc=[]



per1Hz=[]
per2Hz=[]
per3Hz=[]
per4Hz=[]
per5Hz=[]

#def EzCPL(om0,ob0,h,w0,w1,z):
#    aa=(-3.0*w1*z)/(1.0+z)
#    aas= om0*pow(1.0+z,3.0) + ob0*pow(1.0+z,3.0)  + omr(h)*pow(1.0+z,4.0)  + (1 - om0 -ob0 - omr(h) )*pow(1.0+z,3.0*(1.0+w0+w1))*np.exp(aa)
    #return pow(aas,0.5)
#    return pow(max(aas, 1e-4), 0.5)

#Aqui defino vectores para graficar las funciones	
rz2=np.arange(0.0,2.6,0.1)#estos los uso para los plots

#calcula los errores de Ez y qz OC para diversos redshifts
for i in range(len(rz2)):
    HHzo=[] #ESte vector servira para guardar todos los de Ez a un redshift en particular para todas las lineas del muestreo
    #QQo=[] #ESte vector servira para guardar todos los de qz a un redshift en particular para todas las lineas del muestreo
    for j in range(len(OOoc)):
        HHzo.append(EzCPL(OOoc[j],HHoc[j],NNoc[j],MMoc[j],PPoc[j],rz2[i])*NNoc[j]*100.0)
        #QQo.append(qzcard(OOoc[j],HHoc[j],rz2[i],NNoc[j]))
    muHHzo,stdHHzo = norm.fit(HHzo)
    #muQQo, stdQQo = norm.fit(QQo)
    #bestfitqz= qzcard(0.25,0.66 ,rz2[i],0.23)
    #per1qHz.append(np.percentile(QQo,16))
    #per2qHz.append(np.percentile(QQo,84))
    #per3qHz.append(np.percentile(QQo,0.15))
    #per4qHz.append(np.percentile(QQo,99.85))
    #per5qHz.append(np.percentile(QQo,50))
    per1Hz.append(np.percentile(HHzo,16))
    per2Hz.append(np.percentile(HHzo,84))
    per3Hz.append(np.percentile(HHzo,0.15))
    per4Hz.append(np.percentile(HHzo,99.85))
    per5Hz.append(np.percentile(HHzo,50))
    #q1sgoc.append(bestfitqz + (per2qHz[i] - per5qHz[i]))
    #q1sgnoc.append(bestfitqz - (per5qHz[i] - per1qHz[i]))
    #q3sgoc.append(bestfitqz + (per4qHz[i] - per5qHz[i]) )
    #q3sgnoc.append( bestfitqz - (per5qHz[i] - per3qHz[i]))
    bestfit=EzCPL(0.2566,0.0608,0.7315,-1.4016, -0.5261, rz2[i])*0.7315*100
    E1sgoc.append(bestfit + (per2Hz[i] - per5Hz[i]))
    E1sgnoc.append(bestfit - (per5Hz[i] - per1Hz[i]))
    E3sgoc.append(bestfit + (per4Hz[i] - per5Hz[i]) )
    E3sgnoc.append( bestfit - (per5Hz[i] - per3Hz[i]))
    print('stdHHz=\t{0}  at redshift {1} para Original Cardassian'.format(stdHHzo,rz2[i]))

																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																				
print('Termine de propagar los errors de Hz para CPL con bf de Hz')

																					

#Ahora grafiquemos														

#EzLCDM(ocdm,ob0,or0,z)


figHzOC= plt.figure()
plt.xlim([0,2.5])
plt.plot(rz2, EzCPL(0.2566,0.0608,0.7315,-1.4016, -0.5261, rz2[i])*0.7315*100, 'r-', linewidth = 1,  label="$\mathrm{OHD_{\mathrm{hpl}}}$") #Grafica de Ez cardassiano teorica con best fit de hzpl con prior gaussiano de Riess
plt.plot(rz2, EzCPL(0.2566,0.0608,0.7315,-1.4016, -0.5261, rz2[i])*0.7315*100, 'y*', linewidth = 1,  label="$\mathrm{J3}$") #Grafica de Ez cardassiano teorica con best fit de hzpl con prior gaussiano de Riess
plt.plot(rz2, EzLCDM(0.259785860231,0.05,0.00001,rz2)*0.708912681029*100, 'ks', ms=4,  linewidth = 1,  label="$\Lambda$CDM") #Grafica de lcdm teorica con prior plano con la muestra de Hpl
plt.plot(rz2, E1sgoc, 'r--', linewidth = 1,  label="$1 \sigma$")
plt.plot(rz2, E1sgnoc, 'r--', linewidth = 1)
plt.plot(rz2, E3sgoc, 'r:', linewidth = 1,  label="$3 \sigma$")
plt.plot(rz2, E3sgnoc, 'r:', linewidth = 1)
plt.errorbar(rdeh, hdez, yerr=error, fmt='.b', linewidth = 1) #Grafica del error de hz
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.text(0.2, 240, '$\mathrm{CPL}$', fontsize=13) 
plt.legend(loc = 4, numpoints=1, fontsize=11)
plt.title("Redshift history of CPL")   # Establece el titulo del grafico
plt.xlabel("z", fontsize=11)   # Establece el titulo del eje x
plt.ylabel("H(z) km s$^{-1}$Mpc$^{-1}$", fontsize=11)   # Establece el titulo del eje y
figHzOC.savefig('HzCPL_CHAINS.pdf')
plt.show()










																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																						
      