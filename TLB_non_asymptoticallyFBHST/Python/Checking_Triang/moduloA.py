import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d  # este es mi preferido
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


#######
# 
#   (dr/dphi)^2 = r^2*(1/r + r^2/b^2 -1)*(1 + eta*r^2)
#
#   rF = r*mu,  bF = b*mu, etaF = eta*beta/mu^2
#   mu = 2M
#   etaF, beta parámetros del modelo de Horndeski
#   eta = 0 reduce al caso de la métrica de Sch
#
#
######

def eq():
    """ 
    Definiendo uf con phi=phi+delta para sympy
    Es necesario definir:
    phi, mu, b, eta, beta, epsilon, delta = sp.symbols('phi, mu, b, eta, beta, epsilon, delta', real=True)
    """
    phi, mu, b, eta, beta, epsilon, delta = sp.symbols('phi, mu, b, eta, beta, epsilon, delta', real=True)

    uEq = (15*epsilon**2*mu**2*sp.pi*sp.cos(delta+phi))/(32*b**3) +\
          sp.cos(delta+phi)*((epsilon*mu*sp.cos(delta+phi))/(2*b**2) -\
          (b*epsilon*eta*sp.cot(delta+phi))/(2*beta) +\
          (epsilon*mu*sp.sec(delta+phi))/(2*b**2))+sp.sin(delta+phi)/b +\
          sp.cos(delta+phi)*((-15*epsilon**2*mu**2*(delta+phi))/(16*b**3) +\
          (epsilon**2*mu*eta*sp.cos(delta+phi))/(2*beta) +\
          (b**3*epsilon**2*eta**2*sp.cot(delta+phi))/(8*beta**2) +\
          (epsilon**2*mu*eta*sp.csc((delta+phi)/2)**2)/(8*beta) -\
          (b**3*epsilon**2*eta**2*sp.cot(delta+phi)*sp.csc(delta+phi)**2)/(8*beta**2) -\
          (epsilon**2*mu*eta*sp.sec((delta+phi)/2)**2)/(8*beta) -\
          (3*epsilon**2*mu**2*sp.sin(2*(delta+phi)))/(32*b**3) +\
          (5*epsilon**2*mu**2*sp.tan(delta+phi))/(8*b**3))
    return uEq

# datos de phi
def phidat(theta, delta, Nptos=200):
    """
    delta = 0 -> [theta, pi-theta]
    delta > 0 -> [theta, pi/2]
    delta < 0 -> [pi/2, pi-theta]
    """
    if delta==0:
        dat = np.linspace(theta, np.pi-theta, Nptos)
    elif delta > 0:
        dat = np.linspace(theta, np.pi/2, Nptos)
    else:
        dat = np.linspace(np.pi/2, np.pi-theta, Nptos)
    return dat

# Solución Numérica
def systemP(phi, V, arg):
    """
    Ecuación diferencial rama positiva
    """
    r = V
    eta, b, = arg

    term = r*(1+eta*r**2)*(b**2-b**2*r+r**3)/(b**2)
    
    if term<0:
        if np.abs(term)<1e-10:
            term = np.abs(term)
            drdtheta = np.sqrt(term)
        else:
            drdtheta = [0]
    else:
        drdtheta = np.sqrt(term)
        
    return drdtheta


def systemN(phi, V, arg):
    """
    Ecuación diferencial rama negativa
    """
    r = V
    eta, b, = arg

    term = r*(1+eta*r**2)*(b**2-b**2*r+r**3)/(b**2)
    
    if term<0:
        if np.abs(term)<1e-10:
            term = np.abs(term)
            drdtheta = -np.sqrt(term)
        else:
            drdtheta = [0]
    else:
        drdtheta = -np.sqrt(term)
        
    return drdtheta


def integracion(param, lim, R_func, s1=systemN, s2=systemP, Nptos=500,
               Rtol=1e-18, Atol=1e-21, fStep=1e-08): # , 
    """
    Resuelve ambas rámas de la ecuación diferencial
    """
        
    # integración
    b, db, eta = param
    thetamin, thetamax = lim


    if thetamin>thetamax:
        # cambiando derivadas cuando se integra de mayor a menor
        sD, sI = s2, s1
    else:
        sD, sI = s1, s2
    
    print('Integración hacia la derecha')
    ######## Solución con pendiente negativa irá hacia la derecha
    r0 = R_func(0, thetamin, b, eta)
    # print(r0)
    arg = [eta, b+db]
    thetaspan = np.linspace(thetamin, thetamax, Nptos)
    
    solD = solve_ivp(sD, [thetamin, thetamax], [r0], args=(arg,), t_eval=thetaspan, first_step=fStep,
                         method='RK45', rtol=Rtol, atol=Atol)
    
    print('Integración hacia la izquierda')
    ######### Solución con pendiente positiva irá hacia la izquierda
    r0 = R_func(0, thetamax, b, eta)
    # print(r0)
    arg = [eta, b+db]
    thetaspan = np.linspace(thetamax, thetamin, Nptos)

    solI = solve_ivp(sI, [thetamax, thetamin], [r0], args=(arg,), t_eval=thetaspan, first_step=fStep,
                         method='RK45', rtol=Rtol, atol=Atol)

    return solD.t, solD.y[0], solI.t, solI.y[0]


def full(param, lim, R_func, s1=systemN, s2=systemP, Nptos=500,
               Rtol=1e-09, Atol=1e-12, fStep=1e-08):
    """
    Construyendo la solución completa

    param = []
    """
    
    theta1, r1, theta2, r2 = integracion(param, lim, R_func, s1=s1, s2=s2, 
                                 Nptos=Nptos, Rtol=Rtol,
                                 Atol=Atol, fStep=fStep)    

    thetaMin, thetaMax = lim
    if thetaMin>thetaMax:
        thetamin = thetaMax
        thetamax = thetaMin
        itD = interp1d(theta2, r2, kind='quadratic')
        itI = interp1d(theta1, r1, kind='quadratic')
    else:
        thetamin = thetaMin
        thetamax = thetaMax
        itD = interp1d(theta1, r1, kind='quadratic')
        itI = interp1d(theta2, r2, kind='quadratic')

    func = lambda theta: itD(theta)-itI(theta)

    thetaC = fsolve(func, thetamin)
    print(thetaC)

    radD = np.linspace(thetamin, thetaC, Nptos, endpoint=False)
    radI = np.linspace(thetaC, thetamax, Nptos)
    datD = itD(radD)
    datI = itI(radI)
    
    radFull = np.concatenate((radD, radI), axis=None)
    datFull = np.concatenate((datD, datI), axis=None)
    itFull = interp1d(radFull, datFull, kind='quadratic')

    return itFull, radFull, datFull , [theta1, r1, theta2, r2]


def triang(paramI, deltbValV, deltaphiV, R_func, s1=systemN, s2=systemP, Nptos=500,
           Rtol=1e-09, Atol=1e-12, fStep=1e-08):
    """
    se construye el triángulo
    Variables
    etaVal -> valor de eta, Sch es etaVal=0
    thetamin, thetamax -> ángulos de la integración, recordar que thetamax = np.pi-thetamin
    bVal -> parámetro de impacto, este luego se modifica a cada caso

    Vectores de entrada
    deltbValV -> deltas para la variación del parámetro de impacto: deltbValV = [0, #1, #1] los últimos dos son iguales
    deltaphiV -> variación de los ángulos de integración. 
                 Normalmente se integra de thetamin -> pi, o thetamax -> 0, sin embargo lo adecuado es hasta pi/2
                 deltaphiV = [thetamin, pi/2, pi/2], yo uso deltaphiV = [thetamin, 1, 1] pq necesito unir trayectorias
    """

    etaVal, thetamin, thetamax, bVal = paramI
    deltb1, deltb2, deltb3 = deltbValV
    deltaphi1, deltaphi2, deltaphi3 = deltaphiV

    Limt = [[thetamin, np.pi-deltaphi1], 
            [thetamin, np.pi-deltaphi2],  # np.pi/2
            [thetamax, 0+deltaphi3] # np.pi/2
            ]
    
    param = [[bVal, deltb1, etaVal],
             [bVal, deltb2, etaVal],
             [bVal, deltb3, etaVal]
            ]
    
    DatosS = []
    DatosSF = []
    for i in range(3):  # 0 -> horizontal, 1-> monotona decreciente, 2-> monotona creciente
        itFull, radFull, datFull, datF = full(param[i], Limt[i], R_func, s1=s1, s2=s2, 
                                             Nptos=Nptos, Rtol=Rtol,
                                             Atol=Atol, fStep=1e-08)
        DatosS.append([itFull, radFull, datFull])
        DatosSF.append(datF)
    
    return DatosS, DatosSF


def Intind(paramI, deltbValV, deltaphiV, R_func, s1=systemN, s2=systemP, Nptos=500,
               Rtol=1e-09, Atol=1e-12, fStep=1e-08):
    """
    se integra individual
    Variables
    etaVal -> valor de eta, Sch es etaVal=0
    thetamin, thetamax -> ángulos de la integración, recordar que thetamax = np.pi-thetamin
    bVal -> parámetro de impacto, este luego se modifica a cada caso

    Vectores de entrada
    deltbValV -> deltas para la variación del parámetro de impacto: deltbValV = [0, #1, #1] los últimos dos son iguales
    deltaphiV -> variación de los ángulos de integración. 
                 Normalmente se integra de thetamin -> pi, o thetamax -> 0, sin embargo lo adecuado es hasta pi/2
                 deltaphiV = [thetamin, pi/2, pi/2], yo uso deltaphiV = [thetamin, 1, 1] pq necesito unir trayectorias
    """
     
    etaVal, thetamin, thetamax, bVal = paramI
    deltb1, deltb2, deltb3 = deltbValV
    deltaphi1, deltaphi2, deltaphi3 = deltaphiV

    ######### SOLUCIÓN HORIZONTAL #######
    lim = [thetamin, np.pi-deltaphi1]  # siempre ha de ser thetamin
    param = [bVal, deltb1, etaVal]  # deltb1 = 0 siempre
    theta1, r1, theta2, r2 = integracion(param, lim, R_func, s1=s1, s2=s2, 
                                             Nptos=Nptos, Rtol=Rtol,
                                             Atol=Atol, fStep=fStep)

    ######### SOLUCIÓN PENDIENTE NEGATIVA #######
    param = [bVal, deltb2, etaVal]
    lim = [thetamin, np.pi-deltaphi2]  # np.pi/2
    theta12, r12, theta22, r22 = integracion(param, lim, R_func, s1=s1, s2=s2, 
                                                 Nptos=Nptos, Rtol=Rtol,
                                                 Atol=Atol, fStep=fStep)

    ######### SOLUCIÓN PENDIENTE POSITIVA #######
    lim = [thetamax, 0+deltaphi3]  # np.pi/2
    param = [bVal, deltb3, etaVal]  # debe ser deltb2 = deltb3
    theta13, r13, theta23, r23 = integracion(param, lim, R_func, s1=s1, s2=s2, 
                                                 Nptos=Nptos, Rtol=Rtol,
                                                 Atol=Atol, fStep=fStep)
    
    datosF = [[theta1, r1, theta2, r2], [theta12, r12, theta22, r22], [theta13, r13, theta23, r23]]

    return datosF
