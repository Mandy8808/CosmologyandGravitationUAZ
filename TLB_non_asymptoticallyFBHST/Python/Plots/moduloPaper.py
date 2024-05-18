import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.interpolate import interp1d  # este es mi preferido
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


######################
#  VARIABLES
#  u=b/r;   l=L b^2/rS=Lb/rS;    rS=2GM/b
#
#  Sch. dSitter
#  (du/dphi)^2+u^2(1-rS u)=1+Lb/3
#
#  Hordeski
#  (du/dphi)^2+u^2(1-rS u)+Lb*(u*rS+1/u^2)=1+Lb
#################

##
# Series Analíticas
##

# u serie para de Sitter
def SchDes_uSer(arg):
    """
    Lb -> lambda barra : L*b**2
    p -> phi
    d -> delta
    rs -> 2Gm/b
    """
    d, p, Lb, rs = arg
    ord0 = np.sin(p-d)*(1+Lb/6-Lb**2/72)
    ord1 = rs*(1+Lb/3)*(3+np.cos(2*p-2*d))/4
    ord2 = rs**2*(1+Lb/2+Lb**2/24)*(37*np.sin(p-d)-3*np.sin(3*p-3*d)+30*(np.pi-2*(p-d))*np.cos(p-d))/64

    uSerie = ord0+ord1+ord2
    return uSerie

# Angulo para de sitter
def SchdeSitter_Ang_Analit(arg):
    """
    d -> delta
    p2 -> phi2
    b1, b3 -> impact parameters
    L -> Lambda física
    """
    d, p2, b1, b3, L, G, M, c = arg
    
    csc = lambda x: 1/np.sin(x)
    sec = lambda x: 1/np.cos(x)
    cot = lambda x: np.cos(x)/np.sin(x)

    ord1 = 4*G*M*((np.cos(d+p2)+np.sin(d))/b3-np.cos(p2)/b1)/c**2+\
        G*M*L*(2*b1/(np.cos(p2)-1)-4*(b1-b3*np.cos(d))*np.cos(p2)-2*b3/(1+np.cos(d+p2))+\
        b3*csc((d+p2)/2)**2+b1*sec(p2/2)**2+4*b3*(np.sin(d)-np.sin(d)*np.sin(p2)+sec(d)*np.tan(d)))/(6*c**2)+\
        G*M*L**2*(b1**3*(np.cos(2*p2)-7)*cot(p2)**3*csc(p2)+\
        b3**3*(-((-7+np.cos(2*(d+p2)))*cot(d+p2)**3*csc(d+p2))+(7+np.cos(2*d))*sec(d)*np.tan(d)**3))/(36*c**2)
    
    ord2 = G**2*M**2*(15*(b1-b3)*(b1+b3)*(np.pi-2*p2)+b3**2*np.sin(2*p2)-2*b1**2*np.cos(p2)*np.sin(2*d+p2))/(4*b1**2*b3**2*c**4)+\
        G**2*M**2*L**2*(2*b1**2*(-30*np.pi+60*p2+97*cot(p2))*csc(p2)**4+62*b1**2*np.cos(3*p2)*csc(p2)**5+
        b3**2*(12*(5*(np.pi-2*(d+p2))-11*cot(d+p2))*csc(d+p2)**4+2*sec(d)**5*(60*d*np.cos(d)-97*np.sin(d)+\
        31*np.sin(3*d))-31*csc(d+p2)**6*np.sin(4*(d+p2))))/(384*c**4)+\
        G**2*M**2*L*(-15*np.pi*csc(p2)**2+30*p2*csc(p2)**2+cot(p2)*(-30+32*csc(p2)**2)+15*np.pi*csc(d+p2)**2-\
        30*d*csc(d+p2)**2-30*p2*csc(d+p2)**2+cot(d+p2)*(30-32*csc(d+p2)**2)+30*d*sec(d)**2-\
        2*np.sin(2*d)+2*np.sin(2*p2)-2*np.sin(2*(d+p2))+30*np.tan(d)-32*sec(d)**2*np.tan(d))/(24*c**4)

    ser = ord1 + ord2
    return ser


####
# Ecuaciones
####

# Sch dSitter

def eqSchdSitter(phi, u, arg):
    """
    phi -> ángulo (variable independiente)
    u -> variable dependiente 
    arg -> constantes de entrada
        arg = [sig, uC, Lb, rS] 
            uC -> valor de u donde du=0
            Lb -> valor de Lambda barra: Lb=L*b^2
            rS -> Sch. radius
    
    IMPORTANTE: la integración siempre se ha de hacer de un ángulo menor a uno mayor
    """
    sig, _, Lb, rS, = arg

    duSquared = 1+Lb/3-u**2*(1-u*rS)
    duSquared = 0 if duSquared<=0 else duSquared 

    du = np.sqrt(duSquared) if sig=='+' else -np.sqrt(duSquared)
    
    return du

def U0(phiMax, argU0, U0S, sig, Nptos=5000, Rtol=1e-15, Atol=1e-18, fStep=1e-15):
    """
    Se encuentra la condición incial en los extremos usando el hecho de que la derivada du/dphi=0 en phi=pi/2

    phiMax -> posición angular del extremo, es decir PhiMin
    U0S -> semilla usada para encontrar la solución du/dphi=0, notar que esta se obtiene a partir de la serie
    sig -> signo de la rama que se usará para resolver la EDO
    argU0 -> [_, phiMin, Lb, rS]
            phiMin -> fijado a pi/2
            Lb -> valor de Lambda barra: Lb=L*b^2
            rS -> Sch. radius

    """
    _, phiMin, Lb, rS = argU0
    
    func = lambda u: (1+Lb/3-u**2*(1-u*rS))
    U01 = fsolve(func, U0S, xtol=1e-10)[0]  # u(pi/2) donde du=0
    du = 1e-12  # delta para moverlo del pto de equilibrio
    thetaspan = np.linspace(phiMin, phiMax, Nptos)
    arg = [sig[0], None, Lb, rS]
    sol1 = solve_ivp(eqSchdSitter, [phiMin, phiMax], [U01-du], args=(arg,), t_eval=thetaspan,
                     first_step=fStep, method='DOP853', rtol=Rtol, atol=Atol)
    u0 = sol1.y[0][-1]
    phi0 =  sol1.t[-1]
    return u0, phi0


# Resolución
def integracion(i, paramF, geoDat, DatLim, Nptos=5000, Rtol=1e-15, Atol=1e-18, fStep=1e-15): # , 
    """
    Resuelve ambas rámas de la ecuación diferencial

    paramF -> [Lambda, RSch] cantidades físicas
              Lambda -> lambda barra : constante cosmológica
              RSch -> radio de Sch de la fuente estudiada
    
    geoDat -> [delta, b1, b2] datos de la geodesica
            delta -> delta
            b1 -> parámetro correspondiente a la geodesica que se calcula
            b2 -> parámetro de impacto usado para escalar todo
                  Para construir el triángulo tomamos b2 = al parámetro de impacto de C1
                  Para calcular geodesicas y no afectar en nada, b2=b1

    DatLim -> [Phimin, Phimax]  es un vector con los diferentes límites de integración
             PhiMin, PhiMax -> ángulos de integración
    
    Out: Perfiles de ambas ramas como u=b/r
    """
    #### Limites de integración
    PhiMin, PhiMax = DatLim

    #### Rama de soluciones (pendiente) 
    sig = ['-', '+'] if PhiMin>PhiMax else ['+', '-']

    #### Parametros
    Lambda, RSch = paramF
    delta, b1, b2 = geoDat
    
    #### Factor de conversión
    fac = b1/b2

    #### Cantidades Adim
    Lb = Lambda*b1**2
    rS = RSch/b1

    #### Condición Inicial. Notar que se usa b2, calculandose solo para la C1 
    phiCond = np.pi/2
    argU0 = [delta, phiCond, Lambda*b2**2, RSch/b2]
    U0s = SchDes_uSer(argU0)
    u0, phi0 = U0(PhiMin, argU0, U0s, sig, Nptos=Nptos, Rtol=Rtol, Atol=Atol, fStep=fStep)

    # test
    #argU0 = [delta, PhiMin, Lambda*b2**2, RSch/b2]
    #U0s = SchDes_uSer(argU0)
    #print(u0, U0s, phi0, PhiMin)
    #print('u0', u0, 'fac', fac)
    u0 = u0*fac # Condición inicial
    
    # Encontrando el u crítico correspondiente a la geodesica que se calcula
    # donde du/dphi=0
    func = lambda u: (1+Lb/3-u**2*(1-u*rS))
    uC = fsolve(func, u0, xtol=1e-10)[0]  # uC donde du=0
    
    ##### Integración hasta uC
    # condicional
    def ucrit(phi, u, arg): 
        _, uC, _, _, = arg
        return u-uC
    ucrit.terminal = True
    ucrit.direction = 1
    
    thetaspan = np.linspace(PhiMin, PhiMax, Nptos)
    arg = [sig[0], uC, Lb, rS]
    sol1 = solve_ivp(eqSchdSitter, [PhiMin, PhiMax], [u0], args=(arg,), t_eval=thetaspan, first_step=fStep,
                     dense_output=True, events=ucrit, method='DOP853', rtol=Rtol, atol=Atol) #Radau 
    
    ### Checking that b3 is not equivalent to inf 
    #print('evento', sol1.t_events, sol1.t[-1], sol1.y[0][-1]/fac)

    ##### Integración a partir de uC
    du = 1e-12
    u0 = uC-du if i==0 else sol1.y[0][-2] # Condición inicial
    PhiMin = np.pi/2 if i==0 else sol1.t[-1]
    thetaspan = np.linspace(PhiMin, PhiMax, Nptos)

    arg = [sig[1], None, Lb, rS]
    sol2 = solve_ivp(eqSchdSitter, [PhiMin, PhiMax], [u0], args=(arg,), t_eval=thetaspan, first_step=fStep,
                     dense_output=True, method='DOP853', rtol=Rtol, atol=Atol) #'DOP853' 'RK45'
    
    solFphi = np.concatenate((sol1.t, sol2.t[1:]))
    solFu = np.concatenate((sol1.y[0]/fac, (sol2.y[0]/fac)[1:]))
    return solFphi, solFu


def Intind(paramF, Val_b, delta, Nptos=5000,
               Rtol=1e-15, Atol=1e-20, fStep=1e-20):
    """
    se construye el triángulo
    Vectores de entrada
    paramF -> [Lambda, RSch, Phimin, Phimax] cantidades en unidades físicas
              Lambda -> constante cosmológica
              RSch -> radio de Sch. del objeto bajo estudio
              Phimin, Phimax -> ángulos de la integración, recordar que Phimax = np.pi-Phimin

    Val_b -> [b1, b2, b3] parámetros de impacto de cada geodésica
              b1 -> geodésica horizontal
              b2=b3 -> geodésicas restantes del triángulo
              
    delta -> [d1, d2, d3 ]
             d1 la geodesica horizontal, esta la usaremos para obtener el u0 en los
             extremos Phimin, Phimax de la geodesica

    Términos
    geoDat -> [delta, b1, b2] datos de la geodesica
            delta -> delta
            b1 -> parámetro correspondiente a la geodesica que se calcula
            b2 -> parámetro de impacto usado para escalar todo
                  Para construir el triángulo tomamos b2 = al parámetro de impacto de C1
                  Para calcular geodesicas y no afectar en nada, b2=b1
    """

    Lambda, RSch, Phimin, Phimax = paramF
    b1, b2, b3 = Val_b
    d1, d2, d3 = delta

    # Cantidades Adim
    paramF = [Lambda, RSch]
    DatLimit =[[Phimin, Phimax], [Phimin, np.pi/2], [Phimax, np.pi/2]]
    GeoDat = [[d1, b1, b1], [d2, b2, b1], [d3, b3, b1]]
    
    ### RECORDAR QUE SE ESTA RESCALADO CON b ####### 
    DatGeo = []
    for i in [0, 1, 2]: # 0 -> HORIZONTAL, 1 -> PENDIENTE POSITIVA, 2 -> PENDIENTE NEGATIVA
        #print(i)
        DatLim, geoDat = DatLimit[i], GeoDat[i]
        solFphi, solFu = integracion(i, paramF, geoDat, DatLim, Nptos=Nptos, Rtol=Rtol, Atol=Atol, fStep=fStep)
        DatGeo.append([solFphi, solFu])

    return DatGeo


def triang(paramF, Val_b, delta, Nptos=5000,
           Rtol=1e-15, Atol=1e-18, fStep=1e-20):
    """
    se construye el triángulo
    Vectores de entrada
    paramF -> [Lambda, RSch, Phimin, Phimax] cantidades en unidades físicas
              Lambda -> constante cosmológica
              RSch -> radio de Sch. del objeto bajo estudio
              Phimin, Phimax -> ángulos de la integración, recordar que Phimax = np.pi-Phimin

    Val_b -> [b1, b2, b3] parámetros de impacto de cada geodésica
              b1 -> geodésica horizontal
              b2=b3 -> geodésicas restantes del triángulo
              
    delta -> [d1, d2, d3 ]
             d1 la geodesica horizontal, esta la usaremos para obtener el u0 en los
             extremos Phimin, Phimax de la geodesica
    """

    DatGeo = Intind(paramF, Val_b, delta, Nptos=Nptos,
               Rtol=Rtol, Atol=Atol, fStep=fStep)
    
    phi1, u1 = DatGeo[0]
    phi2, u2 = DatGeo[1]
    phi3, u3 = DatGeo[2]

    xf = lambda theta, u: np.cos(theta)/u
    yf = lambda theta, u: np.sin(theta)/u

    c1 = [xf(phi1, u1), yf(phi1, u1)]
    c2 = [xf(phi2, u2), yf(phi2, u2)]
    c3 = [xf(phi3, u3), yf(phi3, u3)]
    
    return c1, c2, c3, [[phi1, u1 ], [phi2, u2], [phi3, u3]]


def AngulP(dat, paramF, b, ptos):
    """
    Cálculo de phi_p
    dat -> [phi, u, u']
            u -> solución de la geodesica C_p
            u' -> derivada de la solución u, respecto a phi, correspondiente a la geodesica C_p

    paramF -> [Lambda, RSch, Phimin, Phimax] cantidades en unidades físicas
              Lambda -> constante cosmológica
              RSch -> radio de Sch. del objeto bajo estudio
              Phimin, Phimax -> ángulos de la integración, recordar que Phimax = np.pi-Phimin

    b -> parametro de impacto correspondiente a la geodesica C_1

    ptos -> ptos a los que se desea saberl el ángulo
    """
    # Cantidades Adim
    phi, u, du = dat
    Lambda, RSch, _, _ = paramF
    Lb = Lambda*b**2
    rS = RSch/b

    # filtrando
    bNaN = np.isnan(du)
    ind = [not(i) for i in bNaN]

    u = u[ind]
    du = du[ind]
    phi = phi[ind]

    tan = -u*np.sqrt(1-Lb/(3*u**2)-rS*u)/du
    tanF = interp1d(phi, tan, kind='linear') # quadratic
    angP = np.arctan(tanF(ptos))
    return angP

def AngSchDSitter(paramF, Val_b, delta, Datptos):
    """
    Datptos -> [PhiMax, pi/2, PhiMin]
               PhiMin, PhiMax -> ángulos de integración

    paramF -> [Lambda, RSch, Phimin, Phimax] cantidades en unidades físicas
              Lambda -> constante cosmológica
              RSch -> radio de Sch. del objeto bajo estudio
              Phimin, Phimax -> ángulos de la integración, recordar que Phimax = np.pi-Phimin

    Val_b -> [b1, b2, b3] parámetros de impacto de cada geodésica
              b1 -> geodésica horizontal
              b2=b3 -> geodésicas restantes del triángulo

    delta -> [d1, d2, d3 ]
             d1 la geodesica horizontal, esta la usaremos para obtener el u0 en los
             extremos Phimin, Phimax de la geodesica

    Datptos -> [phi1, phi2, phi3] coordenadas angulares del triangulo
               phi1 -> phiMax = pi-phiMin
               phi2 -> pi/2
               phi3 -> phiMin
    """

    # Calculando geodesicas
    DatGeo = Intind(paramF, Val_b, delta)
    phi1, u1 = DatGeo[0]  # C1
    phi2, u2 = DatGeo[1]  # C2
    phi3, u3 = DatGeo[2]  # C3

    # Calculando las derivadas
    # puedo usar duf(dat, param), donde param = [Lb, rS] con Lb = Lambda*b**2, rS = RSch/b
    dc1 = np.gradient(u1, phi1, edge_order=2) # C1
    dc2 = np.gradient(u2, phi2, edge_order=2) # C2    
    dc3 = np.gradient(u3, phi3, edge_order=2) # C3
    
    Lb = paramF[0]*Val_b[1]**2
    rS = paramF[1]/Val_b[1]

    #der = np.sqrt(1+Lb/3-u2**2*(1-u2*rS))
    #fac = Val_b[1]/Val_b[0]
    #der = np.sqrt(1+Lb/3-(u2*fac)**2*(1-u2*fac*rS))*Val_b[0]/Val_b[1]
    #plt.plot(phi2, der, 'r--')
    #plt.plot(phi2, dc2, 'b-')
    #plt.show()


    # CALCULANDO ANGULOS de los extremos
    # C1
    ptosC1 = [Datptos[0], Datptos[2]]
    dat1 = [phi1, u1, dc1]
    b1 = Val_b[0]
    angC1 = AngulP(dat1, paramF, b1, ptosC1)

    # C2
    ptosC2 = [Datptos[2], Datptos[1]]
    dat2 = [phi2, u2, dc2]
    angC2 = AngulP(dat2, paramF, b1, ptosC2)

    # C3
    ptosC3 = [Datptos[1], Datptos[0]]
    dat3 = [phi3, u3, dc3]
    angC3 = AngulP(dat3, paramF, b1, ptosC3)

    beta1 = angC3[1]-angC1[0]
    beta2 = angC1[1]-angC2[0]
    beta3 = -angC3[0]+angC2[1]  
    # np.pi-angC3[0]+angC2[1] es correcto, pero como integramos al reves phiMax -> pi/2 la derivada numerica
    # sería en la otra dirección. Imaginar triangulo plano

    return [angC1, angC2, angC3], [beta1, beta2, beta3]

def duf(dat, param):
    """
    dat -> [phi, u] datos de entrada
           phi -> ángulo, variable dependiente 
           u -> solución

    param = [Lb, rS]
    """
    phi, u = dat
    Lb, rS =  param

    # dividiendo
    phi1 = phi[phi<np.pi/2]
    phi2 = phi[phi>=np.pi/2]
    u1 = u[phi<np.pi/2]
    u2 = u[phi>=np.pi/2]
    
    du1 = np.sqrt(1+Lb/3-u1**2*(1-u1*rS))
    du2 = -np.sqrt(1+Lb/3-u2**2*(1-u2*rS))
    
    return np.concatenate((phi1, phi2)), np.concatenate((du1, du2))

########
# No
#######

def full(paramF, geoDat, Nptos=500, Rtol=1e-09, Atol=1e-12, fStep=1e-08):
    """
    Resuelve la solución completa de la ecuación diferencial

    paramF -> [Lambda, RSch, Phimin, Phimax] cantidades físicas
              Lambda -> lambda barra : constante cosmológica
              RSch -> radio de Sch de la fuente estudiada
              PhiMin, PhiMax -> ángulos de integracióm

    geoDat -> [delta, b1, b2] datos de la geodesica
            delta -> delta
            b1 -> parámetro correspondiente a la geodesica que se calcula
            b2 -> parámetro de imapacto usado para la condición inicial 
                  y conicida con la segunda geodesica el origen

            Para solo construir la geodesica sin importar el triangulo tomar b1=b2

    lim -> [PhiMin, PhiMax]
        PhiMin, PhiMax -> ángulos de integracióm
    
    deltbVal -> delta que se utiliza para la serie

    Out: Perfil completo y Perfiles de ambas ramas
    """
    phi1, u1, phi2, u2 = integracion(paramF, geoDat,
                                     Nptos=Nptos, Rtol=Rtol,
                                     Atol=Atol, fStep=fStep)
    
    _, _, PhiMin, PhiMax = paramF
    if PhiMin>PhiMax:
        Phimin = PhiMax
        Phimax = PhiMin
        itD = interp1d(phi2, u2, kind='quadratic')
        itI = interp1d(phi1, u1, kind='quadratic')
    else:
        Phimin = PhiMin
        Phimax = PhiMax
        itD = interp1d(phi1, u1, kind='quadratic')
        itI = interp1d(phi2, u2, kind='quadratic')

    func = lambda theta: itD(theta)-itI(theta)
    print(func(Phimin))
    thetaC = fsolve(func, Phimin)
    print(thetaC)

    phiD = np.linspace(Phimin, thetaC, Nptos, endpoint=False)
    phiI = np.linspace(thetaC, Phimax, Nptos)
    uD = itD(phiD)
    uI = itI(phiI)
    
    phiFull = np.concatenate((phiD, phiI), axis=None)
    uFull = np.concatenate((uD, uI), axis=None)
    itFull = interp1d(phiFull, uFull, kind='quadratic')

    return itFull, phiFull, uFull , [phi1, u1, phi2, u2]

    ## checking
    #if PhiMin < PhiMax:
    #    sys.exit("No se cumple la condición PhiMin < PhiMax")