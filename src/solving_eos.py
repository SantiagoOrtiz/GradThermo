# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 11:27:55 2023
@author: Santi, Rama and Andreas
"""

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.integrate import quad
import os, sys

currentdir = os.path.dirname(os.path.realpath('__file__'))
sys.path.append(currentdir)


class EOS:   

    def __init__(self, molecule='NH3'):
        critical_path = os.path.join(currentdir, 'src', 'critical.csv')
        db_critical = pd.read_csv(critical_path).set_index('Parameter').to_dict()
        self.ω = db_critical[molecule]['w']
        self.Tc = db_critical[molecule]['Tc']
        self.Pc = db_critical[molecule]['Pc']
        stacmech_path = os.path.join(currentdir, 'src', 'stacmech.csv')
        db_stacmech = pd.read_csv(stacmech_path).set_index('Parameter').to_dict()        
        self.Do = db_stacmech[molecule]['Do']
        self.mw = db_stacmech[molecule]['mw']
        self.Ov = np.array([db_stacmech[molecule]['Ov'+str(n)] for n in range(1, 9+1) if db_stacmech[molecule]['Ov'+str(n)] !=0])
        self.Or = np.array([db_stacmech[molecule]['Or'+str(n)] for n in range(1, 3+1) if db_stacmech[molecule]['Or'+str(n)] !=0])
        self.sig = db_stacmech[molecule]['Sigma']
        self.Wo = db_stacmech[molecule]['Wo']
        antonine_path = os.path.join(currentdir, 'src', 'antonine.csv')
        db_antonine = pd.read_csv(antonine_path).set_index('molecule')
        self.antoineq = db_antonine.loc[molecule]


    def __Z(self, ʋ, P, T, R):
        return P*ʋ/(R*T)
        
    def __a(self, R):
        return 0.45724 * (np.power(R, 2)*np.power(self.Tc, 2))/self.Pc
        
    def __b(self, R):
        return 0.07780 * R*self.Tc/self.Pc
    
    def __fω(self):
        return 0.37464 + 1.5422*self.ω - 0.26992*np.power(self.ω, 2)
    
    def __α(self, T):
        Tr = T/self.Tc
        ωpol = self.__fω()
        return np.power(1 + (1 - np.sqrt(Tr))*ωpol, 2)
    
    def __ʋpol(self, ʋ, R):
        b = self.__b(R)
        return np.power(ʋ, 2) + 2*b*ʋ - np.power(b, 2)
        
    def __PengRobinson(self, ʋ, T, P, R):               
        a = self.__a(R)
        b = self.__b(R)
        ʋpol = self.__ʋpol(ʋ, R)
        f = (R*T)/(ʋ - b) - a*self.__α(T)/ʋpol - P
        return f

    def __dαdT(self, T):
        Tc = self.Tc
        Tr = T/Tc        
        ωpol = self.__fω()
        dαdT = 2*(1 + (1-np.sqrt(Tr))*ωpol)*(-ωpol/(2*np.sqrt(Tc*T)))
        return dαdT  
    
    def __dPdT(self, ʋ, T, R):    
        a = self.__a(R)
        b = self.__b(R)
        ʋpol = self.__ʋpol(ʋ, R)
        return R/(ʋ-b) - a/ʋpol*self.__dαdT(T)
    
    
    ################ Solver for Peng-Robinson Equation of State ################

    def antoine(self, T, P=None):
        T = np.array(T) if type(T) != np.ndarray else T
        if np.array(P).any():
            P = np.array([P]) if type(P) != np.ndarray else P
        else:
            P = 101.325E3*np.ones_like(T, float)
    
        def evalT(TP, antoineq=self.antoineq):
            T, P = TP
            if (T >= antoineq['T1'][0]) and (T < self.Tc) and (P < self.Pc):
                abcrow = antoineq[(antoineq['T1'] <= T) & (T < antoineq['T2'])]
                A = abcrow.iloc[0]['A']
                B = abcrow.iloc[0]['B']
                C = abcrow.iloc[0]['C']
                return np.power(10, A - B/(T + C))
            elif T < antoineq['T1'][0]:                
                print('The provided temperatures are out of bounds in the Antonine Equation.')
                raise Exception(f" If the Temperature < {antoineq['T1'][0]}, a phase has to be provided.")
            else:
                print(f'Some of the given T and P values lie outside (>) the range of critical conditions: Tc={self.Tc} & Pc={self.Pc}')
                return self.Pc/1E5        
        return np.array([*map(evalT, [*zip(T, P)])])*1E5

    def solve_eos(self, T_, P_, phase=None, R=8.3144598):
        T = np.ravel(np.array(T_))
        P = np.ravel(np.array(P_))
        ʋ_solution = np.zeros_like(T, float)
        
        def fsolvei(self, Ti, Pi, ʋ0, R):
            eos = lambda ʋ: self.__PengRobinson(ʋ, Ti, Pi, R)
            ʋ_solution = fsolve(eos, ʋ0)[0]
            return ʋ_solution
        
        if phase:
            if (phase.lower() == 'liquid') or (phase.lower() == 'l'):
                v0 = 1.2*self.__b(R)*np.ones_like(T, float)
            elif (phase.lower() == 'gas') or (phase.lower() == 'g'):
                v0 = (R*T/P)
            else:
                raise Exception("phase argument only allows 'liquid' or 'gas'")
        else:           
            Pv = self.antoine(T, P)
            v0 = (R*T/P)*(P <= Pv) + (1.2*self.__b(R))*(P > Pv)

        for i, (Ti, Pi, ʋ0) in enumerate(zip(T, P, v0)):
            ʋ_solution[i] = fsolvei(self, Ti, Pi, ʋ0, R)
        return np.reshape(ʋ_solution, np.shape(T_))


    ##################### Departure functions calculations ##################### 

    def ΔS_dep(self, T, P, R=8.3144598, phase=None):
        ʋ = self.solve_eos(T, P, phase=phase)
        fʋ = lambda ʋ: self.__dPdT(ʋ, T, R) - R/ʋ
        intʋ = quad(fʋ, np.Infinity, ʋ)
        ΔS = intʋ[0] + R*np.log(self.__Z(ʋ, P, T, R))
        return ΔS
    
    def ΔH_dep(self, T, P, R=8.3144598, phase=None):
        vmol1 = self.solve_eos(T,P, phase=phase)
        b = self.get__b()
        a = self.get__a()
        k  = self.get__α(T) - T*self.get__dαdT(T)
        root_2 = np.power(2,0.5)
        kk = k * a/(b*2*root_2) 
        int21 = (np.log((vmol1 + b*(1 - root_2))/(vmol1 + b*(1 + root_2))))
        return  kk*(int21) + P*vmol1 - R*T
    
    def ΔG_dep(self, T, P, R=8.3144598, phase=None):
        return self.G_real(T, P, phase=phase) -  self.G_ig(T, P)
    

    ############# Ideal gas calculations from statistical mechanics ############# 

    def __Avib(self, T):
        Ov = self.Ov
        return np.log(1-np.exp(-Ov/T))

    def __Cv_vib(self, T):
        Ov = self.Ov
        return (Ov/T)**2*np.exp(Ov/T)/(np.exp(Ov/T)-1)**2

    def __Svib(self, T):
        Ov = self.Ov
        return (Ov/T)/(np.exp(Ov/T)-1) - np.log(1-np.exp(-Ov/T))

    def __Uvib(self, T):
        Ov = self.Ov
        return (Ov/T)/(np.exp(Ov/T)-1)

    def ʋ_ig(self, T, P, R=8.3144598):
        return R*T/P

    def A_ig(self, T, P, h=6.62607E-34, k=1.38065E-23, Na=6.02214E+23, R=8.3144598):
        Do = self.Do; mw = self.mw; Or = self.Or; sig = self.sig; Wo = self.Wo
        vig = self.ʋ_ig(T, P, R)
        avib = self.__Avib(T)
        return -( R*T*np.log( ((2*np.pi*mw*k*T/(h**2))**(3/2))*(vig*np.exp(1)/Na) )
                  +  R*T*np.log( (1/sig)*((np.pi*(T**3)/(Or[0]*Or[1]*Or[2]))**(1/2)))
                  + Do*4184 - R*T*np.sum(avib) + np.log(Wo)*R*T )

    def Cvig_SM(self, T, R=8.3144598):
        return R*(3/2+3/2+np.sum(self.__Cv_vib(T)))

    def S_ig(self, T, P, h=6.62607E-34, k=1.38065E-23, Na=6.02214E+23, R=8.3144598):
        mw = self.mw; Or = self.Or; sig = self.sig; Wo = self.Wo
        vig = self.ʋ_ig(T, P, R)
        svib = self.__Svib(T)
        Sig = R*( np.log( (2*np.pi*mw*k*T/(h**2))**(3/2) * (vig*np.exp(5/2)/Na))
                 + np.log( ((1/sig)*(np.pi*(T**3)*np.exp(3)/(Or[0]*Or[1]*Or[2]))**0.5) ) 
                 + np.log(Wo) + np.sum(svib) )
        return Sig

    def U_ig(self, T, R=8.3144598):
        Do = self.Do
        uvib = self.__Uvib(T)
        return 3/2*R*T+3/2*R*T-Do*4184+R*T*np.sum(uvib)

    def H_ig(self, T, P, R=8.3144598):
        Hig = self.U_ig(T, R) + P*self.ʋ_ig(T, P, R)
        return Hig

    def G_ig(self, T, P, R=8.3144598):
        Gig = self.H_ig(T, P, R) - T*self.S_ig(T, P)
        return Gig

    def print_SMEq(self, T, P):
        Gig_check = (self.A_ig(T, P) + P*self.ʋ_ig(T, P))/self.G_ig(T, P)-1
        print('Difference in G, from U v. A calculations (potential error): ',Gig_check) #should equal zero
        print('')
        print('Ideal Gas Thermodynamic values for Chloromethane:')
        print('Uig:',round(self.U_ig(T)/1000,6),'kJ/mol')
        print('Aig:',round(self.A_ig(T, P)/1000,6),'kJ/mol')
        print('Hig:',round(self.H_ig(T, P)/1000,6),'kJ/mol')
        print('Sig:',round(self.S_ig(T, P),10),'J/(mol*K)')
        print('Gig:',round(self.G_ig(T, P)/1000,6),'kJ/mol')
        print('Cv:',round(self.Cvig_SM(T),8),'kJ/mol')        
    
    
    ######### ΔH, ΔS, ΔG, for IDEAL process from (P1, T1) to (P2, T2) #########

    def ΔH_ig(self, T, P, R = 8.3144598):
        T1,T2 = T
        P1,P2 = P       
        return self.H_ig(T2,P2) - self.H_ig(T1,P1)
        
    def ΔS_ig(self, T,P,  R = 8.3144598):
        T1,T2 = T
        P1,P2 = P        
        return  self.S_ig(T2, P2) - self.S_ig(T1,P1) 
    
    def ΔG_ig(self,T, P, R = 8.3144598):
        T1, T2 = T
        P1, P2 = P
        return self.G_ig(T2,P2) - self.G_ig(T1,P1)
    

    ########## ΔH, ΔS, ΔG, for REAL process from (P1, T1) to (P2, T2) ##########

    def H_real(self,T, P, R = 8.3144598, phase=None):
        return (self.ΔH_dep(T, P, phase=phase) + self.H_ig(T,P))

    def S_real(self, T, P, R = 8.3144598, phase=None):
        return self.ΔS_dep(T,P, phase=phase) + self.S_ig(T,P)
    
    def G_real(self,T, P, R = 8.3144598, phase=None):
        return self.H_real(T, P,phase=phase) - T*self.S_real(T, P,phase=phase)        
    
    
    def ΔH_real(self, T, P, R = 8.3144598, phase=None):
        T1,T2 = T
        P1,P2 = P
        phase1, phase2 = phase
        return self.H_real(T2, P2, phase=phase2) - self.H_real(T1, P1, phase=phase1) 
          
    def ΔS_real(self, T, P, R = 8.3144598, phase=None):
        T1,T2 = T
        P1,P2 = P
        phase1, phase2 = phase
        return -self.S_real(T1,P1,phase=phase1) + self.S_real(T2,P2,phase=phase2) 
    
    def ΔG_real(self,T, P, R = 8.3144598, phase=None):
        T1, T2 = T
        P1, P2 = P
        phase1, phase2 = phase
        return   self.G_real(T2, P2,phase=phase2) - self.G_real(T1, P1,phase=phase1)


    ################### Providing access to private methods ###################
    
    def get__PengRobinson(self):
        peng_robinson = lambda ʋ, T, P: self.__PengRobinson(ʋ, T, P, R=8.3144598)
        return peng_robinson
    
    def get__Avib(self, T):
        return self.__Avib(T)
    
    def get__Cv_vib(self, T):
        return self.__Cv_vib(T)

    def get__Svib(self, T):
        return self.__Svib(T)

    def get__Uvib(self, T):
        return self.__Uvib(T)
    
    def get__a(self, R=8.3144598):
        return self.__a(R)
    
    def get__b(self, R=8.3144598):
        return self.__b(R)
    
    def get__α(self, T, R=8.3144598):
        return self.__α(T)
    
    def get__dαdT(self,T, R=8.3144598):
        return self.__dαdT(T)
