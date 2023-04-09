# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 11:27:55 2023
@author: santi
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
        dbpath = os.path.join(currentdir, 'src', 'database_new.csv')
        db_dict = pd.read_csv(dbpath).set_index('Parameter').to_dict()
        self.ω = db_dict[molecule]['w']
        self.Tc = db_dict[molecule]['Tc']
        self.Pc = db_dict[molecule]['Pc']
        self.Do = db_dict[molecule]['Do']
        self.mw = db_dict[molecule]['mw']
        self.Ov = np.array([db_dict[molecule]['Ov'+str(n)] for n in range(1, 9+1) if db_dict[molecule]['Ov'+str(n)] !=0])
        self.Or = np.array([db_dict[molecule]['Or'+str(n)] for n in range(1, 3+1) if db_dict[molecule]['Or'+str(n)] !=0])
        self.sig = db_dict[molecule]['Sigma']
        self.Wo = db_dict[molecule]['Wo']
        
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
        
    def solve_eos(self, T_, P_, ʋ0=1.0E-2, R=8.314):
        
        if (type(T_) != list) or (type(P_) != list):
            T = np.ravel(np.array([T_]))
            P = np.ravel(np.array([P_]))
        else:
            T = np.ravel(T_)
            P = np.ravel(P_)
        
        ʋ_solution = np.zeros_like(T, float)
        
        def fsolvei(self, Ti, Pi, ʋ0, R):
            eos = lambda ʋ: self.__PengRobinson(ʋ, Ti, Pi, R)
            ʋ_solution = fsolve(eos, ʋ0)[0]
            return ʋ_solution
        
        for i, (Ti, Pi) in enumerate(zip(T, P)):
            ʋ_solution[i] = ʋ0 = fsolvei(self, Ti, Pi, ʋ0, R)
            
        return np.reshape(ʋ_solution, np.shape(T_))


    def ΔS_dep(self, T, P, R=8.314):
        ʋ = self.solve_eos(T, P)
        fʋ = lambda ʋ: self.__dPdT(ʋ, T, R) - R/ʋ
        intʋ = quad(fʋ, np.Infinity, ʋ)
        ΔS = intʋ[0] + R*np.log(self.__Z(ʋ, P, T, R))
        return ΔS
    
    
    def ΔH_dep(self, T, P, R=8.314):
        ʋ = self.solve_eos(T, P)
        fʋ = lambda ʋ: T*self.__dPdT(ʋ, T, R) - self.__PengRobinson(ʋ ,T, P, R)
        intʋ = quad(fʋ, np.Infinity, ʋ)
        ΔH = intʋ[0] + self.__PengRobinson(ʋ, T, P, R)*ʋ - R*T
        return ΔH
    
    def ΔG_dep(self, T, P, R=8.314):
        return self.ΔH_dep(T, P) - T * self.ΔS_dep(T, P)
    
    
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
<<<<<<< HEAD
        vig = self.__ʋ_ig(T, P, R)
=======
        vig = self.ʋ_ig(T, P, R)
>>>>>>> 501c1e5df0a860d61d0f2b0484d5fc807e9888eb
        svib = self.__Svib(T)
        Sig = R*( np.log( (2*np.pi*mw*k*T/(h**2))**(3/2) * (vig*np.exp(5/2)/Na))
                 + np.log( ((1/sig)*(np.pi*(T**3)*np.exp(3)/(Or[0]*Or[1]*Or[2]))**0.5) ) 
                 + np.log(Wo) + np.sum(svib) )
        return Sig

    def U_ig(self, T, R=8.3144598):
        Do = self.Do
        uvib = self.__Uvib(T)
        return 3/2*R*T+3/2*R*T-Do*4184+R*T*np.sum(uvib)

<<<<<<< HEAD
    def __ʋ_ig(self, T, P, R=8.3144598):
        return R*T/P

=======
>>>>>>> 501c1e5df0a860d61d0f2b0484d5fc807e9888eb
    def H_ig(self, T, P, R=8.3144598):
        Hig = self.U_ig(T, R) + P*self.__ʋ_ig(T, P, R)
        return Hig

    def G_ig(self, T, P, R=8.3144598):
        Gig = self.H_ig(P, R) - T*self.S_ig(T, P)
        return Gig

    def check_SMEq(self, T, P):
        Gig_check = (self.A_ig(T, P) + P*self.ʋ_ig(T, P))/self.G_ig(T, P)-1
        print('Difference in G, from U v. A calculations (potential error): ',Gig_check) #should equal zero
        print('')
        print('Ideal Gas Thermodynamic values for Chloromethane:')
        print('Hig:',round(self.H_ig(T, P)/1000,2),'kJ/mol')
        print('Sig:',round(self.S_ig(T, P),2),'J/(mol*K)')
        print('Gig:',round(self.G_ig(T, P)/1000,2),'kJ/mol')
        
    #Hig_SM, Gig_SM, Sig_SM, Cvig_SM
    
    def ΔH_ig(self, T, R = 8.314):
        T1,T2 = T
        fcp = lambda T: self.Cvig_SM(T) + R
        intcp = quad(fcp, T1, T2)
        
        return intcp[0]
        
    def ΔS_ig(self, T,P,  R = 8.314):
        T1,T2 = T
        P1,P2 = P
        fcp = lambda T: (self.Cvig_SM(T) + R)/T
        intcp = quad(fcp, T1, T2)
        return intcp[0] - R*np.log(P2/P1)
    
    def ΔG_ig(self,T, P, R = 8.314):
        return self.ΔH_ig(T) - T*self.ΔS_ig(T,P)
    
    def ΔH_real(self, T, P, R = 8.314):
        T1,T2 = T
        P1,P2 = P
        return -self.ΔH_dep(T1,P1) + self.ΔH_ig(T) + self.ΔH_dep(T2,P2)
          
    def ΔS_real(self, T, P, R = 8.314):
        T1,T2 = T
        P1,P2 = P
        return -self.ΔS_dep(T1,P1) + self.ΔS_ig(T,P) + self.ΔS_dep(T2,P2) 
    
    def ΔG_real(self,T, P, R = 8.314):
        T1, T2 = T
        P1, P2 = P
        
        delHT1 = self.H_ig(T1, P1) + self.ΔH_dep(T1, P1)
        delST1 = self.S_ig(T1,P1) + self.ΔS_dep(T1, P1)
        delGT1 = delHT1 - T1*delST1
        
        delHT2 = self.H_ig(T2, P2) + self.ΔH_dep(T2, P2)
        delST2 = self.S_ig(T2,P2) + self.ΔS_dep(T2, P2)
        delGT2 = delHT2 - T2*delST2
                
        return   delGT2 - delGT1
        
    def get__PengRobinson(self):
        peng_robinson = lambda ʋ, T, P: self.__PengRobinson(ʋ, T, P, R=8.314)
        return peng_robinson