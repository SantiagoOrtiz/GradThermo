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
        dbpath = os.path.join(currentdir, 'src', 'database.csv')
        db_dict = pd.read_csv(dbpath).set_index('Parameter').to_dict()
        self.ω = db_dict[molecule]['ω']
        self.Tc = db_dict[molecule]['Tc']
        self.Pc = db_dict[molecule]['Pc']
        
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
        
    def __PengRobinson(self, ʋ, T, P, R=8.314):               
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


    def ΔS_dep(self, P, T, R):
        
        ʋ = self.solve_eos(T, P)
        fʋ = lambda ʋ: self.__dPdT(ʋ, T, R) - R/ʋ
        intʋ = quad(fʋ, 0.0, ʋ)  # np.Infinity
        ΔS = intʋ + R*np.log(self.__Z(ʋ, P, T, R))
        return ΔS
    
    
    def get__PengRobinson(self):
        peng_robinson = lambda ʋ, T, P: self.__PengRobinson(ʋ, T, P)
        return peng_robinson
    
