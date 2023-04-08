# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 11:27:55 2023
@author: santi
"""

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
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
    
    def __α(self, T, ω, Tc):
            Tr = T/Tc
            ωpol = 0.37464 + 1.5422*ω - 0.26992*np.power(ω, 2)
            alpha = np.power(1 + (1 - np.sqrt(Tr))*ωpol, 2)
            return alpha
        
    def __PengRobinson(self, ʋ, T, P, R=8.314):
        ω=self.ω; Tc=self.Tc; Pc=self.Pc
                
        a = 0.45724 * (np.power(R, 2)*np.power(Tc, 2))/Pc
        b = 0.07780 * R*Tc/Pc
        ʋpol = np.power(ʋ, 2) + 2*b*ʋ - np.power(b, 2)
        f = (R*T)/(ʋ - b) - a*self.__α(T, ω, Tc)/ʋpol - P
        return f

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
    
    def get__PengRobinson(self):
        peng_robinson = lambda ʋ, T, P: self.__PengRobinson(ʋ, T, P)
        return peng_robinson
    
