#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 11:17:48 2025

@author: Aditya K. Rao
@github: @adirao-projects
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import toolkit as tk

from c_const_calc import calc_c 


def load_data(file_path, date='00.00', rawpath=False):
    if rawpath:
        df = pd.read_csv(file_path, delimiter='\t', 
                         names=['Raw', 'Cor'], skiprows=1)
    else:
        df = pd.read_csv(fr'../Data/{date}/{file_path}', delimiter='\t', 
                     names=['Raw', 'Cor'], skiprows=1)
    
    df['uRaw'] = df['Raw'].std()
    df['uCor'] = df['Cor'].std()
    
    df['Adj'] = df['Raw'] - df['Cor']
    
    return df


def calc_Nval(Ndata):
    n = Ndata['Adj'].mean()
    u = Ndata['Adj'].std()
    
    return n, u


def calc_Eab(nvals, c):
    num = 0
    den = 0
    
    num = nvals['A90B90'] + nvals['A00B00']
    den = num + nvals['A90B00'] + nvals['A00B90']
    
    num += -nvals['A90B00'] -nvals['A00B90']
    den -= 4*c
    
    return num/den


if __name__ == '__main__':
    df_A00_B00 = load_data('../Data/03.18/N_B_0_A_0.txt',rawpath=True)
    df_A90_B00 = load_data('../Data/03.18/N_B_0_A_90.txt',rawpath=True)
    df_A00_B90 = load_data('../Data/03.18/N_B_90_A_0.txt',rawpath=True)
    df_A90_B90 = load_data('../Data/03.18/N_B_90_A_90.txt',rawpath=True)
    df_raw = load_data('../Data/03.18/Raw_Coincidence.txt',rawpath=True)
    
    plt.figure(figsize=(20,10))
    df_A90_B00['Adj'].plot(label=r'$N(90^\circ, 0^\circ)$')
    df_A00_B90['Adj'].plot(label=r'$N(0^\circ, 90^\circ)$')
    plt.grid('on')
    plt.axhline(0, linestyle='--', label='$C=0$')
    plt.ylabel('Adjusted Coincidence Value')
    plt.xlabel('Index of Data')
    plt.legend(loc='upper right')
    plt.savefig('../Images/03.18/CcalcRes')
    plt.show()
    
    N_A00_B00 = df_A00_B00['Adj'].mean()
    N_A90_B00 = df_A90_B00['Adj'].mean()
    N_A00_B90 = df_A00_B90['Adj'].mean()
    N_A90_B90 = df_A90_B90['Adj'].mean()
    
    uN_A00_B00 = df_A00_B00['Adj'].std()
    uN_A90_B00 = df_A90_B00['Adj'].std()
    uN_A00_B90 = df_A00_B90['Adj'].std()
    uN_A90_B90 = df_A90_B90['Adj'].std()
    
    C = (1/2)*(N_A90_B00 + N_A00_B90)
    uC = (1/2)*np.abs(uN_A90_B00 + uN_A00_B90)
    
    # E Values for alpha beta etc.
    evalues = ['EA0B22p5', # E(a, b')
               'EAm45B22p5', # E(a', b')
               'EA0Bm22p5', # E(a, b)
               'EAm45Bm22p5' # E(a',b)
               ]
    
    S = 0
    
    for v in evalues:
        angA = re.findall(r'A(m?\d+p?\d?)', v)[0]
        angB = re.findall(r'B(m?\d+p?\d?)', v)[0]
        
        angA = angA.replace('m','-')
        angA = float(angA.replace('p','.'))
        angB = angB.replace('m','-')
        angB = float(angB.replace('p','.'))

        angA00 = str(int(angA)).replace('-','m')
        angB00 = str(angB).replace('-','m').replace('.','p')
        angA90 = str(int(angA+90)).replace('-','m')
        angB90 = str(angB+90).replace('-','m').replace('.','p')
        
        #print(angA00, angB00, angA90, angB90)
        N9090 = calc_Nval(load_data(fr'NA{angA90}B{angB90}_{v}.txt', 
                                    date='03.27'))
        N0000 = calc_Nval(load_data(fr'NA{angA00}B{angB00}_{v}.txt',
                                    date='03.27'))
        N9000 = calc_Nval(load_data(fr'NA{angA90}B{angB00}_{v}.txt', 
                                    date='03.27'))
        N0090 = calc_Nval(load_data(fr'NA{angA00}B{angB90}_{v}.txt', 
                                    date='03.27'))
        
        nvals = {'A00B00':N0000[0],
                 'A90B90':N9090[0],
                 'A90B00':N9000[0],
                 'A00B90':N0090[0]}
        
        
        if v == 'EAm45B22p5':
            S -= calc_Eab(nvals, C)
        else:
            S += calc_Eab(nvals, C)
        
    print(S)
        
        
        
    
