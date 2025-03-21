#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 12:14:27 2025

@author: Aditya K. Rao
@github: @adirao-projects
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import toolkit as tk

def load_data(file_path):
    df = pd.read_csv(file_path, delimiter='\t', names=['Raw', 'Cor'],
                     skiprows=1)
    
    df['uRaw'] = df['Raw'].std()
    df['uCor'] = df['Cor'].std()
    
    df['Adj'] = df['Raw'] - df['Cor']
    
    return df


def qwp_angle(N_A00_B00, N_A90_B90, C):
    
    arg = (N_A90_B90 - C)/(N_A00_B00 - C)
    
    theta_l = np.arctan(np.sqrt(arg))
    
    return theta_l

def hwp_angle(qwp, N_A45_B45, N_A00_B00, N_A90_AB0, C):
    arg = (4*(N_A45_B45))/(N_A00_B00 + N_A90_B90 - 2*C)
    arg = (1/(np.sin(2*qwp)))*(arg-1)
    
    phi = np.arccos(arg)
    
    return phi


if __name__ == '__main__':
    df_A00_B00 = load_data('../Data/03.18/N_B_0_A_0.txt')
    df_A90_B00 = load_data('../Data/03.18/N_B_0_A_90.txt')
    df_A00_B90 = load_data('../Data/03.18/N_B_90_A_0.txt')
    df_A90_B90 = load_data('../Data/03.18/N_B_90_A_90.txt')
    df_raw = load_data('../Data/03.18/Raw_Coincidence.txt')
    
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
    
    print(f'{N_A90_B00} +/- {uN_A90_B00}')
    print(f'{N_A00_B90} +/- {uN_A00_B90}')
    print(f'{C} +/- {uC}')
    
    qwp = qwp_angle(N_A00_B00, N_A90_B90, C)
    uqwp = 0
    print(f'{qwp} +/- {uqwp}')
    
    #df_A45_B45 = load_data('../Data/03.18/N_B_45_A_45.txt')
    
    