#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 11:50:32 2025

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
    
    print(df)
    
    df['uRaw'] = df['Raw'].std()
    df['uCor'] = df['Cor'].std()
    
    return df


def calculate_tau(df):
    tau = 0
    
    df.plot(x='Raw', y='Cor', grid=True, kind='scatter')
    plt.show()
    meta = {
        'title' : 'Intial Plot of Coincidence Values',
        'xlabel' : 'Raw Coincidence',
        'ylabel' : 'Expected Statistical Correction',
        'data-label' : 'Data',
        'fit-label' : 'Linear Fit',
        'loc' : 'lower right',
        'save-name' : 'coincwithres'
        }
    
    data = tk.quick_analyze(df['Raw'], df['Cor'], fit_type='linear',
                            show=True, yerr=df['uCor'], meta=meta,
                            params = ['1/tau'], chi=True, res=True,
                            path='../Images/03.14/', save=True)
    tau = 1/data['popt'][0]
    utau = np.abs(data['pstd'][0]*(tau**2))
    return tau, utau
    

if __name__ == '__main__':
    df = load_data('../Data/03.11/Raw_Coincidence.txt')

    tau, utau = calculate_tau(df)
    print(f'{tau} +/- {utau}')
