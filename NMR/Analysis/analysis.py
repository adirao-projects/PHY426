#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 10:25:15 2025

@author: Aditya K. Rao
@github: @adirao-projects
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import toolkit as tk
import toolkitnew as tkn

def load_data(csvname):
    # ch1 is ENV out
    # ch2 is Q out
    # ch3 is I out
    
    df = pd.read_csv(csvname, names=['Time', 'ENV', 'Q', 'I'], skiprows=2)
    df_units = {'Time':'s', 'ENV':'V', 'Q':'V', 'I':'V'}
    return df, df_units

def pulse(x, y0, x0, w0, h, tau):
    if x<=x0:
        return 0
    
    elif x<=w0:
        return h
    
    else:
        return h*np.exp((x-w0)*tau)+y0
        

def fiteqn(t, M0, T1):
    return M0*(1-np.exp(-t/T1))

def get_uncert(df):
    df_ = df.where(df['Time']<0)
    df['uENV'] = df_['ENV'].max()-df_['ENV'].min()
    df['uQ'] = df_['Q'].var()
    df['uI'] = df_['I'].var()
    
    return df


def fidcalc(df, units, name):
    df = df[df['Time']<0.001]
    df = df[df['Time']>=2e-4]

    meta = {'title':f'Fit to tail of {name} Pulse',
            'xlabel':'Time (s)',
            'ylabel':'ENV (V)',
            'fit-label': r"$A\exp{\tau(x-x_0)}$",
            'data-label': "Pulse Data",
            'loc':'upper right',
            'save-name':f'{name}fid'}

    tkn.quick_analyze(df['Time'], df['ENV'], yerr=df['uENV'], fit_type='exp',
                      res=True, chi=True, guess=(1.75, -100, 0.00025),
                      meta = meta, show=True, save=True, dataout=True,
                      params=('A', r'\tau', 'x_0'), rounding=False)

def t1calc(df, units, name):
    df = df['Time']<0.001
    df = df[df['Time']>=2e-4]
    vpulse = np.vectorize(pulse)

    meta = {'title':f'Fit to tail of {name} Pulse',
            'xlabel':'Time (s)',
            'ylabel':'ENV Voltage (V)',
            'fit-label': r"$M_0(1 - \exp{-\frac{t}{T_1}})$",
            'data-label': "Pulse Data",
            'loc':'upper right',
            'save-name':f'{name}.png'}

    tkn.quick_analyze(df['Time'], df['ENV'], xerr=df['uENV'], fit_type='custom',
                      model_function_custom = fiteqn, res=True, chi=True,
                      meta = meta, show=True, save=True, dataout=True,
                      params=('A', r'\tau', 'x_0'))
    
    
def t2calc(df, units, name):
    df = df[df['Time']<0.001]
    df = df[df['Time']>=2e-4]
    vpulse = np.vectorize(pulse)

    meta = {'title':f'Fit to tail of {name} Pulse',
            'xlabel':'Time (s)',
            'ylabel':'ENV (V)',
            'fit-label': r"$A\exp{\tau(x-x_0)}$",
            'data-label': "Pulse Data",
            'loc':'upper right',
            'save-name':f'{name}'}

    tkn.quick_analyze(df['Time'], df['ENV'], xerr=df['uENV'], fit_type='exp',
                      res=True, chi=True, guess=(1.75, -100, 0.00025),
                      meta = meta, show=True)


if __name__ == '__main__':
    
    
    hmo = ['../Data/hmo1.csv', '../Data/hmo2.csv', '../Data/hmo3.csv']
    lmo = ['../Data/lmo1.csv', '../Data/lmo2.csv', '../Data/lmo3.csv']
    h2o = ['../Data/h2o1.csv', '../Data/h2o2.csv', '../Data/h2o3.csv']
    
    for sample_nam, sample_dat in zip(('hmo', 'lmo', 'h2o'), (hmo, lmo, h2o)):
        for data_type, data_path in zip(('pi2', 'pi', 'pipi2'), sample_dat):
            df, units = load_data(data_path)
            df = get_uncert(df)
            df = df.dropna()
            
            if data_type == 'pipi2':
                pass
                #t1calc(df, units, sample_nam)
                
            elif data_type == 'pi2':
                fidcalc(df, units, sample_nam)

