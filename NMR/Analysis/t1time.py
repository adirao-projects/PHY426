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
        

def get_uncert(df):
    df_ = df.where(df['Time']<0)
    df['uENV'] = df_['ENV'].max()-df_['ENV'].min()
    df['uQ'] = df_['Q'].var()
    df['uI'] = df_['I'].var()
    
    return df

def get_t1(df, units, name):
    df = df[df['Time']<0.001]
    df = df[df['Time']>=2e-4]
    vpulse = np.vectorize(pulse)

    # data = tk.curve_fit_data(df['Time'], df['ENV'], uncertainty=df['uENV'],
    #                          fit_type='custom', chi=False, res=True,
    #                          model_function_custom=vpulse,
    #                          guess=(0.05, 0.00005, 0.0002, 1.75, -0.0001))
    
    data = tk.curve_fit_data(df['Time'], df['ENV'], uncertainty=df['uENV'],
                            fit_type='exp', chi=True, res=True,
                            guess=(1.75, -100, 0.00025))

    print(data['popt'])
    meta = {'title':f'Fit to tail of {name} Pulse',
            'xlabel':'Time (s)',
            'ylabel':'ENV (V)',
            'fit-label': r"$A\exp{\tau(x-x_0)}$",
            'data-label': "Pulse Data",
            'loc':'upper right',
            'save-name':f'{name}.png'}
    
    # tk.block_print(data=[f"Chi:{data['chisq']}\pm", f"A:{data['popt'][0]}",
    #                      f"B:{data['popt'][1]}", f"x_0:{data['popt'][2]}"], 
    #                title='Fit Parameters')
    tk.parameter_print(['A', r'\tau', 'x_0'], data['popt'], data['pstd'])
    
    print(r'\chi^2_{\text{red}}='+str(data['chisq']))
    print(f"T_1 = {np.abs(1/data['popt'][1])}")
    tk.quick_plot_residuals(df['Time'], df['ENV'], 
                            data['plotx'], data['ploty'],
                            uncertainty=df['uENV'], 
                            residuals=data['residuals'],
                            meta=meta)

def get_t1new(df, units, name):
    df = df[df['Time']<0.001]
    df = df[df['Time']>=2e-4]
    vpulse = np.vectorize(pulse)

    meta = {'title':f'Fit to tail of {name} Pulse',
            'xlabel':'Time (s)',
            'ylabel':'ENV (V)',
            'fit-label': r"$A\exp{\tau(x-x_0)}$",
            'data-label': "Pulse Data",
            'loc':'upper right',
            'save-name':f'{name}.png'}

    tkn.quick_analyze(df['Time'], df['ENV'], xerr=df['uENV'], fit_type='exp',
                      res=True, chi=True, guess=(1.75, -100, 0.00025),
                      meta = meta)

def get_fft(df):
    print('here')
    
    print(df['I'].head())
    
    i_dat = df['I'].to_numpy()
    
    
    
    #df.plot(x='Time', y='Q')
    
    i_fft = np.fft.rfft(i_dat)
    res = df['Time'].iloc[1] - df['Time'].iloc[0]
    #print(q_fft)
    #print(res)
    
    
    print(i_dat.shape[0])
    
    freq = np.fft.rfftfreq(i_dat.shape[0], res)
    
    #freq = np.arange(0, i_dat.shape[0], res)
    
    plt.plot(freq, np.abs(i_fft), color='black')
    
    i_dat_new = i_dat - np.mean(i_dat)
    
    i_fft_new = np.fft.rfft(i_dat_new)
    
    i_fft_new_abs = np.abs(i_fft_new)
    
    plt.plot(freq, i_fft_new_abs, color='black', linestyle='dashed')
    
    #imax = np.max(i_fft_new_abs)
    
    fmax = np.argmax(i_fft_new_abs)
    
    
    print(freq[fmax])
    
    print(20.9e6 + freq[fmax])
    print(20.9e6 - freq[fmax])
    
    
    plt.show()

if __name__ == '__main__':
    df_hmo1, units_hmo1 = load_data('../Data/heavyminoil.csv')
    #df_hmo1.plot(x='Time', y='ENV', legend=False, grid='on', 
    #             title='Raw Pulse Data for Heavy Mineral Oil',
    #             xlabel='Time (s)', ylabel='ENV (V)', figsize=(14, 8))
    df_hmo1 = get_uncert(df_hmo1)
    
    #get_fft(df_hmo1)
    
    #print(df_hmo1.head())
    
    #t1_hmo1new = get_t1new(df_hmo1, units_hmo1, 'Heavy Mineral Oil')
    t1_hmo1 = get_t1(df_hmo1, units_hmo1, 'Heavy Mineral Oil')
    
    #df_lmo1, units_lmo1 = load_data('../Data/lightminoil.csv')
    #df_lmo1.plot(x='Time', y='ENV')
