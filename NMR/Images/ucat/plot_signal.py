#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 04 13:28:27 2025

@author: Aditya K. Rao
@github: @adirao-projects
"""

import pandas as pd

def load_data(name):
    df = pd.read_csv(f'{name}.csv', names=['Time', 'ENV', 'Q', 'I'], skiprows=2)
    return df

def get_uncert(df):
    df_ = df.where(df['Time']<0)
    df['uENV'] = df_['ENV'].max()-df_['ENV'].min()
    df['uQ'] = df_['Q'].var()
    df['uI'] = df_['I'].var()
    
    return df


def plot(df, name):
    sig = df.plot(x='Time', y='ENV', legend=False, grid='on', 
                title='Raw Pulse Data', color=['black'],
                xlabel='Time (s)', ylabel='Observed Voltage (V)', 
                figsize=(14, 8)).get_figure()

    
    dif = df.plot(x='Time', y='I', legend=False, grid='on', 
                title='Raw Pulse Data', color=['black'],
                xlabel='Time (s)', ylabel='Difference Voltage (V)', 
                figsize=(14, 8)).get_figure()


    sig.savefig(f'{name}sig')
    dif.savefig(f'{name}dif')

if __name__ == '__main__':
    # LMO : 56, 57, 58
    # HMO : 60, 61, 62
    # H20 : 64, 65, 66
    scope_files = ['56', '57', '58', '60', '61', '62', '64', '65', '67', '68']
    #scope_files = ['46', '47', '48', '49']
    for scp in scope_files:
        df = load_data(f"scope_{scp}")
        plot(df, scp)
        