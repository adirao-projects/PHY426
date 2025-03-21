import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from uncertainties import ufloat
import toolkit as tk


nm2m = 1e-9
m2nm = 1e9

# Constants
LAM0_n = 284.6*nm2m # meters
LAM0_u = 0.4*nm2m # meters

LAM0 = (LAM0_n, LAM0_u)


def load_data(path):
    df = pd.read_csv(path).dropna()
    
    return df


def equation(lam, m, b):
    return m/(lam + LAM0[0]) + b


def analyze(df_scope, df_wvlen, nm):
    # Average out scope values
    
    df = pd.DataFrame()
    df['reading'] = df_scope.groupby('color')['reading'].mean()
    df['uncert'] = df_scope.groupby('color')['reading'].std()
    
    df['wavelength'] = df_wvlen.set_index('color')['wavelength']
    
    df.dropna(inplace=True)
    
    df.plot(x='wavelength', y='reading', kind='scatter')
    
    
    plt.show()
    
    print(df, nm)
    plot(df, nm)


def plot(df, nm):
    meta = {'title':f'Spectrascope Data {nm}',
            'xlabel':'Wavelength',
            'ylabel':'Readings',
            'data-label':'Measurements',
            'fit-label':r'$\frac{m}{\lambda+\lambda_0} + b$',
            'loc':'upper right',
            'save-name':'spec-ideal'}
    
    data = tk.curve_fit_data(df['wavelength'], df['reading'], fit_type='custom', 
                            model_function_custom=equation, chi=True, res=True,
                            uncertainty=df['uncert'], guess=(5e3, 3e-1))
    print(data['popt'], data['pstd'])
    tk.quick_plot_residuals(df['wavelength'], df['reading'], 
                            data['graph-horz'], data['graph-vert'], 
                            data['residuals'], uncertainty=df['uncert'],
                            meta=meta)
    
    print(data['chi-sq'])

    plt.show()



if __name__ == '__main__':
    df_H_scope = load_data('../Data/ExpB-H-2025.02.03-1.csv')
    df_H_wvlen = load_data('../Data/ExpB-H-2025.02.03-2.csv')
    
    analyze(df_H_scope, df_H_wvlen, 'Hydrogen')
    
    df_He_scope = load_data('../Data/ExpB-He-2025.02.03-1.csv')
    df_He_wvlen = load_data('../Data/ExpB-He-2025.02.03-2.csv')
    
    df_He_scope['color'].replace({'red2':'red'}, inplace=True)
    
    analyze(df_He_scope, df_He_wvlen, 'Helium')
