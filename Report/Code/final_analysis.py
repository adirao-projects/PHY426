import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import toolkit as tk

# Constants
# Note that we are measuring a 3x3 grid which is where the multiplication
# by 3 comes from
GRID_SIZE = 1 # mm
GRID_SIZE = GRID_SIZE*3

CALIPER_UNCERT = 0.01

# Measurement Uncertainty remains the same
GRID_SIZE_u = 0.5 #mm uncert

OFFSETS = {
    "LENS":59.95,
    'LENS-T':25.42,
    "IMG":59.82,
    'LAMP':90.74,
    'APP':59.82
    }

CODENAME = {1 : 'Biconvex',
            2 : 'Thick-Biconvex',
            3 : 'Convex-Concave',
            4 : 'Concave-Convex',
            5 : 'Plano-Convex'
        }

def thin_lens_eqn(p, f):
    return 1/(1/f - 1/p)

def thin_lens_eqn_inv(pinv, finv):
    return finv - pinv

def thick_lens_eqn(p, f, fp, bp):
    return (1/f - 1/(p-fp))**(-1)+bp
    
def lens_maker(n, r1, r2, d):
    return (n-1)((1/r1) - (1/r2)+ ((n-1)*d)/(n*r1*r2))

def fit_mag_eqn(pq, a):
    return a*pq

def fit_thin_mag_eqn(p,q, a):
    return a*(q/p)

def fit_thick_mag_eqn(p, q, fp, bp, a):
    # p and q are measured from lens surfaces
    # Convert to principal plane distances
    p_prime = p - fp
    q_prime = q - bp
    return -a*(q_prime / p_prime)

def load_data(path, code):
    df = pd.read_csv(path)
    
    df['g'] = df['gridsize'].div(3)
    
    df['p'] = np.abs(df['lens'] + 0.5*OFFSETS['LENS']*0.1 \
                     - df['source']- + 0.5*OFFSETS['APP']*0.1)
    df['q'] = np.abs(df['lens'] + 0.5*OFFSETS['LENS']*0.1 \
                     - df['image'] - 0.5*OFFSETS['IMG']*0.1)
    pvals = []
    pvals_u = []
    qvals = []
    qvals_u = []
    gvals = []
    gvals_u = []
    for i in range(1, 5):
        lenscode = float(f'{code}.{i}')
        df_ = df[df['lens-code']==lenscode]
        pvals.append(df_['p'].mean())
        pvals_u.append(df_['p'].std())
        qvals.append(df_['q'].mean())
        qvals_u.append(df_['q'].std())
        gvals.append(-df_['gridsize'].mean())
        gvals_u.append(df_['gridsize'].std())
        
    data = {'p':np.array(pvals),
            'q':np.array(qvals),
            'g':np.array(gvals),
            'pu':np.array(pvals_u),
            'qu':np.array(qvals_u),
            'gu':np.array(gvals_u)}
        
    return df, data

"""
def analyze(df, grid_size):    

    df['Uncert'] = pd.Series([CALIPER_UNCERT for _ in range(len(df))])
    
    print(df)    

    
    img = [ufloat(i,CALIPER_UNCERT) for i in df['image'].to_numpy()]
    lens = [ufloat(i,CALIPER_UNCERT) for i in df['lens'].to_numpy()]
    src = [ufloat(i,CALIPER_UNCERT) for i in df['source'].to_numpy()]
    
    img = df['image'].to_numpy()
    lens = df['lens'].to_numpy()
    src =  df['source'].to_numpy()
    
    # Adding offsets
    img = img + 0.5*OFFSETS['IMG']*0.1
    lens = lens + 0.5*OFFSETS['LENS']*0.1 + OFFSETS['LENS-T']*0.1
    src = src - 0.5*OFFSETS['APP']
    
    #print(img, obj, lens)
    
    #mag_uncert = ufloat(df_mean['gridsize'], df_std['gridsize'])
    #gridsize_uncert = ufloat(GRID_SIZE, GRID_SIZE_u)
    
    p = np.abs(lens - src)
    q = np.abs(lens - img)
    
    #print(p,q)
    
    m = df['gridsize'].div(3)
    
    magnification = -(q/p)
    
    #magnification = mag_uncert/gridsize_uncert
   
    output =(p.tolist(),
             q.tolist(),
             m.tolist())
    
    return output
"""    


def fit_thin_lens(pvals, qvals, qvals_u, lcode):    
    meta = {'title':fr'Linear Fit to Thin Lens for {CODENAME[lcode]}',
            'xlabel':r'$\frac{1}{p}$ values ($\text{cm}^{-1}$)',
            'ylabel':r'$\frac{1}{q}$ values ($\text{cm}^{-1}$)',
            'data-label':'Measurements',
            'fit-label':r'$\frac{1}{q}=\frac{1}{f}-\frac{1}{p}$',
            'loc':'upper right',
            'fit-color':'xkcd:neon purple',
            'save-name':f'{CODENAME[lcode]}-inv'}
    
    qvals_inv = 1/qvals
    qvals_inv_u = np.sqrt(qvals_u/(qvals**2))
    pvals_inv = 1/pvals
    
    data = tk.analyze(pvals_inv, qvals_inv, fit_type='custom', 
                      model_function_custom=thin_lens_eqn_inv,
                      yerr=qvals_inv_u, res=True, chi=True,
                      save=True, meta=meta, dataout=True, 
                      params=[r'$\frac{1}{f}$',], rounding=True, 
                      show=True, path='../figures')
    
    
    f = 1/data['popt'][0]
    f_u = np.sqrt(data['pstd'][0]/(data['popt'][0]**2))
    
    meta2 = {'title':fr'Fit to Thin Lens for {CODENAME[lcode]}',
            'xlabel':r'$p$ (cm)',
            'ylabel':r'$q$ (cm)',
            'data-label':'Measurements',
            'fit-label':r'$q=\left(\frac{1}{f}-\frac{1}{p}\right)^{-1}$',
            'loc':'upper right',
            'fit-color':'xkcd:neon purple',
            'save-name':f'{CODENAME[lcode]}-thin'}
    
    data2 = tk.analyze(pvals, qvals, fit_type='custom', 
                      model_function_custom=thin_lens_eqn, guess=(f,),
                      yerr=qvals_u, res=True, chi=True,
                      save=True, meta=meta2, dataout=True, 
                      params=[r'$f$',], rounding=True, 
                      show=True, path='../figures')
    
    
    meta3 = {'title':fr'Fit to Thick Lens Equation for {CODENAME[lcode]}',
            'xlabel':r'$p$ (cm)',
            'ylabel':r'$q$ (cm)',
            'data-label':'Measurements',
            'fit-label':r'$q=\left(\frac{1}{f}-\frac{1}{p - f_p}\right)^{-1}+b_p$',
            'loc':'upper right',
            'fit-color':'xkcd:cornflower blue',
            'save-name':f'{CODENAME[lcode]}-thick'}
    
    data3 = tk.analyze(pvals, qvals, fit_type='custom', 
                      model_function_custom=thick_lens_eqn,
                      yerr=qvals_u, res=True, chi=True, guess=(f,0,0),
                      save=True, meta=meta3, dataout=True, 
                      params=[r'f', r'f_p', r'b_p'], rounding=True, 
                      show=True, path='../figures')
    
    
    plt.figure(figsize=(15,10))
    plt.plot(data2['plotx'], data2['ploty'], color='XKCD:neon purple', ls='-.',
             label='Thin Lens')
    plt.plot(data3['plotx'], data3['ploty'], color='XKCD:cornflower blue', 
             ls='--', label='Thick Lens')
    plt.errorbar(pvals, qvals, yerr=qvals_u, #xerr=uncertainty_x,
                     markersize='5', fmt='o', elinewidth=3, capsize=5,
                     color='XKCD:red',
                     ecolor='XKCD:slate',
                     label='Measurements',)
    plt.grid('on')
    plt.legend('upper right')
    plt.xlabel('$p$ (cm)', fontsize=45)
    plt.ylabel('$q$ (cm)', fontsize=45)
    plt.title(f'Thick and Thin Lens Fit for {CODENAME[lcode]}')
    plt.legend('upper right')
    plt.savefig(f'../figures/{CODENAME[lcode]}')
    plt.show()
    plt.close()
    
    print(f'{f}+/-{f_u}')
    
def fit_mag(pvals, qvals, qvals_u, gvals, gvals_u):
    pq = qvals/pvals
    
    meta = {'title':'Verification of Magnification Relation',
            'xlabel':r'$\frac{q}{p}$ values (arb.)',
            'ylabel':r'$M$ values (arb.)',
            'data-label':'Measurements',
            'fit-label':r'$M=\zeta\dfrac{q}{p}$',
            'loc':'upper right',
            'fit-color':'xkcd:pumpkin',
            'save-name':'pqm'}
    plotx = np.linspace(min(pq), max(pq), 1000)
    ploty = fit_mag_eqn(plotx, -1)
    # plt.plot(plotx, ploty, ls='-.', color='xkcd:salmon pink')
    
    data = tk.analyze(pq, gvals, fit_type='custom',  guess=(-1,),
                      model_function_custom=fit_mag_eqn,
                      yerr=gvals_u, res=True, chi=True,
                      save=True, meta=meta, dataout=True, 
                      params=[r'$\zeta$',], rounding=True, 
                      show=False, path='../figures')
    

    plt.figure(figsize=(15,10))
    plt.plot(data['plotx'], data['ploty'], color='XKCD:pumpkin', ls='-.',
             label='Fit to Magnification equaton')
    plt.plot(plotx, ploty, color='XKCD:salmon pink', 
             ls='--', label='Theoretically Calculated Magnification')
    plt.errorbar(pq, gvals, yerr=gvals_u, #xerr=uncertainty_x,
                     markersize='5', fmt='o', elinewidth=3, capsize=5,
                     color='XKCD:red',
                     ecolor='XKCD:slate',
                     label='Measurements',)
    plt.grid('on')
    plt.legend('upper right')
    plt.xlabel(r'$\frac{q}{p}$ values (arb.)')
    plt.ylabel('Magnification (arb.)')
    plt.show()
    plt.close()

if __name__ == '__main__':
    

    
    for code in range (1, 5):
        print('------')
        print(CODENAME[code])
        df, data = load_data('../Data/finaldata.csv', code=code)
        fit_thin_lens(data['p'], data['q'], data['qu'], lcode=code)
        fit_mag(data['p'], data['q'], data['qu'], data['g'], data['gu'])
    
    
    
    

    