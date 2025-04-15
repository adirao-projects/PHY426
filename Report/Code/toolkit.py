# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 12:34:34 2024
Updated on Mon Oct 14 21:22:09 2024
Updated on Mon Feb 10 09:51:03 2025
Updated on Sat Mar 01 21:03:54 2025
Updated on Mon Mar 03 14:45:13 2025
Updated on Thu Mar 06 17:57:55 2025
Updated on Fri Mar 21 14:46:28 2025
Updated on Sun Mar 23 19:18:31 2025

Lab Toolkit

@author: Aditya K. Rao
@github: @adirao-projects
"""
import os
import math
import textwrap
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import gridspec

font = {'family' : 'DejaVu Sans',
        'size'   : 30}

plt.rc('font', **font)

def fit_data(xdata, ydata, fit_type, override=(False, None),
             yerr=None, xerr=None, guess=None,
             res=False, chi=False, model_function_custom=None):
    """
    

    Parameters
    ----------
    xdata : TYPE
        DESCRIPTION.
    ydata : TYPE
        DESCRIPTION.
    fit_type : TYPE
        DESCRIPTION.
    override : TYPE, optional
        DESCRIPTION. The default is False.
    yerr : TYPE, optional
        DESCRIPTION. The default is None.
    res : TYPE, optional
        DESCRIPTION. The default is False.
    chi : TYPE, optional
        DESCRIPTION. The default is False.
    xerr : TYPE, optional
        DESCRIPTION. The default is None.
    model_function_custom : TYPE, optional
        DESCRIPTION. The default is None.
    guess : TYPE, optional
        DESCRIPTION. The default is None.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    def chi_sq_red(measured_data:list[float], expected_data:list[float],
               uncertainty:list[float], v: int):
        if isinstance(uncertainty, float):
            uncertainty = [uncertainty]*len(measured_data)
        chi_sq = 0

        for m, e, u in zip(measured_data, expected_data, uncertainty):
            chi_sq += ((m-e)/u)**2

        chi_sq = (1/v)*chi_sq

        return chi_sq


    def residual_calculation(y_data: list, exp_y_data) -> list[float]:
        residuals = []
        for v, u in zip(y_data, exp_y_data):
            residuals.append(u-v)

        return residuals

    def model_function_linear_int(x, m, c):
        return m*x+c

    def model_function_exp(x, a, b, c):
        return a*np.exp(b*(x-c))

    def model_function_log(x, a, b):
        return b*np.log(x+a)

    def model_function_linear(x, m):
        return m*x

    def model_function_xlnx(x, b, c):
        return b*x*(np.log(x)) + c

    def model_function_ln(x, b, c):
        return b*(np.log(x)) + c

    def model_function_sqrt(x, a):
        return a*np.sqrt(x)

    model_functions = {
        'linear' : model_function_linear,
        'linear-int' : model_function_linear_int,
        'xlnx' : model_function_xlnx,
        'log' : model_function_log,
        'ln' : model_function_ln,
        'exp' : model_function_exp,
        'sqrt' : model_function_sqrt,
        'custom' : model_function_custom
        }

    try:
        model_func = model_functions[fit_type]

    except:
        raise ValueError(f'Unsupported fit-type: {fit_type}')


    if not override[0]:
        new_xdata = np.linspace(min(xdata), max(xdata), num=100)

        if isinstance(yerr, int):
            abs_sig =True
        else:
            abs_sig = False

        if guess is not None:
            popt, pcov = curve_fit(model_func, xdata, ydata, sigma=yerr,
                               maxfev=20000, absolute_sigma=abs_sig, p0=guess)
        else:
            popt, pcov = curve_fit(model_func, xdata, ydata, sigma=yerr,
                               maxfev=20000, absolute_sigma=abs_sig)

        param_num = len(popt)

        exp_ydata = model_func(xdata,*popt)

        deg_free = len(xdata) - param_num

        new_ydata = model_func(new_xdata, *popt)

        residuals = None
        chi_sq = None

        if res:
            residuals = residual_calculation(exp_ydata, ydata)

        if chi:
            chi_sq = chi_sq_red(ydata, exp_ydata, yerr, deg_free)

        data_output = {
            'popt' : popt,
            'pcov' : pcov,
            'plotx': new_xdata,
            'ploty': new_ydata,
            'chisq' : chi_sq,
            'residuals' : residuals,
            'pstd' : np.sqrt(np.diag(pcov))
            }

        return data_output

    return model_func(xdata, *override[1:])

def plot_data(xdata, ydata, plot_x, plot_y,
              residuals=[], yerr=[], xerr=[],
              res=True, dataout=False, save=False, meta=None, show=False,
              imgpath='./figures/'):
    """
    Options to plot with and without residuals. Defaults to including residuals
    """
    fig = plt.figure(figsize=(14,14))
    gs = gridspec.GridSpec(ncols=11, nrows=11, figure=fig)
    if res:
        main_fig = fig.add_subplot(gs[:6,:])
        res_fig = fig.add_subplot(gs[8:,:])
        res_fig.grid('on')
    else:
        main_fig = fig.add_subplot(gs[:,:])

    main_fig.grid('on')

    if isinstance(yerr, int):
        yerr = [yerr]*len(xdata)

    elif len(yerr) == 0:
        for y in ydata:
            yerr.append(y.std_dev)

    metadata = {'title' : 'TITLE',
                'xlabel' : 'XLABEL',
                'ylabel' : 'YLABEL',
                'chisq' : 'inf',
                'fit-label': 'Best Fit',
                'data-label': 'Data',
                'save-name' : 'PLOT',
                'fit-color': 'tab:blue',
                'err-color':'tab:gray',
                'data-color':'tab:red',
                'loc' : 'lower right'}

    if meta is not None:
        metadata.update(meta)

    main_fig.set_title(metadata['title'], fontsize = 46)
    if len(xerr)==0:
        main_fig.errorbar(xdata, ydata, yerr=yerr, #xerr=uncertainty_x,
                          markersize='5', fmt='o', elinewidth=3, capsize=5,
                          color=metadata['data-color'],
                          ecolor=metadata['err-color'],
                          label=metadata['data-label'],)
    else:
        main_fig.errorbar(xdata, ydata, yerr=yerr, xerr=xerr,
                          markersize='5', fmt='o', elinewidth=3, capsize=5,
                          color=metadata['data-color'],
                          label=metadata['data-label'],
                          ecolor=metadata['err-color'])

    main_fig.plot(plot_x, plot_y, linestyle='dashed',
                  color=metadata['fit-color'],
                  label=metadata['fit-label'])

    main_fig.set_xlabel(metadata['xlabel'], fontsize = 45)
    main_fig.set_ylabel(metadata['ylabel'], fontsize = 45)
    main_fig.legend(loc=metadata['loc'])

    if res:
        res_fig.errorbar(xdata, residuals, markersize='5', fmt='o', yerr=yerr,
                         color=metadata['data-color'], elinewidth=3, capsize=5,
                         ecolor=metadata['err-color'], alpha=0.7)
        res_fig.axhline(y=0, linestyle='dashed', color=metadata['fit-color'])
        res_fig.set_title('Residuals')

    if save:
        plt.savefig(f"{imgpath}/{meta['save-name']}")

    if show:
        plt.show()

def quick_analyze(*params):
    """
    Wrapper of analyze() to avoid depreciation.

    Parameters
    ----------
    *params : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return analyze(*params)

def analyze(xdata, ydata, fit_type, model_function_custom=None,
            yerr=None, xerr=[], res=False, chi=False, guess=None,
            save=False, meta=None, dataout=True, params=None,
            rounding=True, show=False, path='./figures'):
    """
    

    Parameters
    ----------
    xdata : TYPE
        DESCRIPTION.
    ydata : TYPE
        DESCRIPTION.
    fit_type : TYPE
        DESCRIPTION.
    model_function_custom : TYPE, optional
        DESCRIPTION. The default is None.
    yerr : TYPE, optional
        DESCRIPTION. The default is None.
    xerr : TYPE, optional
        DESCRIPTION. The default is [].
    res : TYPE, optional
        DESCRIPTION. The default is False.
    chi : TYPE, optional
        DESCRIPTION. The default is False.
    guess : TYPE, optional
        DESCRIPTION. The default is None.
    save : TYPE, optional
        DESCRIPTION. The default is False.
    meta : TYPE, optional
        DESCRIPTION. The default is None.
    dataout : TYPE, optional
        DESCRIPTION. The default is True.
    params : TYPE, optional
        DESCRIPTION. The default is None.
    rounding : TYPE, optional
        DESCRIPTION. The default is True.
    show : TYPE, optional
        DESCRIPTION. The default is False.
    path : TYPE, optional
        DESCRIPTION. The default is './figures'.

    Returns
    -------
    data : TYPE
        DESCRIPTION.

    """


    data = fit_data(xdata, ydata, fit_type=fit_type, yerr=yerr,
                    chi=chi, res=res, guess=guess,
                    model_function_custom=model_function_custom)

    plot_data(xdata, ydata, data['plotx'], data['ploty'],
              residuals=data['residuals'], yerr=yerr,
              xerr=xerr, res=res, save=save, meta=meta,
              show=show, imgpath=path)
    if show:
        if not rounding:
            poptvals = data['popt']
            pstdvals = data['pstd']

        else:
            poptvals = []
            pstdvals = []

            for n, t, d in zip(params,  data['popt'], data['pstd']):
                d = round(d, -int(np.floor(np.log10(np.abs(d)))))
                dp = len(str(d).split(".")[1])
                t = round(t, dp)

                poptvals.append(t)
                pstdvals.append(d)

        if chi:
            poptvals.append(data['chisq'])
            pstdvals.append(0.0)
            params.append('chi2r')

        param_print(params, poptvals, pstdvals, latex=False)

    if save and dataout:
        with open(f'{path}/{meta["save-name"]}.dat', 'w') as f:
            for n, t, d in zip(params,  data['popt'], data['pstd']):
                if rounding:
                    d = round(d, -int(np.floor(np.log10(np.abs(d)))))
                    dp = len(str(d).split(".")[1])
                    t = round(t, dp)

                f.write(fr'{n} : ${n} = {t} \pm {d}$\n')

            chisq = round(data['chisq'], \
                -int(np.floor(np.log10(np.abs(data['chisq'])))))
            f.write(r'chisq : $\chi^2_{\text{red}}$ = ' + str(chisq) + '\n')

        return data

    if dataout:
        return data

def block_print(data: list[str], title: str, delimiter='=') -> None:
    """
    Prints a formated block of text with a title and delimiter

    Parameters
    ----------
    data : list[str]
        Text to be printed (should be input as one block of text).
    title : str
        Title of the data being output.
    delimiter : str, optional
        Delimiter to be used. The default is '='.

    Returns
    -------
    None.

    Examples
    --------
    >>> r_log = 100114.24998718781
    >>> r_dec = 0.007422298127465114
    >>> data = [f'r^2 value (log): {r_log}', 
                f'r^2 value (real): {r_dec}']
    >>> block_print(data, 'Regression Coefficient', '=')
    ============================ Regression Coefficient ============================
    r^2 value (log): 100114.24998718781
    r^2 value (real): 0.007422298127465114
    ================================================================================
    """
    term_size = os.get_terminal_size().columns

    breaks = 1
    str_len = len(title)+2
    while  str_len >= term_size:
        breaks += 1
        str_len = math.ceil(str_len/2)


    str_chunk_len = math.ceil(len(title)/breaks)
    str_chunks = textwrap.wrap(title, str_chunk_len)
    output = ''
    for chunk in str_chunks:
        border = delimiter*(math.floor((term_size - str_chunk_len)/2)-1)
        output = f'{border} {chunk} {border}\n'

    output=output[:-1]

    output+= '\n'+ '\n'.join(data) + '\n'
    output+=delimiter*term_size

    print(output)


def param_print(names, popt, pstd, latex=True):
    """
    Prints all parameters for easy input into a LaTeX document.

    Parameters
    ----------
    names : list[str]
        Symbols/Names of each parameter
    popt : list[float]
        Value of each parameter.
    pstd : list[float]
        Standard deviation/uncertainty associated with each parameter.
    latex : bool, optional
        Print in LaTeX format. The default is True.
        
    Returns
    -------
    None.
    """

    data = []
    for n, t, d in zip(names, popt, pstd):
        if latex:
            data.append(fr'${n} = {t} \pm {d}$')

        else:
            data.append(f'{n} = {t} +/- {d}')
    block_print(title='Parameters', data=data)

def numerical_methods(method_type, args=None, custom_method=None):
    """
    

    Parameters
    ----------
    method_type : TYPE
        DESCRIPTION.
    args : TYPE, optional
        DESCRIPTION. The default is None.
    custom_method : TYPE, optional
        DESCRIPTION. The default is None.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    def gaussxw(n):
        # Initial approximation to roots of the Legendre polynomial
        a = np.linspace(3,4*n-1,n)/(4*n+2)
        x = np.cos(np.pi*a+1/(8*n*n*np.tan(a)))

        # Find roots using Newton's method
        epsilon = 1e-15
        delta = 1.0
        while delta>epsilon:
            p0 = np.ones(n,float)
            p1 = np.copy(x)
            for k in range(1,n):
                p0,p1 = p1,((2*k+1)*x*p1-k*p0)/(k+1)
            dp = (n+1)*(p0-x*p1)/(1-x*x)
            dx = p1/dp
            x -= dx
            delta = max(abs(dx))

        # Calculate the weights
        w = 2*(n+1)*(n+1)/(n*n*(1-x*x)*dp*dp)

        return x, w

    def gaussxwab(n,a,b):
        x,w = gaussxw(n)
        return 0.5*(b-a)*x+0.5*(b+a),0.5*(b-a)*w

    methods = {
    'gausswx' : gaussxw,
    'gaussxwab' : gaussxwab,
    'custom' : custom_method
    }

    try:
        method = methods[method_type]

    except:
        raise ValueError(f'Unsupported method-type: {method_type}')

    return method(*args)

def interpolation_methods(method_type, args=None, custom_method=None):
    """
    

    Parameters
    ----------
    method_type : TYPE
        DESCRIPTION.
    args : TYPE, optional
        DESCRIPTION. The default is None.
    custom_method : TYPE, optional
        DESCRIPTION. The default is None.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    methods = {
    'custom' : custom_method
    }

    try:
        method = methods[method_type]

    except:
        raise ValueError(f'Unsupported method-type: {method_type}')

    return method(*args)
