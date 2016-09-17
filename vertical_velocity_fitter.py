# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 16:59:56 2015

@author: jc3e13
"""

import os
import numpy as np
import scipy.optimize as op
import vertical_velocity_model as vvm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import corner
import pickle
import my_paths

def fitter(Float, p0, pfixed, **kwargs):
    """This function takes an EM-APEX float, fits a vertical velocity model
    using the given arguments, estimates errors using bootstrapping technique.

    Parameters
    ----------
    Float : EMApexFloat object
        The float to fit.
    p0 : array
        Initial values for the model parameters.
    pfixed : array
        Array same size as p0 with fixed value or None for non-fixed.
    hpids : array
        Half profiles to over which to optimise.
    Plims : tuple
        Pressure limits of fit (min, max).
    profiles: string
        Set to 'all' to use both up and down profiles, otherwise set to
        'updown' to perform fit separated on up and down profiles.
    cf_key: string
        Argument to select cost function, see docstring in model file for
        details.
    save_name : strings
        Set this to a path/filename.p in order to save fit output dictionary
        as a pickle. Data will not be saved otherwise.

    Returns
    -------
    wfi : Dictionary
        Dictionary containing all fitting information.

    Notes
    -----

    """

    hpids = kwargs.pop('hpids', np.arange(600))
    Plims = kwargs.pop('Plims', (60., 2000.))
    profiles = kwargs.pop('profiles', 'all')
    cf_key = kwargs.pop('cf_key', 'diffsq')
    save_path = kwargs.pop('save_path', my_paths.processed)
    save_name = kwargs.pop('save_name', None)
    w_f_thresh = kwargs.pop('w_f_thresh', 0.1)
    N_bootstrap = kwargs.pop('N_bootstrap', 200)
    method = kwargs.pop('method', 'L-BFGS-B')

    __, idxs = Float.get_profiles(hpids, ret_idxs=True)
    hpids = Float.hpid[idxs]

    Pmin, Pmax = Plims

    # Extract data.
    __, ppos = Float.get_timeseries(hpids, 'ppos')
    __, P = Float.get_timeseries(hpids, 'P')
    __, rho = Float.get_timeseries(hpids, 'rho')
    __, w_f = Float.get_timeseries(hpids, 'Wz')
    invalid = (P < Pmin) | (P > Pmax) | (np.abs(w_f) < w_f_thresh)
    use = ~invalid
    w_f = w_f[use]

    if profiles == 'all':
        model_func_name = 'still_water_model_1'
        data = [ppos[use], P[use], rho[use]]
        data_names = kwargs.pop('data_names', ['ppos', 'P', 'rho'])

        default_param_names = ['$V_0$', '$C_D$', r'$\alpha_p$', '$p_0$',
                               r'$\alpha_k$', '$k_0$', '$M$']
        param_names = kwargs.pop('param_names', default_param_names)

        default_bounds = [(0., 1.), (1e-3, 2e-1), (1e-7, 1e-5), (0., 6000.),
                          (5e-7, 3e-6), (9., 227.), (27.5, 28.)]
        param_bounds = kwargs.pop('param_bounds', default_bounds)

    elif profiles == 'updown':
        model_func_name = 'still_water_model_1_updown'
        __, up = Float.get_timeseries(hpids, 'ascent_ctd')
        data = [ppos[use], P[use], rho[use], up[use]]
        data_names = kwargs.pop('data_names', ['ppos', 'P', 'rho', 'ascent_ctd'])

        default_param_names = ['$V_0$', '$C_D^u$', '$C_D^d}$', r'$\alpha_p$',
                               '$p_0$', r'$\alpha_k$', '$k_0$', '$M$']
        param_names = kwargs.pop('param_names', default_param_names)

        default_bounds = [(0., 1.), (1e-3, 2e-1), (1e-3, 2e-1), (1e-7, 1e-5),
                          (0., 6000.), (5e-7, 3e-6), (9., 227.), (27.5, 28.)]
        param_bounds = kwargs.pop('param_bounds', default_bounds)
    else:
        raise ValueError("profiles can be 'all' or 'updown'")

    still_water_model = getattr(vvm, model_func_name)
    cargs = (pfixed, still_water_model, w_f, data, cf_key)

    def cost(params, fixed, model, wf, data, cf_key='diffsq'):
        cost_ = vvm.cost(params, fixed, model, wf, data, cf_key)
        return np.sum(cost_)

    res = op.minimize(cost, p0, args=cargs, method=method, bounds=param_bounds)

#    minimizer_kwargs = {'args': cargs, 'bounds': param_bounds}
#    res = op.basinhopping(cost, p0, minimizer_kwargs=minimizer_kwargs)

    p = res.x

#    p, __, info, mesg, ier = op.leastsq(vvm.cost, p0, args=cargs,
#                                        full_output=True)

    ps = []
    for i in range(N_bootstrap):
        rand_idx = np.random.rand(*w_f.shape) < 0.25
        rand_data = [d[rand_idx] for d in data]
        cargs = (pfixed, still_water_model, w_f[rand_idx], rand_data,
                 cf_key)
#        rand_params, __ = op.leastsq(vvm.cost, p0, args=cargs)
        res = op.minimize(cost, p0, args=cargs, method=method,
                          bounds=param_bounds)
        ps.append(res.x)

    ps = np.array(ps)
    pcov = np.cov(ps.T)
    pmean = np.mean(ps, 0)
    pcorr = np.corrcoef(ps.T)

    wfi = {
        'p0': p0,
        'pfixed': pfixed,
        'param_names': param_names,
        'param_bounds': param_bounds,
        'hpids': hpids,
        'profiles': profiles,
        'model_func_name': model_func_name,
        'cf_key': cf_key,
        'Plims': Plims,
        'data_names': data_names,
        'p': p,
        'ps': ps,
        'pmean': pmean,
        'pcov': pcov,
        'pcorr': pcorr,
#        'info': info,
#        'mesg': mesg,
#        'ier': ier,
        'res': res,
        'w_f_thresh': w_f_thresh}

    if save_name is not None:
        with open(os.path.join(save_path, save_name), 'wb') as f:
            pickle.dump(wfi, f)

    return wfi


def assess_w_fit(Float, save_figures=False, save_id='', save_dir=''):
    """ """

    wfi = Float.__wfi
    hpids = wfi['hpids']
    floatID = Float.floatID

    # Histogram of vertical water velocity.
    __, P = Float.get_timeseries(hpids, 'P')
    __, w_f = Float.get_timeseries(hpids, 'Wz')
    __, Ww = Float.get_timeseries(hpids, 'Ww')

    nans = np.isnan(Ww)
    P, w_f, Ww = P[~nans], w_f[~nans], Ww[~nans]

    w_f_thresh = wfi['w_f_thresh']
    Pmin, Pmax = wfi['Plims']

    invalid = (P < Pmin) | (P > Pmax) | (np.abs(w_f) < w_f_thresh)
    use = ~invalid

    Ww_fit = Ww[use]
    Ww_mean = np.nanmean(Ww_fit)
    Ww_std = np.nanstd(Ww_fit)

    fig = plt.figure(figsize=(3.125, 3))
    bins = np.arange(-0.10, 0.102, 0.002)
    __, __, patches1 = plt.hist(Ww, bins=bins, histtype='stepfilled')
    plt.setp(patches1, 'facecolor', 'b', 'alpha', 0.75)
    __, __, patches2 = plt.hist(Ww_fit, bins=bins, histtype='stepfilled')
    plt.setp(patches2, 'facecolor', 'r', 'alpha', 0.75)

    plt.xlim(np.min(bins), np.max(bins))
    plt.xlabel('$W_w$ (m s$^{-1}$)')
    plt.xticks(rotation=45)
    title_str = ("Float {}\nmean = {:1.2e} m s$^{{-1}}$\nstd = {:1.2e} "
                 "m s$^{{-1}}$").format(floatID, Ww_mean, Ww_std)
    plt.title(title_str)

    if save_figures:
        name = save_id + '_ww_histogram.png'
        fname = os.path.join(save_dir, name)
        plt.savefig(fname, bbox_inches='tight')
        plt.close(fig)

    # Time series of different velocity measures.
    hpid_use = np.arange(10,17)
    __, idx = Float.get_profiles(hpid_use, ret_idxs=True)

    fig = plt.figure(figsize=(6.5, 3))

    time, Ww = Float.get_timeseries(hpid_use, 'Ww')
    __, Wz = Float.get_timeseries(hpid_use, 'Wz')
    __, Ws = Float.get_timeseries(hpid_use, 'Ws')
    time[time < 700000.] = np.NaN
    plt.plot(time, Ww)
    plt.plot(time, Wz)
    plt.plot(time, Ws)
    plt.ylabel('$W_w$, $W_f$, $W_s$ (m s$^{-1}$)')
    plt.xlabel('Time')
    plt.xticks(rotation=45)
    title_str = ("Float {}, half profiles {}").format(floatID, Float.hpid[idx])
    plt.title(title_str)
    plt.legend(['$W_w$', '$W_f$', '$W_s$'])

    if save_figures:
        name = save_id + '_ww_wf_w0_timeseries.png'
        fname = os.path.join(save_dir, name)
        plt.savefig(fname, bbox_inches='tight')
        plt.close(fig)

    # Scatter section of water velocity.

    __, z = Float.get_timeseries(Float.hpid, 'z')
    __, Ww = Float.get_timeseries(Float.hpid, 'Ww')
    __, d = Float.get_timeseries(Float.hpid, 'dist_ctd')

    fig = plt.figure(figsize=(6.5, 3))
    plt.scatter(d, z, c=Ww, edgecolor='none', cmap=plt.get_cmap('bwr'))
    plt.ylim(np.min(z), np.max(z))
    plt.xlim(np.min(d), np.max(d))
    plt.xlabel('Distance (km)')
    plt.ylabel('Depth (m)')
    plt.xlim(np.min(Float.dist), np.max(Float.dist))
    title_str = ("Float {}").format(floatID)
    plt.title(title_str)
    cbar = plt.colorbar(orientation='horizontal', extend='both')
    cbar.set_label('$W_w$ (m s$^{-1}$)')
    plt.clim(-0.15, 0.15)

    if save_figures:
        name = save_id + '_ww_scatter_section.png'
        fname = os.path.join(save_dir, name)
        plt.savefig(fname, bbox_inches='tight')
        plt.close(fig)

#    # Varience of Ww2 with N2
#    Ww2 = Float.rWw.flatten(order='F')**2
#    N2 = Float.rN2.flatten(order='F')
#    Ww2 = Ww2[N2 > 0.]
#    N = np.sqrt(N2[N2 > 0.])
#    bins = np.linspace(0., np.max(N), 20)
#    idxs = np.digitize(N, bins)
#    Wmean = []
#    Wstd = []
#    Nm = (bins[1:] + bins[:-1])/2.
#    N0 = 1e-4
#    for i in xrange(len(bins) - 1):
#        Wmean.append(np.nanmean(Ww2[idxs == i+1]))
#        Wstd.append(np.nanstd(Ww2[idxs == i+1]))
#    plt.errorbar(Nm, Wmean, Wstd)
#    plt.plot(Nm, 0.25*N0/Nm)

    # Parameter estimates and correlations.
    pnames = wfi['param_names']
    N = len(pnames)
    ticks = np.arange(0.5, N, 1)

    fig = plt.figure(figsize=(3.125, 3))
    plt.pcolormesh(np.flipud(wfi['pcorr']), cmap=plt.get_cmap('PiYG'))
    cbar = plt.colorbar()
    cbar.set_label('Correlation')
    plt.clim(-1, 1)
    plt.xticks(ticks, pnames)
    plt.yticks(ticks, pnames[::-1])
    title_str = ("Float {}").format(floatID)
    plt.title(title_str)

    if save_figures:
        name = save_id + '_param_corr.png'
        fname = os.path.join(save_dir, name)
        plt.savefig(fname, bbox_inches='tight')
        plt.close(fig)

    not_fixed = np.array([(p is None) for p in wfi['pfixed']])
    ps = wfi['ps'][:, not_fixed]
    p = wfi['p'][not_fixed]
    params0 = wfi['p0'][not_fixed]
    corner.corner(ps, labels=np.array(pnames)[not_fixed])
    fig = plt.gcf()
    fig.set_size_inches(7, 7)
    axs = fig.axes
    N = np.shape(ps)[1]

    formatter = ticker.ScalarFormatter(useOffset=False)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 2))

    for i in xrange(N):
        for j in xrange(N):
            idx = i*N + j
            if i == N - 1:
                axs[idx].xaxis.set_major_formatter(formatter)
            if (j == 0) and (i > 0):
                axs[idx].yaxis.set_major_formatter(formatter)
            if i == j:
                axs[idx].vlines(p[i], *axs[idx].get_ylim(), color='r')
                axs[idx].vlines(params0[i], *axs[idx].get_ylim(),
                                color='g')

    if save_figures:
        name = save_id + '_param_matrix_scatter.png'
        fname = os.path.join(save_dir, name)
        plt.savefig(fname, bbox_inches='tight')
        plt.close(fig)
