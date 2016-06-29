# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 16:59:56 2015

@author: jc3e13
"""

import scipy as sp
import scipy.optimize as op
import vertical_velocity_model as vvm


def fitter(Float, hpids, params0, fixed, Plims=(60., 1500.),
           profiles='all', cf_key='diffsq', data_names=['ppos', 'P', 'rho']):
    """This function takes an EM-APEX float, fits a vertical velocity model
    using the given arguments, estimates errors using bootstrapping technique.

    Parameters
    ----------
    Float : EMApexFloat object
        The float to fit.
    hpids : array
        Half profiles to optimise model for.
    params0 : array
        The variable values that need regridding.

    Returns
    -------
    wfi : Dictionary
        Dictionary containing all fitting information.

    Notes
    -----

    """

#    if model == '1':
    still_water_model = vvm.still_water_model_1
#    elif model == '2':
#        still_water_model = vvm.still_water_model_2
#    else:
#        raise ValueError('Cannot find model.')

    __, idxs = Float.get_profiles(hpids, ret_idxs=True)
    hpids = Float.hpid[idxs]

    Pmin, Pmax = Plims

#    print('Fitting model.')

    if profiles == 'all':

        __, ppos = Float.get_timeseries(hpids, 'ppos')
        __, P = Float.get_timeseries(hpids, 'P')
        __, rho = Float.get_timeseries(hpids, 'rho')

        use = (P > Pmin) & (P < Pmax)

        data = [ppos[use], P[use], rho[use]]

        __, w_f = Float.get_timeseries(hpids, 'Wz')
        w_f = w_f[use]

        cargs = (fixed, still_water_model, w_f, data, cf_key)

        p, pcov, info, mesg, ier = op.leastsq(vvm.cost, params0, args=cargs,
                                              full_output=True)

        # Bootstrapping.
#        print('Starting bootstrap.')
        ps = []
        # 200 random data sets are generated and fitted
        for i in range(200):
            rand_idx = sp.random.rand(*w_f.shape) < 0.25
            rand_data = [d[rand_idx] for d in data]
            cargs = (fixed, still_water_model, w_f[rand_idx], rand_data,
                     cf_key)
            rand_params, __ = op.leastsq(vvm.cost, params0, args=cargs)
            ps.append(rand_params)

        ps = sp.array(ps)
        pcov = sp.cov(ps.T)
        pmean = sp.mean(ps, 0)
        pcorr = sp.corrcoef(ps.T)

    elif profiles == 'updown':

        raise RuntimeError('Code being rebuilt.')

#        up_idxs = emapex.up_down_indices(hpids, 'up')
#        up_hpids = hpids[up_idxs]
#        down_idxs = emapex.up_down_indices(hpids, 'down')
#        down_hpids = hpids[down_idxs]
#
#        # We now contain variables in lists.
#        p = 2*[0]
#        ps = 2*[0]
#        pmean = 2*[0]
#        pcov = 2*[0]
#        pcorr = 2*[0]
#        info = 2*[0]
#        mesg = 2*[0]
#        ier = 2*[0]
#
#        # This is a bit of hack since profiles gets changed here... bad.
#        for i, _hpids in enumerate([up_hpids, down_hpids]):
#
#            # Fitting.
#            data = [Float.get_interp_grid(_hpids, P_vals, 'P', data_name)[2]
#                    for data_name in data_names]
#
#            __, __, w_f = Float.get_interp_grid(_hpids, P_vals, 'P', 'Wz')
#
#            cargs = (fixed, still_water_model, w_f, data, cf_key)
#
#            p[i], pcov[i], info[i], mesg[i], ier[i] = \
#                op.leastsq(cost, params0, args=cargs, full_output=True)
#
#            # Bootstrapping.
#            print('Starting bootstrap.')
#
#            bps = []
#            # 100 random data sets are generated and fitted
#            for __ in range(100):
#                rand_idx = np.random.rand(*w_f.shape) < 0.25
#                rand_data = [d[rand_idx] for d in data]
#                cargs = (fixed, still_water_model, w_f[rand_idx], rand_data,
#                         cf_key)
#                rand_params, __ = op.leastsq(cost, params0, args=cargs)
#                bps.append(rand_params)
#
#            ps[i] = np.array(bps)
#            pcov[i] = np.cov(ps[i].T)
#            pmean[i] = np.mean(ps[i], 0)
#            pcorr[i] = np.corrcoef(ps[i].T)

    else:
        raise ValueError("profiles can be 'all' or 'updown'")

    wfi = {
        'params0': params0,
        'fixed': fixed,
#        'model_func': still_water_model,
        'hpids': hpids,
        'profiles': profiles,
        'cf_key': cf_key,
        'Plims': Plims,
        'data_names': data_names,
        'p': p,
        'ps': ps,
        'pmean': pmean,
        'pcov': pcov,
        'pcorr': pcorr,
        'info': info,
        'mesg': mesg,
        'ier': ier}

    return wfi

# TODO: Impliment this again beware against including too many imports (pymc)
#def pymc_fitter(Float, hpids, Plims=(60., 1500.), samples=10000):
#
#    __, ppos = Float.get_timeseries(hpids, 'ppos')
#    __, P = Float.get_timeseries(hpids, 'P')
#    __, rho = Float.get_timeseries(hpids, 'rho')
#
#    Pmin, Pmax = Plims
#    use = (P > Pmin) & (P < Pmax)
#
#    data = [ppos[use], P[use], rho[use]]
#
#    __, w_f = Float.get_timeseries(hpids, 'Wz')
#    w_f = w_f[use]
#
#    def model():
#
#        # Priors.
#        V_0 = pymc.Uniform('V_0', 2.5e-2, 2.7e-2, value=2.6e-2)
#        CA = pymc.Uniform('CA', 1e-2, 1e-1, value=3.5e-2)
#        alpha_p = pymc.Uniform('alpha_p', 1e-7, 1e-5, value=3.76e-6)
#        p_0 = pymc.Uniform('p_0', 1800., 2200., value=2000.)
#        alpha_ppos = pymc.Uniform('alpha_ppos', 1e-7, 1e-5, value=1.156e-6)
#        ppos_0 = pymc.Uniform('ppos_0', 10., 20., value=16.)
#        M = pymc.Uniform('M', 27.0, 27.3, value=27.179)
#        fixed = 7*[None]
#        sig = pymc.Uniform('sig', 0., 0.1, value=0.02)
#
#        @pymc.deterministic()
#        def float_model(V_0=V_0, CA=CA, alpha_p=alpha_p, p_0=p_0,
#                        alpha_ppos=alpha_ppos, ppos_0=ppos_0, M=M):
#            params = [V_0, CA, alpha_p, p_0, alpha_ppos, ppos_0, M]
#            return still_water_model_1(params, data, fixed)
#
#        # Likelihood
#        y = pymc.Normal('y', mu=float_model, tau=1./0.009**2, value=w_f,
#                        observed=True)
#
#        return locals()
#
#    M = pymc.MCMC(model())
#    burn = samples/2
#    thin = 5
#    M.sample(samples, burn, thin)
#    pymc.Matplot.plot(M, common_scale=False)
#
#    chain = np.asarray([M.trace('V_0')[:], M.trace('CA')[:],
#                        M.trace('alpha_p')[:], M.trace('p_0')[:],
#                        M.trace('alpha_ppos')[:], M.trace('ppos_0')[:],
#                        M.trace('M')[:]])
#    labels = [r'$V_0$', r'$C_D^*$', r'$\alpha_p$', r'$p_0$', r'$\alpha_k$',
#              r'$k_0$', r'$M$']
#    triangle.corner(np.transpose(chain), labels=labels)
#
#    return M