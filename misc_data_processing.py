# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 15:44:20 2016

@author: jc3e13
"""

import os
import numpy as np
import scipy.signal as sig
import pickle
import window as wdw
import TKED_parameterisations as TKED


def adiabatic_level_float(Float, P_bin_width, save_dir):
    """Smooth buoyancy frequency and save to file."""

    Pg = getattr(Float, 'P')
    SAg = getattr(Float, 'SA')
    Tg = getattr(Float, 'T')
    lats = getattr(Float, 'lat_start')

    N2_ref = np.NaN*Pg.copy()

    for i, (P, SA, T, lat) in enumerate(zip(Pg.T, SAg.T, Tg.T, lats)):
        print("hpid: {}".format(Float.hpid[i]))
        N2_ref[:, i] = TKED.adiabatic_level(P, SA, T, lat, P_bin_width)

    save_name = "{:g}_N2_ref_{:g}dbar.p".format(Float.floatID, P_bin_width)
    file_path = os.path.join(save_dir, save_name)

    pickle.dump(N2_ref, open(file_path, 'wb'))


def smooth_density_float(Float, z_bin_width, save_dir):
    """Smooth potential density and save to a file.

    NOTE: This should be a more general function to smooth anything.

    """

    srho_1 = np.nan*Float.rho_1.copy()

    for i in xrange(len(Float.hpid)):
        print("hpid: {g:}".format(Float.hpid[i]))
        srho_1[:, i] = wdw.moving_polynomial_smooth(
            Float.z[:, i], Float.rho_1[:, i], width=100., deg=1.)

    save_name = "srho_{:g}_{:g}mbin.p".format(Float.floatID, z_bin_width)
    file_path = os.path.join(save_dir, save_name)

    pickle.dump(srho_1, open(file_path, 'wb'))


def w_scales_float(Float, hpids, xvar, x, width=10., overlap=-1., lc=30., c=1.,
                   eff=0.2, btype='highpass', we=1e-3, ret_noise=False,
                   ret_VKE=False):
    """Wrapper for the w_scales function that takes Float objects.

    For xvar == timeheight, the signal is first low pass filtered in time and
    then high pass filtered in space. lc must be an array with two elements,
    the cut off time period and the cut off length.

    """

    __, idxs = Float.get_profiles(hpids, ret_idxs=True)
    Np = len(idxs)
    dx = x[1] - x[0]

    if xvar == 'time':
        __, __, w = Float.get_interp_grid(hpids, x, 'dUTC', 'Ww')
        __, __, N2 = Float.get_interp_grid(hpids, x, 'dUTC', 'N2_ref')
    elif xvar == 'height':
        __, __, w = Float.get_interp_grid(hpids, x, 'z', 'Ww')
        __, __, N2 = Float.get_interp_grid(hpids, x, 'z', 'N2_ref')
    elif xvar == 'eheight':
        __, __, w = Float.get_interp_grid(hpids, x, 'zw', 'Ww')
        __, __, N2 = Float.get_interp_grid(hpids, x, 'zw', 'N2_ref')
    elif (xvar == 'timeheight') or (xvar == 'timeeheight'):
        # First low-pass in time.
        dt = 1.
        t = np.arange(0., 15000., dt)
        __, __, wt = Float.get_interp_grid(hpids, t, 'dUTC', 'Ww')
        xc = 1./lc[0]  # cut off wavenumber
        normal_cutoff = xc*dt*2.  # Nyquist frequency is half 1/dx.
        b, a = sig.butter(4, normal_cutoff, btype='lowpass')
        wf = sig.filtfilt(b, a, wt, axis=0)

        # Now resample in depth space.
        w = np.zeros((len(x), Np))

        if xvar == 'timeheight':
            __, __, N2 = Float.get_interp_grid(hpids, x, 'z', 'N2_ref')
            __, __, it = Float.get_interp_grid(hpids, x, 'z', 'dUTC')
        elif xvar == 'timeeheight':
            __, __, N2 = Float.get_interp_grid(hpids, x, 'zw', 'N2_ref')
            __, __, it = Float.get_interp_grid(hpids, x, 'zw', 'dUTC')

        for i in xrange(Np):
            w[:, i] = np.interp(it[:, i], t, wf[:, i])

        btype = 'highpass'
        lc = lc[1]
    else:
        raise ValueError("xvar must either be 'time', 'height', 'eheight' or "
                         "'timeheight'.")

    epsilon = np.zeros_like(w)
    kappa = np.zeros_like(w)
    epsilon_noise = np.zeros_like(w)
    noise_flag = np.zeros_like(w, dtype=bool)
    for i in xrange(Np):
        wp, N2p = w[:, i], N2[:, i]
        epsilon[:, i], kappa[:, i], epsilon_noise[:, i], noise_flag[:, i] = \
            TKED.w_scales(wp, x, N2p, dx, width, overlap, lc, c, eff, btype,
                          we, True)

    if ret_noise:
        return epsilon, kappa, epsilon_noise, noise_flag
    else:
        return epsilon, kappa


def thorpe_float(Float, hpids, xvar='rho_1', zvar='z', R0=0.25, acc=2e-3,
                 xhinge=1030., use_int=False):

    pfls, idxs = Float.get_profiles(hpids, ret_idxs=True)

    thorpe_scales = np.zeros_like(Float.P[:, idxs])
    thorpe_disp = np.zeros_like(Float.P[:, idxs])
    Nsq_thorpe = np.zeros_like(Float.P[:, idxs])

    for i, pfl in enumerate(pfls):
        nnan = ~np.isnan(pfl.P)
        x = getattr(pfl, xvar)[nnan]
        z = getattr(pfl, zvar)[nnan]

        if use_int:
            __, __, x_int = TKED.intermediate_profile(x, xhinge, acc)
            x = x_int

        thorpe_scales[nnan, i], thorpe_disp[nnan, i], Nsq_thorpe[nnan, i], \
            __, __, __, __ = \
            TKED.thorpe_scales(z, x, R0=R0, acc=acc, full_output=True)

    return thorpe_scales, thorpe_disp, Nsq_thorpe

#def analyse_profile(Pfl, params=default_params):
#    """ """
#
#    if params['zmin'] is None:
#        params['zmin'] = np.nanmin(Pfl.z)
#
#    # First remove NaN values and interpolate variables onto a regular grid.
#    dz = params['dz']
#    z = np.arange(params['zmin'], params['zmax']+dz, dz)
#    U = Pfl.interp(z, 'zef', 'U_abs')
#    V = Pfl.interp(z, 'zef', 'V_abs')
#    dUdz = Pfl.interp(z, 'zef', 'dUdz')
#    dVdz = Pfl.interp(z, 'zef', 'dVdz')
#    strain = Pfl.interp(z, 'z', 'strain_z')
#    N2_ref = Pfl.interp(z, 'z', 'N2_ref')
#    lat = (Pfl.lat_start + Pfl.lat_end)/2.
#
#    return analyse(z, U, V, dUdz, dVdz, strain, N2_ref, lat, params)
#
#
#def analyse_float(Float, hpids, params=default_params):
#    """ """
#    # Nothing special for now. It doesn't even work.
#    __, idxs = Float.get_profiles(hpids, ret_idxs=True)
#    return [analyse_profile(Pfl, params) for Pfl in Float.Profiles[idxs]]