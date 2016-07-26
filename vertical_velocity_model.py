# -*- coding: utf-8 -*-
"""
Created on Tue Apr 08 14:33:17 2014

@author: jc3e13
"""

import numpy as np


def still_water_model_1(params, data, fixed):
    """Calculates vertical velocity for an EM-APEX float if it were moving in
    still water.

    Parameters
    ----------
    params : array
        The model parameters, in this order:
        V_0, float volume when neutrally buoyant. (1.) [m3].
        CA, combination of drag and surface area. (1.) [m2].
        alpha_p, coeff. of expansion with pressure. (3.76e-6) [-].
        p_0, pressure when float is neutrally buoyant. (2000.) [dbar].
        alpha_ppos, coeff. of expansion with piston position. (1.156e-6) [m3].
        ppos_0, piston position when neutrally buoyant. (16.) [-].
        M, float mass. (27.179) [kg].
    data : array_like
        The data used in the model:
        ppos, piston position. [-].
        p, pressure. [dbar].
        rho, density. [kg m-3].
    fixed : array
        Array same size as params with fixed value or None for non-fixed.

    Returns
    -------
    w : array
        Vertical velocity for given parameters and data.

    """
    ppos, p, rho = data

    g = -9.8  # Gravitational acceleration [m s-2].

    # Apply fixed value for those parameters that are fixed.
    for i, val in enumerate(fixed):
        if val is not None:
            params[i] = val

    V_0, CA, alpha_p, p_0, alpha_ppos, ppos_0, M = params

    # Float volume
    V = V_0*(1 - alpha_p*(p - p_0)) + alpha_ppos*(ppos - ppos_0)

    return np.sign(rho*V - M)*np.sqrt(np.abs(g*(M - rho*V))/(rho*CA))


def still_water_model_1_updown(params, data, fixed):
    """Calculates vertical velocity for an EM-APEX float if it were moving in
    still water, using a separate drag coefficient for up and down profile
    data.

    Parameters
    ----------
    params : array
        The model parameters, in this order:
        V_0, float volume when neutrally buoyant. (1.) [m3].
        CA_up, combination of drag and surface area. (1.) [m2].
        CA_down, drag for down profiles. (1.) [m2].
        alpha_p, coeff. of expansion with pressure. (3.76e-6) [-].
        p_0, pressure when float is neutrally buoyant. (2000.) [dbar].
        alpha_ppos, coeff. of expansion with piston position. (1.156e-6) [m3].
        ppos_0, piston position when neutrally buoyant. (16.) [-].
        M, float mass. (27.179) [kg].
    data : array_like
        The data used in the model:
        ppos, piston position. [-].
        p, pressure. [dbar].
        rho, density. [kg m-3].
        up, array with value 1 when data from up profile and 0 for down.
    fixed : array
        Array same size as params with fixed value or None for non-fixed.

    Returns
    -------
    w : array
        Vertical velocity for given parameters and data.

    """

    ppos, p, rho, up = data

    g = -9.8  # Gravitational acceleration [m s-2].

    # Apply fixed value for those parameters that are fixed.
    for i, val in enumerate(fixed):
        if val is not None:
            params[i] = val

    V_0, CA_up, CA_down, alpha_p, p_0, alpha_ppos, ppos_0, M = params

    # Float volume
    V = V_0*(1 - alpha_p*(p - p_0)) + alpha_ppos*(ppos - ppos_0)

    w_up = up*np.sqrt(np.abs(g*(M - rho*V))/(rho*CA_up))
    w_down = (1 - up)*np.sqrt(np.abs(g*(M - rho*V))/(rho*CA_down))

    return np.sign(rho*V - M)*(w_up + w_down)


def still_water_model_2(params, data, fixed):
    """Currently cannot fix parameters."""
    a, b, c, d = params
    ppos, p, rho = data

    w_sqrd = a + b*ppos + c*p + d/rho

    # Not sure about this condition...
    is_going_down = w_sqrd < 0.

    w = np.sqrt(np.abs(w_sqrd))
    w[is_going_down] = -1.*w[is_going_down]

    return w

#    Wref(i) = sqrt(x(i,4)/dens0);
#    alpha(i) = -x(i,2)*dens0/x(i,4);
#    kappa(i) = x(i,3)*dens0/x(i,4);
#    k0(i) = -x(i,1)/x(i,2) - x(i,4)/(x(i,2)*dens0);
#


def cost(params, fixed, model, wf, data, cf_key='diffsq'):
    """The cost function should be minimised when the model parameters are
    optimised.

    Parameters
    ----------
    params : 1-D numpy.ndarray.
        The profile ID numbers at for which to construct grid.
    model : function.
        A model of the float in still water.
    wf : numpy.ndarray
        The float absolute velocity.
    model_args : tuple.
        Additional arguments to model, (model(params, *model_args)).
    cf_key : string.
        Key to select cost function, either 'sqdiff' or 'diff_square'

    Returns
    -------
    c : numpy.ndarray
        The cost calculated from cost_func.

    Notes
    -----
    Uses the Profile.interp function.


    """

    ws = model(params, data, fixed)

    if cf_key == 'sqdiff':
        c = ws**2 - wf**2
    elif cf_key == 'diffsq':
        c = (ws - wf)**2
    else:
        raise ValueError('Incorrect cf_key')

    return c
