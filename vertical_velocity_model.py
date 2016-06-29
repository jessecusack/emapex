# -*- coding: utf-8 -*-
"""
Created on Tue Apr 08 14:33:17 2014

@author: jc3e13
"""

import numpy as np


def still_water_model_1(params, data, fixed):
    """Calculates and returns the vertical velocity that the float would have
    if it were moving in still water.

    params:

    0: V_0 = 1.  # Float volume when neutrally buoyant [m3].
    1: CA = 1.  # Combination of drag and surface area [m2].
    2: alpha_p = 3.76e-6  # Coeff. of expansion with pressure [-].
    3: p_0 = 2000.  # Pressure when float is neutrally buoyant [dbar].
    4: alpha_ppos = 1.156e-6  # Coeff. of expansion with piston position [m3].
    5: ppos_0 = 16.  # Piston position when neutrally buoyant [-].
    6: M = 27.179  # Float mass [kg].

    data:

    ppos, p, rho

    fixed:

    List of values to fix with the same numbering as parameters. Use None for
    varying parameters.

    Gravity is given a value of 9.8 m s-2.

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
