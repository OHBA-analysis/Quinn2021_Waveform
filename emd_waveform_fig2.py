#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

#%% -----------------------------------------------------
#
# This generates the cartoon cycles seen in figure 2 before computing their
# instantaneous frequency metrics andcontrol point durations. Figure 2 is then
# createed.


#%% -----------------------------------------------------
# Imports and definitions

import os
import emd
import numpy as np

from emd_waveform_utils import config

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', serif=config['fontname'])


#%% ----------------------------------------------------
# Generate some noiseless oscillations with different shapes

samples_per_cycle = 480
ncycles = 6
ph = np.linspace(0, np.pi*2*ncycles, samples_per_cycle*ncycles)
t = np.linspace(0, ncycles, samples_per_cycle*ncycles)
basis = np.c_[np.zeros_like(ph),
              0.9*np.cos(2*np.pi*1*t)[:, None],
              -0.9*np.cos(2*np.pi*1*t)[:, None],
              1.55*np.sin(2*np.pi*1*t)[:, None],
              -1.55*np.sin(2*np.pi*1*t)[:, None],
              np.sin(2*np.pi*2*t)[:, None],
              -0.8*np.sin(2*np.pi*2*t)[:, None]]
basis = basis * 1/4

phs = ph[:, None] + basis

X = np.sin(phs)

#%% ----------------------------------------------------
# Compute instantaneous frequency and control point shape representations.

IP, IF, IA = emd.spectra.frequency_transform(X, samples_per_cycle, 'hilbert')

cycles = emd.cycles.get_cycle_vector(IP, return_good=True)

pa = []
for ii in range(X.shape[1]):
    aligned, phase_template = emd.cycles.phase_align(IP[:, ii], IF[:, ii],
                                                     cycles=cycles[:, ii],
                                                     npoints=samples_per_cycle)
    pa.append(aligned)

ctrl = []
ctrl_mets = []
for ii in range(X.shape[1]):
    ctrl.append(emd.cycles.get_control_points(X[:, ii], cycles[:, ii], interp=True))
    ctrl_mets.append(emd.cycles.get_control_point_metrics(ctrl[ii]))

#%% ----------------------------------------------------
# Housekeepingfor figure

width = config['3col_width'] / 25.4
height = width * .8

cols = plt.cm.Set1(np.linspace(0, 1, basis.shape[1]))
cols = np.r_[np.array([0, 0, 0, 1])[None, :], cols]
column = np.array([0, 1, 1, 2, 2, 3, 3])
column2 = np.array([1, 2, 2, 2, 2, 2, 2])
column3 = np.array([0, 1, 2, 1, 2, 1, 2])
col_sep = 700
xt = np.sort(np.r_[np.arange(4)*col_sep, np.arange(4)*col_sep+samples_per_cycle])
xt_seconds = ['0', '1', '0', '1', '0', '1', '0', '1']
xt_phase= ['0', r'2$\pi$', '0', r'2$\pi$', '0', r'2$\pi$', '0', r'2$\pi$']
titles = ['Sinusoid', 'Extrema Asymmetry', 'Edge Asymmetry', 'Extrema Curvature']

#%% ----------------------------------------------------
# Create summary figure


plt.figure(figsize=(width*2, height*2))

# Plot canonical shapes on top row
plt.subplot(411, frameon=False)
plt.xticks(xt, xt_seconds)
plt.yticks([])
lin_inds = cycles[:, 0] == 1
for ii in range(basis.shape[1]):
    inds = cycles[:, ii] == 2
    xx = np.arange(samples_per_cycle)+col_sep*column[ii]
    plt.plot(xx, np.sin(phs[lin_inds, 0]), 'k:')
    plt.plot(xx, np.sin(phs[inds, ii]), color=cols[ii, :], linewidth=2)
    plt.plot(xx, -np.ones_like(np.arange(samples_per_cycle)), 'k', linewidth=.5)
    plt.text(xx[0]+samples_per_cycle/2, 1.2, titles[column[ii]], horizontalalignment='center', fontsize=14)
    #if ii == 0:
    plt.text(xx[0]+samples_per_cycle/2, -1.4, 'Time (s)', horizontalalignment='center', fontsize=11)
    plt.ylabel('Example Waveforms')

# Instantaneous phases on row-2
plt.subplot(412)
lin_inds = cycles[:, 0] == 1
for tag in ['top', 'right', 'bottom']:
    plt.gca().spines[tag].set_visible(False)
plt.xticks(xt, xt_seconds)
plt.ylabel('Phase (radians)')
plt.yticks([0, np.pi, 2*np.pi], ['0', '$\\pi$', '2$\\pi$'])
for ii in range(basis.shape[1]):
    inds = cycles[:, ii] == 2
    lp = phs[lin_inds, 0]
    nlp = phs[inds, ii]
    xx = np.arange(samples_per_cycle)+col_sep*column[ii]
    plt.plot(xx, lp-lp[0], 'k:')
    plt.plot(xx, nlp-nlp[0], color=cols[ii, :], linewidth=2)
    plt.plot(xx, np.zeros_like(np.arange(samples_per_cycle)), 'k', linewidth=.5)
    #if ii == 0:
    plt.text(xx[0]+samples_per_cycle/2, -1.4, 'Time (s)',
             horizontalalignment='center', fontsize=11, verticalalignment='bottom')

# Instantaneous frequencies in row-3
plt.subplot(413)
for tag in ['top', 'right', 'bottom']:
    plt.gca().spines[tag].set_visible(False)
lin_inds = cycles[:, 0] == 1
plt.xticks(xt, xt_phase)
plt.ylabel('Instantaneous\nFrequency (Hz)')
for ii in range(basis.shape[1]):
    inds = cycles[:, ii] == 2
    xx = np.arange(samples_per_cycle)+col_sep*column[ii]
    plt.plot(xx, pa[0], 'k:')
    plt.plot(xx, pa[ii], color=cols[ii, :], linewidth=2)
    plt.plot(xx, 0.75*np.ones_like(np.arange(samples_per_cycle)), 'k', linewidth=.5)
    #if ii == 0:
    plt.text(xx[0]+samples_per_cycle/2, 0.65, 'Phase (rads)',
             horizontalalignment='center', fontsize=11, verticalalignment='bottom')

# Control points on row-4
plt.subplot(414)
for tag in ['top', 'right']:
    plt.gca().spines[tag].set_visible(False)
xtt = np.tile(np.arange(4, samples_per_cycle, samples_per_cycle/4)+60, 4) + np.repeat(np.arange(4)*col_sep, 4)
xtl = np.tile(['Peak', 'Trough', 'Ascent', 'Descent'], 4)
plt.xticks(xtt, xtl, rotation=45)
plt.ylabel('Proportion of\ncycle')
plt.plot([0, xt[-1]], [.5, .5], 'k:')
for ii in range(basis.shape[1]):
    xinds = np.arange(4, samples_per_cycle, samples_per_cycle/8)
    width = samples_per_cycle*.21/column2[ii]
    c = ctrl[ii][0, :]
    if column3[ii] == 1:
        col_step = 0
    else:
        col_step = 1
    if column2[ii] == 2:
        xinds += 30

    pk_time = c[2] / samples_per_cycle
    plt.bar(xinds[0+col_step] + col_sep*column[ii], pk_time, width=width, color=cols[ii, :])
    tr_time = (c[4]-c[2]) / samples_per_cycle
    plt.bar(xinds[2+col_step] + col_sep*column[ii], tr_time, width=width, color=cols[ii, :])
    asc_time = (c[3]-c[1]) / samples_per_cycle
    plt.bar(xinds[4+col_step] + col_sep*column[ii], asc_time, width=width, color=cols[ii, :])
    dsc_time = (c[4]-c[3]+c[1]) / samples_per_cycle
    plt.bar(xinds[6+col_step] + col_sep*column[ii], dsc_time, width=width, color=cols[ii, :])

outname = os.path.join(config['figdir'], 'emd_fig2_instfreq_shape.png')
plt.savefig(outname, dpi=300, transparent=True)
