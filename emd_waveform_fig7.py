#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

#%% -----------------------------------------------------
#
# This script loads a single run of the analysed LFP data and plots up eight
# example cycles with their respective phase-aligned instantaneous frequency
# and normalised waveforms.

#%% -----------------------------------------------------
# Imports and definitions

import os
import emd
import h5py
import pandas
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from emd_waveform_utils import config


def decorate(ax, mode='timex', bottom_row=True):
    for tag in ['top', 'right']:
        ax.spines[tag].set_visible(False)
    if mode == 'phasex':
        if bottom_row:
            xlabels = ['-pi', '-pi/2', '0', 'pi/2', 'pi']
            ax.set_xlabel('Theta Phase (rads)')
        else:
            xlabels = []
        ax.set_xticks(np.linspace(0, 48, 5))
        ax.set_xticklabels(xlabels)
    elif mode == 'timex':
        if bottom_row:
            xlabels = np.arange(5)*50
            ax.set_xlabel('Time (samples)')
        else:
            xlabels = []
        ax.set_xticks(np.arange(5)*50)
        ax.set_xticklabels(xlabels)
    elif mode == 'normx':
        if bottom_row:
            xlabels = np.linspace(0, 1, 3)
            ax.set_xlabel('Proportion of sinusoid')
        else:
            xlabels = []
        ax.set_xticks(np.linspace(0, 48, 3))
        ax.set_xticklabels(xlabels)


def shift_ax(ax, shift):
    pos = list(ax.get_position().bounds)
    pos[0] = pos[0] + shift
    ax.set_position(pos)


#%% ------------------------------------------------------
# Load data

run = 2
run_name = config['recordings'][2]

datafile = os.path.join(config['analysisdir'], run_name + '.hdf5')
F = h5py.File(datafile, 'r')

imf = F['imf'][...]
C = emd.cycles.Cycles(F['IP'][:, 5])

metricfile = os.path.join(config['analysisdir'], run_name + '.csv')
df = pandas.read_csv(metricfile)


#%% ------------------------------------------------------
# Make plot

width = config['3col_width'] / 25.4
height = width

cycle_inds = [50, 134, 445, 897, 1103, 458, 23, 999]

cols = cm.Dark2(np.linspace(0, 1, 8))
lw = 3

plt.figure(figsize=(width*2, height*2))
plt.subplots_adjust(hspace=0.3, wspace=0.4, top=0.95, bottom=0.07, right=.975, left=.085)
for ii in range(8):
    ind = np.floor(ii/2)*6 + 1
    if ii % 2 == 1:
        ind = ind + 3
        xshift = 0.01
    else:
        xshift = -0.03

    ax1 = plt.subplot(4, 6, ind)
    ax2 = plt.subplot(4, 6, ind+1)
    ax3 = plt.subplot(4, 6, ind+2)

    if ii % 2 == 1:
        shift_ax(ax1, xshift)
        shift_ax(ax2, xshift)
        shift_ax(ax3, xshift)
    else:
        shift_ax(ax3, xshift)
        shift_ax(ax2, xshift)
        shift_ax(ax1, xshift)

    start = df['start_sample'][cycle_inds[ii]]
    stop = start + df['duration_samples'][cycle_inds[ii]]
    cycle_slice = slice(start, stop)
    ax1.plot(imf[cycle_slice, :].sum(axis=1), color=[.8, .8, .8])
    ax1.plot(F['zc_waveform'][:, cycle_inds[ii]], linewidth=lw, color=cols[ii, :])
    ax1.set_xlim(0, 200)
    ax1.set_ylabel(r'Amplitude ($\mu$V)')
    decorate(ax1, bottom_row=(ii in [6, 7]), mode='timex')
    if ii == 0 or ii == 1:
        ax1.set_title('Theta cycle')

    ax2.plot(F['pa'][:, cycle_inds[ii]], linewidth=lw, color=cols[ii, :])
    ax2.set_ylim(3.5, 12.5)
    ax2.set_ylabel('Instantaneous Frequency (Hz)')
    decorate(ax2, bottom_row=(ii in [6, 7]), mode='phasex')
    if ii == 0 or ii == 1:
        ax2.set_title('Phase-aligned IF')

    pa = F['pa'][:, cycle_inds[ii]]
    sr = pa.mean() * 49
    phase = emd.spectra.phase_from_freq(pa, sr, phase_start=0)
    phase = np.r_[0, phase]
    ax3.plot(np.sin(np.linspace(0, 2*np.pi, 49)), 'k:')
    ax3.plot(np.sin(phase), linewidth=lw, color=cols[ii, :])
    decorate(ax3, bottom_row=(ii in [6, 7]), mode='normx')
    ax3.set_ylim(-1.66, 1.66)
    ax3.set_yticks(np.linspace(-1, 1, 3))
    ax3.spines['left'].set_bounds(-1, 1)
    ax3.set_ylabel(r'Amplitude (norm)')
    if ii == 0 or ii == 1:
        ax3.set_title('Normalised Waveform')

outname = os.path.join(config['figdir'], 'emd_fig7_singlecycles.png')
plt.savefig(outname, dpi=300, transparent=True)
