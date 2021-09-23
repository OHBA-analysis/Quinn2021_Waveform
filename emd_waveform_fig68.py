#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

# %% -----------------------------------------------------
#
# This script loads the EMD analyses from one run of the LFP data and creates
# figures 6 and 8. Figure 6 shows a segment of the time-series and associated
# EMD metrics and figure 8 shows the single cycle representation of around 2000
# cycles.

# %% -----------------------------------------------------
# Imports and definitions

import os
import emd
import h5py
import sails
import pandas
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from emd_waveform_utils import config

import matplotlib
matplotlib.rc('font', serif=config['fontname'])

# %% ------------------------------------------------------

emd.logger.set_up(level='DEBUG')

run = 2
run_name = config['recordings'][2]

datafile = os.path.join(config['analysisdir'], run_name + '.hdf5')

F = h5py.File(datafile, 'r')
sample_rate = 1250
imf = F['imf'][...]
IP = F['IP'][...]
IA = F['IA'][...]
IF = F['IF'][...]
speed = F['speed'][...]

metricfile = os.path.join(config['analysisdir'], run_name + '.csv')
df = pandas.read_csv(metricfile)

# Carrier frequency histogram definition
edges, bins = emd.spectra.define_hist_bins(2, 128, 128, 'log')

plot_inds = np.arange(7500+1250, 7500+1250+4*1250)

# %% ------------------------------------------
# Create graphical abstract


TINY_SIZE = 6
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=TINY_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=TINY_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

frames = True

def remove_frames(ax, tags=['top', 'right']):
    for tag in tags:
        ax.spines[tag].set_visible(False)

start = 41000
inds = np.arange(start, start+1250*1).astype(int)
tt = np.linspace(0, 1, len(inds))

plt.figure(figsize=(14, 10))
ax1 = plt.axes([0.05, .775, .125, .1], frameon=frames)
ax2 = plt.axes([.308, .725, .125, .2], frameon=frames)
ax3 = plt.axes([.5666, .725, .125, .2], frameon=frames)
ax4 = plt.axes([.825, .725, .125, .2], frameon=frames)
ax5 = plt.axes([.06, .35, .2, .125], frameon=frames)

ax1.plot(tt, imf[inds, :].sum(axis=1), 'k')
ax1.plot(tt, np.zeros_like(tt)-500, 'k', linewidth=0.5)
remove_frames(ax1, tags=['top', 'right', 'bottom'])
ax1.set_xlim(tt[0], tt[-1])
ax1.set_xticks([0, 0.5, 1])
ax1.set_xlabel('Time (secs)')
ax1.set_ylabel(r'Amp ($\mu V$)')
ax1.spines['left'].set_bounds(-500, 500)
ax1.set_yticks([-500, 0, 500])

remove_frames(ax2, tags=['top', 'right', 'bottom', 'left'])
ax2.set_xlim(tt[0], tt[-1])
ax2.set_xticks([0, 0.5, 1])
for ii in range(4):
    ax2.plot(tt, np.zeros_like(tt)-ii*500, 'k', linewidth=0.5)
    ax2.plot((0, 0), (-200-ii*500, 200-ii*500), 'k')
    ax2.text(-.015, 200-ii*500, '200', va='center', ha='right', fontsize=TINY_SIZE)
    ax2.text(-.015, -200-ii*500, '-200', va='center', ha='right', fontsize=TINY_SIZE)
ax2.set_yticks([])
ax2.plot(tt, imf[inds, 2:6] - np.arange(0, 2000, 500)[None, :])
ax2.set_ylabel(r'Amp ($\mu V$)', labelpad=20)
ax2.set_xlabel('Time (secs)')

ip = IP[inds, 5]
ip[np.gradient(ip) < -2] =  np.nan
remove_frames(ax3, tags=['top', 'right', 'left'])
ax3.set_yticks([])
ax3.plot(tt, ip)
ax3.set_xlim(tt[0], tt[-1])
ax3.set_xticks([0, 0.5, 1])
ax3.set_xlabel('Time (secs)')
ax3.plot(tt, IF[inds, 5]-14)
ax3.plot((0, 0), (0, np.pi*2), 'k')
ax3.plot((0, 0), (4-14, 10-14), 'k')
ax3.text(-.015, np.pi*2, r'2$\pi$', va='center', ha='right', fontsize=TINY_SIZE)
ax3.text(-.015, 0, r'0', va='center', ha='right', fontsize=TINY_SIZE)
ax3.text(-.015, 10-14, '10', va='center', ha='right', fontsize=TINY_SIZE)
ax3.text(-.015, 7-14, '7', va='center', ha='right', fontsize=TINY_SIZE)
ax3.text(-.015, 4-14, '4', va='center', ha='right', fontsize=TINY_SIZE)
ax3.text(-.1, 7-14, 'Instantaneous\nFrequency (Hz)', va='center', ha='right', fontsize=SMALL_SIZE, rotation=90)
ax3.text(-.1, np.pi, 'Instantaneous\nPhase (rads)', va='center', ha='right', fontsize=SMALL_SIZE, rotation=90)

inds = np.arange(start, start+1250*4).astype(int)
tt = np.linspace(0, 4, len(inds))
ax4.fill_between(tt, speed[inds], 0, alpha=0.5)
ax4.plot((tt[0], tt[-1]), (2, 2), 'k--')

ii = imf[inds, 5]/100 - 3.5
ax4.plot(tt, ii, 'k')
ii[speed[inds] > 2] = np.nan
ax4.plot(tt, ii, 'r')
ax4.set_xlabel('Time (secs)')
ax4.set_xlim(tt[0], tt[-1])
ax4.set_xticks([0, 1, 2, 3, 4])
ax4.set_yticks([])
remove_frames(ax4, tags=['top', 'right', 'left'])
ax4.plot((0, 0), (0, 5), 'k')
ax4.plot((0, 0), (-5.5, -1.5), 'k')
ax4.text(-.03, 0, '0', va='center', ha='right', fontsize=TINY_SIZE)
ax4.text(-.03, 2, '2', va='center', ha='right', fontsize=TINY_SIZE)
ax4.text(-.03, 4, '4', va='center', ha='right', fontsize=TINY_SIZE)
ax4.text(-.015, -1.5, '200', va='center', ha='right', fontsize=TINY_SIZE)
ax4.text(-.015, -3.5, '0', va='center', ha='right', fontsize=TINY_SIZE)
ax4.text(-.015, -5.5, '-200', va='center', ha='right', fontsize=TINY_SIZE)
ax4.text(-.4, -3.5, 'Amp. ($\mu$V)', va='center', ha='right', fontsize=SMALL_SIZE, rotation=90)
ax4.text(-.4, 2.5, 'Movement\nSpeed (cm/s)', va='center', ha='right', fontsize=SMALL_SIZE, rotation=90)

start = 41000
inds = np.arange(start, start+1250*1).astype(int)
tt = np.linspace(0, 4, len(inds))
C = emd.cycles.Cycles(IP[inds, 5], compute_timings=True)
C.compute_cycle_metric('peak', imf[inds, 5], emd.cycles.cf_peak_sample)
C.compute_cycle_metric('desc', imf[inds, 5], emd.cycles.cf_descending_zero_sample)
C.compute_cycle_metric('trough', imf[inds, 5], emd.cycles.cf_trough_sample)
df_abs = C.get_metric_dataframe()

ax5.plot(imf[inds, 5], 'k')
for ii in range(1, len(df_abs)-1):
    st = df_abs['start_sample'].values[ii]
    pk = st +  df_abs['peak'].values[ii]
    ax5.plot(pk, imf[inds[int(pk)], 5], '^r')
    tr = st +  df_abs['trough'].values[ii]
    ax5.plot(tr, imf[inds[int(tr)], 5], 'vb')
    asc = st +  df_abs['desc'].values[ii]
    ax5.plot(asc, imf[inds[int(asc)], 5], 'oc')
    desc = st
    ax5.plot(desc, imf[inds[int(desc)], 5], 'om')
    if ii == 1:
        plt.legend(['Oscillation', 'Peak', 'Trough', 'Descending Zero', 'Ascending Zero'], frameon=False, bbox_to_anchor=(0.5, -1), loc='center')
remove_frames(ax5, tags=['top', 'right'])
ax5.set_xlim(tt[0], tt[-1])
ax5.set_xticks(np.linspace(0, len(tt), 5))
ax5.set_xticklabels(np.arange(5))
ax5.set_xlabel('Time (secs)')
ax5.set_ylabel(r'Amp ($\mu V$)')
ax5.spines['left'].set_bounds(-300, 300)

ax6 = plt.axes([0.35, 0.42, 0.1, 0.1])
ax7 = plt.axes([0.35, 0.2, 0.1, 0.2])
ax8 = plt.axes([0.495, 0.42, 0.1, 0.1])
ax9 = plt.axes([0.495, 0.2, 0.1, 0.2])

pa = emd.cycles.phase_align(IP[inds, 5], IF[inds, 5], cycles=C)
cind = (3, 7)
ax6.plot(imf[inds[C._slice_cache[cind[0]]], 5], 'r')
ax6.plot(imf[inds[C._slice_cache[cind[1]]], 5], 'b')
remove_frames(ax6, tags=['top', 'right', 'bottom'])
ax6.set_ylabel(r'Amp ($\mu V$)')
ax6.set_xticks([])
ax6.spines['left'].set_bounds(-200, 200)

ax7.plot(IF[inds[C._slice_cache[cind[0]]], 5], 'r')
ax7.plot(IF[inds[C._slice_cache[cind[1]]], 5], 'b')
remove_frames(ax7, tags=['top', 'right'])
ax7.set_xlabel('Time (secs)')
ax7.set_ylabel('Instantaneous\nFrequency (Hz)', rotation=90, fontsize=SMALL_SIZE)

ax8.plot(np.sin(2*np.pi*np.linspace(0, 1)), 'r')
ax8.plot(np.sin(2*np.pi*np.linspace(0, 1)), 'b--')
remove_frames(ax8, tags=['top', 'right', 'bottom'])
ax8.set_ylabel(r'Amp (a.u.)')
ax8.set_xticks([])
ax8.spines['left'].set_bounds(-1, 1)

ax9.plot(pa[0][:, cind[0]], 'r')
ax9.plot(pa[0][:, cind[1]], 'b')
remove_frames(ax9, tags=['top', 'right'])
ax9.set_xlabel('Phase (rads)')
ax9.set_xticks(np.linspace(0, 48, 3))
ax9.set_xticklabels(['0', r'$\pi$', r'2$\pi$'])

inds = np.arange(start, start+1250*12).astype(int)
C = emd.cycles.Cycles(IP[inds, 5], compute_timings=True)
pa, _ = emd.cycles.phase_align(IP[inds, 5], IF[inds, 5], cycles=C)
pa = pa[:, np.isfinite(pa.mean(axis=0))]
goods = np.logical_and((pa.min(axis=0) > 3), (pa.mean(axis=0) <10))

ax10 = plt.axes([0.675, 0.25, .1, .25])
im = ax10.pcolormesh(pa[:, goods].T, vmin=5, vmax=12)
cb = plt.colorbar(im)
cb.set_label('Instantaneous\nFrequency (Hz)')
ax10.set_xlabel('Phase (rads)')
ax10.set_xticks(np.linspace(0, 48, 3))
ax10.set_xticklabels(['0', r'$\pi$', r'2$\pi$'])
ax10.set_ylabel('Cycles')

ax11 = plt.axes([0.9, 0.425, 0.093, 0.12])
ax12 = plt.axes([0.9, 0.25, 0.093, 0.12])
ax13 = plt.axes([0.9, 0.075, 0.093, 0.12])

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
IP2, IF2, IA2 = emd.spectra.frequency_transform(X, samples_per_cycle, 'hilbert')

cycles = emd.cycles.get_cycle_vector(IP2, return_good=True)

lin_inds = cycles[:, 0] == 1
inds = cycles[:, 1] == 2
ax11.plot(np.linspace(0, 1, inds.sum()), np.sin(phs[inds, 1]))
inds = cycles[:, 2] == 2
ax11.plot(np.linspace(0, 1, inds.sum()), np.sin(phs[inds, 2]))
remove_frames(ax11, tags=['top', 'right'])
ax11.set_yticks([-1, 0, 1])
ax11.set_ylabel('Amp (a.u.)')
ax11.set_xlim(0, 1)
ax11.set_xticks([0, 1])
ax11.set_title('Motif 1', fontsize=MEDIUM_SIZE)
ax11.spines['left'].set_bounds(-1, 1)

inds = cycles[:, 3] == 2
ax12.plot(np.linspace(0, 1, inds.sum()), np.sin(phs[inds, 3]))
inds = cycles[:, 4] == 2
ax12.plot(np.linspace(0, 1, inds.sum()), np.sin(phs[inds, 4]))
remove_frames(ax12, tags=['top', 'right'])
ax12.set_yticks([-1, 0, 1])
ax12.set_xlim(0, 1)
ax12.set_ylabel('Amp (a.u.)')
ax12.set_xticks([0, 1])
ax12.set_title('Motif 2', fontsize=MEDIUM_SIZE)
ax12.spines['left'].set_bounds(-1, 1)

inds = cycles[:, 5] == 2
ax13.plot(np.linspace(0, 1, inds.sum()), np.sin(phs[inds, 5]))
inds = cycles[:, 6] == 2
ax13.plot(np.linspace(0, 1, inds.sum()), np.sin(phs[inds, 6]))
remove_frames(ax13, tags=['top', 'right'])
ax13.set_xlabel('Cycle Duration', fontsize=SMALL_SIZE)
ax13.set_yticks([-1, 0, 1])
ax13.set_ylabel('Amp (a.u.)')
ax13.set_xlim(0, 1)
ax13.set_xticks([0, 1])
ax13.set_title('Motif 3', fontsize=MEDIUM_SIZE)
ax13.spines['left'].set_bounds(-1, 1)

outname = os.path.join(config['figdir'], 'emd_fig1_graphicalabstract.png')
plt.savefig(outname, dpi=300, transparent=True)

plt.style.use('default')


# %% ------------------------------------------
# Create figure 5 time-series

width = config['3col_width'] / 25.4
height = width * .6

plot_horiz = True
sparse_horiz = True
plot_vert = True
fontsize_side = 'large'
fontsize_tick = 10
horiz_width = .35

inds = np.arange(20230, 20000+1250*3).astype(int)
start = 193000
start = 41000
inds = np.arange(start, start+1250*2.8).astype(int)
cmap = plt.cm.Set1
cols = cmap(np.linspace(0, 1, 8))
cols[4, :] = [.5, .5, .2, 1]
indx = [5, 1, 2, 3, 4, 0, 6, 7]
cols = cols[indx, :]

plt.figure(figsize=(width*2, height*2))
plt.axes([.08, .025, .95, .95], frameon=False)
plt.xticks([])
plt.yticks([])

# Plot Data
plt.plot(imf[inds, :6].sum(axis=1), color=[.2, .2, .2], linewidth=.5)
plt.plot(imf[inds, 5], color=cols[5, :], linewidth=1)
plt.plot([0, 0], [-350, 350], 'k')
plt.text(-250, 80, 'LFP', fontsize=fontsize_side,
         verticalalignment='center', horizontalalignment='center')
plt.text(-250, 600, 'Cycle No', fontsize=fontsize_side,
         verticalalignment='center', horizontalalignment='center')
plt.text(-250, -80, 'Theta', fontsize=fontsize_side,
         verticalalignment='center', horizontalalignment='center', color='r')

plt.plot([1.9*1250, 2.9*1250], [800, 800], 'k')
plt.text(2.4*1250, 825, '1 Second', horizontalalignment='center',
         verticalalignment='bottom', fontsize=fontsize_side)

# Plot IMFs
step = -500
labels = ['IMF1', 'IMF2', 'IMF3', 'IMF4', 'IMF5', 'IMF6', 'IMF7+']
for ii in range(7):
    yind = -300*(1+ii)+step
    if plot_horiz:
        plt.plot([-10, len(inds)], [yind, yind], color=[.7, .7, .7], linewidth=horiz_width)
    plt.plot([-10, 0], [yind, yind], 'k')
    if ii < 6:
        plt.plot(.5*imf[inds, ii]+yind, color=cols[ii, :])
    else:
        plt.plot(.5*imf[inds, ii:].sum(axis=1)+yind, color=cols[ii, :])
    plt.text(-22, yind, labels[ii], fontsize=fontsize_tick, verticalalignment='center', horizontalalignment='right')
plt.plot([0, 0], [-2800, -600], 'k')
plt.text(-275, -300*(1+3)+step, 'IMFs', fontsize=fontsize_side, verticalalignment='center', horizontalalignment='center')

# Instantaneous Phase
labels = [r'$-\pi$', r'$0$', r'$\pi$']
for ii in range(3):
    yind = -3500+ii*75*((2*np.pi)/2)
    if sparse_horiz and ii == 1:
        plt.plot([-10, len(inds)], [yind, yind], color=[.7, .7, .7], linewidth=horiz_width)
    elif plot_horiz and not sparse_horiz:
        plt.plot([-10, len(inds)], [yind, yind], color=[.7, .7, .7], linewidth=horiz_width)
    plt.plot([-10, 0], [yind, yind], color='k')
    plt.text(-22, yind, labels[ii], fontsize=fontsize_tick, verticalalignment='center', horizontalalignment='right')
plt.plot([0, 0], [-3500, -3500+2*np.pi*75], 'k')
ip = IP[inds, 5]
naninds = np.where(np.diff(ip) < -5.5)[0]+1
ip[naninds] = np.nan
plt.plot(ip*75 - 3500, linewidth=1.5)
plt.text(-300, -3500+1*75*((2*np.pi)/2), 'Instantaneous\nPhase (rads)', fontsize=fontsize_side,
         verticalalignment='center', horizontalalignment='center')

# Instantaneous Frequency
if_to_plot = IF[inds, 5]
ymin_f = np.nanmin(np.round(if_to_plot))
ymin = np.nanmin(ymin_f*40 - 4200)
ymax_f = np.nanmax(np.round(if_to_plot))
ymax = np.nanmin(ymax_f*40 - 4200)
plt.plot([0, 0], [ymin, ymax], 'k')
indx = np.linspace(ymin, ymax, 3)
indx_f = np.linspace(ymin_f, ymax_f, 3)
for ii in range(3):
    if sparse_horiz and ii == 1:
        plt.plot([-10, len(inds)], [indx[ii], indx[ii]], color=[.7, .7, .7], linewidth=horiz_width)
    elif plot_horiz and not sparse_horiz:
        plt.plot([-10, len(inds)], [indx[ii], indx[ii]], color=[.7, .7, .7], linewidth=horiz_width)
    plt.plot([-10, 0], [indx[ii], indx[ii]], color='k')
    plt.text(-22, indx[ii], indx_f[ii], fontsize=fontsize_tick, verticalalignment='center', horizontalalignment='right')
plt.plot(if_to_plot*40 - 4200)
plt.text(-300, indx[1], 'Instantaneous\nFrequency (Hz)', fontsize=fontsize_side, verticalalignment='center', horizontalalignment='center')

# Plot cycle bounds and compute within cycle frequency variability
cycles_to_plot = emd.cycles.get_cycle_vector(IP[inds, 5, None])
cycle_starts = np.where(np.diff(cycles_to_plot, axis=0))[0]
cm = np.zeros_like(inds)*np.nan
cv = np.zeros_like(inds)*np.nan
for ii in range(len(cycle_starts)):
    if plot_vert:
        plt.plot((cycle_starts[ii], cycle_starts[ii]), (-4600, 350), color=[.8, .8, .8], linewidth=.5)
    if ii < len(cycle_starts)-1:
        cm[cycle_starts[ii]:cycle_starts[ii+1]] = IF[inds[cycle_starts[ii]:cycle_starts[ii+1]], 5].mean()
        cv[cycle_starts[ii]:cycle_starts[ii+1]] = IF[inds[cycle_starts[ii]:cycle_starts[ii+1]], 5].std()
        plt.text((cycle_starts[ii]+cycle_starts[ii+1])/2, 600, ii+1,
                      fontsize=fontsize_tick, verticalalignment='center', horizontalalignment='center')

# Within cycle frequency variability
plt.fill_between(np.arange(len(inds)), cv*1e2 - 4600, np.ones_like(inds)-4601)
plt.plot((0, 0), (-4601, -4601+300), 'k')
plt.plot([-15, len(inds)], (-4601, -4601), color=[.7, .7, .7], linewidth=.5)
indx = np.linspace(0, 3, 4)*1e2 - 4600
indx_lab = np.round(np.linspace(0, 3, 4), 2).astype(int)
for ii in range(4):
    if plot_horiz and sparse_horiz is False:
        plt.plot([-10, len(inds)], (indx[ii], indx[ii]), color=[.7, .7, .7], linewidth=horiz_width)
    elif  ii == 0:
        plt.plot([-10, len(inds)], (indx[ii], indx[ii]), color=[.7, .7, .7], linewidth=horiz_width)
    plt.plot((-10, 0), (-4601+100*ii, -4601+100*ii), 'k')
    plt.text(-22, indx[ii], indx_lab[ii], fontsize=fontsize_tick,
             verticalalignment='center', horizontalalignment='right')

plt.text(-300, indx[1:3].mean(), 'Instantaneous\nFrequency\nStd-Dev', fontsize=fontsize_side,
         verticalalignment='center', horizontalalignment='center')

outname = os.path.join(config['figdir'], 'emd_fig6_real_sift.png')
plt.savefig(outname, dpi=300, transparent=True)

# %% --------------------------------------------------------------------
# Create figure 5 - Supplemental
inds2 = inds[:600]

tx = np.linspace(0, 2, 512)

plt.figure(figsize=(14, 10))
plt.subplots_adjust(hspace=0.3)
# Harmonic
plt.subplot(221)
a = np.sin(2*np.pi*tx)
b = np.sin(2*np.pi*2*tx)
plt.plot(tx, a)
plt.plot(tx, b)
plt.plot(tx, a+b-3)
plt.ylim(-5, 3)
plt.legend(['Base Signal', 'High Freq Signal', 'Summed Signal'], frameon=False, fontsize='large')
for tag in ['top', 'right', 'left']:
    plt.gca().spines[tag].set_visible(False)
plt.yticks([])
plt.title('Simulation A')
plt.xlabel('Time (Seconds)')

plt.subplot(222)
b = 0.2*np.sin(2*np.pi*2*tx)
plt.plot(tx, a)
plt.plot(tx, b)
plt.plot(tx, a+b-3)
plt.ylim(-5, 3)
plt.legend(['Base Signal', 'Harmonic', 'Summed Signal'], frameon=False, fontsize='large')
for tag in ['top', 'right', 'left']:
    plt.gca().spines[tag].set_visible(False)
plt.yticks([])
plt.title('Simulation B')
plt.xlabel('Time (Seconds)')

plt.subplot(212)
plt.plot(imf[inds2, :].sum(axis=1), label='Raw Signal')
plt.plot(imf[inds2, 5]-500, label='IMF-6')
plt.plot(imf[inds2, 4]-500, label='IMF-5')
plt.plot(imf[inds2, 4]+imf[inds2, 5]-1000, label='IMF-5 + IMF-6')
plt.legend(frameon=False, fontsize='large')
for tag in ['top', 'right', 'left']:
    plt.gca().spines[tag].set_visible(False)
plt.yticks([])
plt.xticks(np.arange(5)*125, np.arange(5)*100)
plt.xlabel('Time (milliseconds)')
plt.title('Real Data')

outname = os.path.join(config['figdir'], 'emd_fig6_supplemental_zoom.png')
plt.savefig(outname, dpi=300, transparent=True)

# %% --------------------------------------------------------------------
# Create figure 6 - spectra

edges, bins = emd.spectra.define_hist_bins(2, 35, 64, 'linear')

cwt = sails.wavelet.morlet(imf[inds, :6].sum(axis=1), bins, sample_rate, normalise='simple', ret_mode='amplitude')
hht = emd.spectra.hilberthuang(IF[inds, :6], IA[inds, :6], edges, mode='amplitude')
hht = ndimage.gaussian_filter(hht, 1)

t = np.arange(len(inds))

plt.figure(figsize=(width*1.925, height*1.25))

plt.axes([.13, .55, .855, .425], frameon=True)
pcm = plt.pcolormesh(t, bins, hht, cmap='hot_r')
for ii in range(len(cycle_starts)):
    if plot_vert:
        plt.plot((cycle_starts[ii], cycle_starts[ii]), (2, 100), color=[.8, .8, .8], linewidth=.5)
plt.ylim(2, 35)
plt.xticks(np.arange(0, len(inds), sample_rate/2), [])
plt.ylabel('Frequency (Hz)')
for tag in ['top', 'right']:
    plt.gca().spines[tag].set_visible(False)
plt.ylabel('Frequency (Hz)'); plt.xlabel('')
ax = plt.axes([.97, .65, .015, .3])
cb = plt.colorbar(pcm, cax=ax)
ax.yaxis.set_ticks_position('left')
cb.set_label('Power')

plt.axes([.13, .095, .855, .425], frameon=True)
pcm = plt.pcolormesh(t, bins, cwt, cmap='hot_r')
for ii in range(len(cycle_starts)):
    if plot_vert:
        plt.plot((cycle_starts[ii], cycle_starts[ii]), (2, 100), color=[.8, .8, .8], linewidth=.5)
plt.ylim(2, 35)
plt.xticks(np.arange(0, len(inds), sample_rate/2), np.arange(0, len(inds), sample_rate/2)/sample_rate)
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (seconds)')
for tag in ['top', 'right']:
    plt.gca().spines[tag].set_visible(False)
plt.ylabel('Frequency (Hz)'); plt.xlabel('Time (seconds)')
ax = plt.axes([.97, .195, .015, .3])
cb = plt.colorbar(pcm, cax=ax)
ax.yaxis.set_ticks_position('left')
cb.set_label('Power')

outname = os.path.join(config['figdir'], 'emd_fig6_real_sift_spec.png')
plt.savefig(outname, dpi=300, transparent=True)


# %% --------------------------------------------------------------------
# Create Figure 8

def decorate_ax(ax):
    for tag in ['top', 'right']:
        ax.spines[tag].set_visible(False)

waveform = F['zc_waveform'][...]
instfreq = F['zc_instfreq'][...]
pa = F['pa'][...]
ctrl = np.c_[np.zeros_like(df['start_sample']),
             df['peak_sample'],
             df['desc_sample'],
             df['trough_sample'],
             df['duration_samples']]
ctrl_mets = np.c_[df['peak2trough'], df['asc2desc']].T

I = np.argsort(ctrl[:, 4])[::-1]
segments = np.zeros((ctrl.shape[0], 400))*np.nan
for ii in range(ctrl.shape[0]):
    for jj in range(1, ctrl.shape[1]):
        segments[ii, int(np.round(ctrl[ii, jj-1])):int(np.round(ctrl[ii, jj]))] = jj

# Remove cycles with ambiguous peaks
goods = np.setdiff1d(np.arange(segments.shape[0]), np.where(segments[:, 0]==4)[0])
segments = segments[goods, :]
I = np.argsort(ctrl[goods, 4])[::-1]
ctrl_mets = ctrl_mets[:, goods]
pa = pa[:, goods]
instfreq = instfreq[:, goods]
trim = 2700  # Can't see anything if we plot every cycle...
I = I[:-trim]

width = config['2col_width'] / 25.4
height = config['3col_width'] / 25.4

# Figure start
plt.figure(figsize=(width*2, height*2))

# Plot control point segments
plt.axes([.1, .1, .2, .65])
plt.pcolormesh(segments[I, :])
plt.xticks(np.linspace(0, 200, 5), (np.linspace(0, 200, 5)/sample_rate*1000).astype(int))
plt.xlabel('Time (ms)')
plt.xlim(0, 250)
plt.ylabel('# Cycle (Sorted by duration)')
decorate_ax(plt.gca())
plt.axes([.1, .775, .144, .075], frameon=False)
plt.xticks([]);
plt.yticks([])
cols = plt.cm.viridis(np.linspace(0, 1, 4))
for ii in range(4):
    xvals = np.linspace(0, .25)+.25*ii
    plt.plot(xvals, np.sin(2*np.pi*xvals), linewidth=3, color=cols[ii, :])

# Plot control point metrics
plt.axes([.31, .1, .1, .65])
plt.plot(ctrl_mets[0][I], np.arange(len(ctrl_mets[0])-trim), '.')
plt.plot(ctrl_mets[1][I], np.arange(len(ctrl_mets[0])-trim), '.')
plt.plot(np.zeros_like(ctrl_mets[1][I]), np.arange(len(ctrl_mets[0])-trim), 'k', linewidth=.5)
plt.xlim(0, 1)
plt.ylim(0, len(ctrl_mets[0])-trim)
plt.yticks([])
decorate_ax(plt.gca())
plt.gca().spines['left'].set_visible(False)

plt.axes([.31, .775, .1, .15])
plt.hist(ctrl_mets[0][I], np.linspace(-1, 1), alpha=.5)
plt.hist(ctrl_mets[1][I], np.linspace(-1, 1), alpha=.5)
plt.xticks(np.linspace(-.25, .25, 3), [])
plt.legend(['Peak/Trough', 'Ascent/Descent'], frameon=False,
           fontsize=8, loc='center', bbox_to_anchor=(0.5, 0.5, 1, 1))
decorate_ax(plt.gca())
plt.xlim(0, 1)
plt.ylim(0, 800)
plt.title('Control-Point Ratios\n')

# Plot temporally aligned instantaneous frequency
plt.axes([.5, .1, .2, .65])
plt.pcolormesh(instfreq[:, I].T, vmin=6, vmax=14)
decorate_ax(plt.gca())
plt.xticks(np.linspace(0, 200, 5), (np.linspace(0, 200, 5)/sample_rate*1000).astype(int))
plt.xlabel('Time (ms)')
plt.xlim(0, 250)

plt.axes([.5, .775, .2, .15])
plt.plot(np.nanmean(instfreq, axis=1))
decorate_ax(plt.gca())
plt.title('Cycle-Onset Aligned\nInst. Freq')
plt.xticks(np.linspace(0, 200, 5), [])
plt.xlim(0.60)

# Plot phase aligned instantaneous frequency
plt.axes([.75, .1, .2, .65])
pcm = plt.pcolormesh(pa[:, I].T, vmin=6, vmax=14)
plt.xticks(np.arange(5)*12, ['-pi', '-pi/2', '0', 'pi/2', 'pi'])
plt.xlabel('Theta Phase')
plt.yticks(np.arange(8)*200, [])

plt.axes([.75, .775, .2, .15])
plt.plot(np.nanmean(pa, axis=1))
plt.xlim(0, 48)
decorate_ax(plt.gca())
plt.xticks(np.arange(5)*12, [])
plt.title('Phase-Aligned\nInst. Freq')

# Inst. freq colourbar
ax = plt.axes([.685, .45, .015, .18])
cb = plt.colorbar(pcm, cax=ax)
ax.yaxis.set_ticks_position('left')
plt.title('Instantaneous\nFrequency (Hz)', fontsize=9)

outname = os.path.join(config['figdir'], 'emd_fig8_real_phasealign.png')
plt.savefig(outname, dpi=300, transparent=True)

# %% --------------------------------------------------------------------
# Create Figure 8 - REVISED


def decorate_ax(ax):
    for tag in ['top', 'right']:
        ax.spines[tag].set_visible(False)

waveform = F['zc_waveform'][...]
instfreq = F['zc_instfreq'][...]
pa = F['pa'][...]
ctrl = np.c_[np.zeros_like(df['start_sample']),
             df['peak_sample'],
             df['desc_sample'],
             df['trough_sample'],
             df['duration_samples']]
ctrl_mets = np.c_[df['peak2trough'], df['asc2desc']].T

I = np.argsort(ctrl[:, 4])[::-1]
segments = np.zeros((ctrl.shape[0], 400))*np.nan
for ii in range(ctrl.shape[0]):
    for jj in range(1, ctrl.shape[1]):
        segments[ii, int(np.round(ctrl[ii, jj-1])):int(np.round(ctrl[ii, jj]))] = jj

# Remove cycles with ambiguous peaks
goods = np.setdiff1d(np.arange(segments.shape[0]), np.where(segments[:, 0]==4)[0])
segments = segments[goods, :]
I = np.argsort(ctrl[goods, 4])[::-1]
ctrl_mets = ctrl_mets[:, goods]
pa = pa[:, goods]
instfreq = instfreq[:, goods]
trim = 2700  # Can't see anything if we plot every cycle...
I = I[:-trim]
I2 = I[::15]

width = config['2col_width'] / 25.4
height = config['3col_width'] / 25.4

col_height = 0.45
top_height = 0.3

# Figure start
plt.figure(figsize=(width*3, height*2))

# Plot control point segments
plt.axes([.1, .1, .2, col_height])
#plt.pcolormesh(segments[I2, :])
plt.plot(ctrl[I2, 1], np.arange(len(I2)), '^')
plt.plot(ctrl[I2, 2], np.arange(len(I2)), 'x')
plt.plot(ctrl[I2, 3], np.arange(len(I2)), 'v')
plt.plot(ctrl[I2, 4], np.arange(len(I2)), '.')
plt.legend(['Peak', 'Desc', 'Trough', 'Asc'], frameon=False, loc='center', bbox_to_anchor=(0.4, 0.2, 1, 1))
plt.xticks(np.linspace(0, 200, 5), (np.linspace(0, 200, 5)/sample_rate*1000).astype(int))
plt.xlabel('Time (ms)')
plt.xlim(0, 250)
plt.ylim(0, len(I2))
plt.ylabel('# Cycle (Sorted by duration)')
decorate_ax(plt.gca())

plt.axes([.1, .6, .2, top_height-0.05])
plt.plot((0.5, 0.5), (0, 800), 'k--')
plt.hist(ctrl_mets[0][I], np.linspace(-1, 1), alpha=.5)
plt.hist(ctrl_mets[1][I], np.linspace(-1, 1), alpha=.5)
#plt.xticks(np.linspace(-.25, .25, 3))
plt.legend(['Sinusoid', 'Peak/Trough', 'Ascent/Descent'], frameon=False,
           fontsize=10, loc='center', bbox_to_anchor=(0.5, 0.4, 1, 1))
decorate_ax(plt.gca())
plt.xlim(0, 1)
plt.ylim(0, 800)
plt.title('Control-Point Ratios\n')
plt.xlabel('Ratio')
plt.ylabel('Num Cycles')

# Plot temporally aligned instantaneous frequency
plt.axes([.425, .1, .2, col_height])
plt.imshow(instfreq[:, I2].T, interpolation='nearest',  vmin=6, vmax=12, origin='lower', aspect='auto')
decorate_ax(plt.gca())
plt.xticks(np.linspace(0, 200, 5), (np.linspace(0, 200, 5)/sample_rate*1000).astype(int))
plt.xlabel('Time (ms)')
plt.xlim(0, 250)

plt.axes([.425, .6, .2, top_height/2])
mn = np.nanmean(instfreq[:, I], axis=1)
sem = np.nanstd(instfreq[:, I], axis=1)
sem = sem / np.sqrt(np.sum(np.isnan(instfreq[:, I])==False, axis=1))
plt.errorbar(np.arange(313), mn, yerr=sem, errorevery=4)
decorate_ax(plt.gca())
plt.xticks(np.linspace(0, 200, 5), (np.linspace(0, 200, 5)/sample_rate*1000).astype(int))
plt.xlim(0, 250)
plt.legend(['Avg IF (std-error of mean)'], loc='center', bbox_to_anchor=(0.3, 0.5, 1, 1), frameon=False)
plt.ylabel('Instantaneous\nFrequency (Hz)')

plt.axes([.425, .8, .2, 0.075])
plt.plot(np.nanmean(waveform[:, I], axis=1), 'k')
for tag in ['top', 'right', 'bottom']:
    plt.gca().spines[tag].set_visible(False)
plt.xticks([])
plt.ylim(-200, 200)
plt.xlim(0, 250)
plt.legend(['Avg Waveform'], loc='center', bbox_to_anchor=(0.3, 0.5, 1, 1), frameon=False)
plt.ylabel(r'Amplitude ($\mu$V)')
plt.title('Cycle-Onset Alignment\n\n')#\nInstantaneous. Frequency\n(std-error of mean)')

# Plot phase aligned instantaneous frequency
plt.axes([.75, .1, .2, col_height])
pcm = plt.imshow(pa[:, I2].T, interpolation='nearest',  vmin=6, vmax=12, origin='lower', aspect='auto')
plt.xticks(np.arange(5)*12, ['-pi', '-pi/2', '0', 'pi/2', 'pi'])
plt.xlabel('Theta Phase (rads)')
decorate_ax(plt.gca())

plt.axes([.75, .6, .2, top_height/2])
mn = np.nanmean(pa[:, I], axis=1)
sem = np.nanstd(pa[:, I], axis=1) / np.sqrt(I.shape[0])
plt.errorbar(np.arange(48), mn, yerr=sem, errorevery=2)
plt.xlim(0, 48)
decorate_ax(plt.gca())
plt.xticks(np.arange(5)*12, ['-pi', '-pi/2', '0', 'pi/2', 'pi'])
plt.ylabel('Instantaneous\nFrequency (Hz)')
plt.legend(['Avg IF (std-error of mean)'], loc='center', bbox_to_anchor=(0.3, 0.5, 1, 1), frameon=False)

plt.axes([.75, .8, .2, 0.075])
plt.plot(196*np.sin(2*np.pi*np.linspace(0, 1, 48)), 'k')
for tag in ['top', 'right', 'bottom']:
    plt.gca().spines[tag].set_visible(False)
plt.xticks([])
plt.xlim(0)
plt.ylim(-200, 200)
plt.legend(['Avg Waveform'], loc='center', bbox_to_anchor=(0.3, 0.5, 1, 1), frameon=False)
plt.ylabel(r'Amplitude ($\mu$V)')
plt.title('Phase Alignment\n\n')#\nInstantaneous. Frequency\n(std-error of mean)')

# Inst. freq colourbar
ax = plt.axes([.635, .25, .015, .18])
cb = plt.colorbar(pcm, cax=ax)
ax.yaxis.set_ticks_position('left')
plt.title('Instantaneous\nFrequency (Hz)\n', fontsize=12)

outname = os.path.join(config['figdir'], 'emd_fig8_real_phasealign_revised.png')
plt.savefig(outname, dpi=300, transparent=True)

outname = os.path.join(config['figdir'], 'emd_fig8_real_phasealign_revised.pdf')
plt.savefig(outname, dpi=300, transparent=True)


from scipy import stats
base = 'M={0}, SD={1}, t({2})={3}, p={4}'
tt = stats.ttest_1samp(ctrl_mets[0][I], 0.5)
print('Control point ratios = peak to trough - 1 sample ttest')
print(base.format(ctrl_mets[0][I].mean(), ctrl_mets[0][I].std(), len(I)-1, tt.statistic, tt.pvalue))

tt = stats.ttest_1samp(ctrl_mets[1][I], 0.5)
print('Control point ratios = ascent to descent - 1 sample ttest')
print(base.format(ctrl_mets[1][I].mean(), ctrl_mets[1][I].std(), len(I)-1, tt.statistic, tt.pvalue))
