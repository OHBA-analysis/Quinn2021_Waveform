#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

#%% -----------------------------------------------------
#
# This script loads the EMD analyses from one run of the LFP data and creates
# figures 6 and 8. Figure 6 shows a segment of the time-series and associated
# EMD metrics and figure 8 shows the single cycle representation of around 2000
# cycles.

#%% -----------------------------------------------------
# Imports and definitions

import os
import emd
import h5py
import sails
import pandas
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from emd_waveform_methods_utils import config

import matplotlib
matplotlib.rc('font', serif=config['fontname'])

#%% ------------------------------------------------------

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


#%% ------------------------------------------
# Create figure 5 time-series

width = config['3col_width'] / 25.4
height = width * .6

plot_horiz = True
sparse_horiz = True
plot_vert = True
fontsize_side = 15
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
plt.axes([.07, .025, .95, .95], frameon=False)
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
         verticalalignment='bottom', fontsize=fontsize_side-2)

## Plot IMFs
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
plt.text(-275, -3500+1*75*((2*np.pi)/2), 'Phase (rads)', fontsize=fontsize_side,
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
plt.text(-275, indx[1], 'IF (Hz)', fontsize=fontsize_side, verticalalignment='center', horizontalalignment='center')

# Plot cycle bounds and compute within cycle frequency variability
cycles_to_plot = emd.cycles.get_cycle_inds(IP[inds, 5, None])
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
    if plot_horiz and sparse_horiz is False :
        plt.plot([-10, len(inds)], (indx[ii], indx[ii]), color=[.7, .7, .7], linewidth=horiz_width)
    elif  ii == 0:
        plt.plot([-10, len(inds)], (indx[ii], indx[ii]), color=[.7, .7, .7], linewidth=horiz_width)
    plt.plot((-10, 0), (-4601+100*ii, -4601+100*ii), 'k')
    plt.text(-22, indx[ii], indx_lab[ii], fontsize=fontsize_tick,
             verticalalignment='center', horizontalalignment='right')

plt.text(-275, indx[1:3].mean(), 'IF Std-Dev', fontsize=fontsize_side,
         verticalalignment='center', horizontalalignment='center')

outname = os.path.join(config['figdir'], 'emd_fig6_real_sift.png')
plt.savefig(outname, dpi=300, transparent=True)

#%% --------------------------------------------------------------------
# Create figure 5 - spectra

edges, bins = emd.spectra.define_hist_bins(2, 35, 64, 'linear')

cwt = sails.wavelet.morlet(imf[inds, :6].sum(axis=1), bins, sample_rate, normalise='simple', ret_mode='amplitude')
hht = emd.spectra.hilberthuang(IF[inds, :6], IA[inds, :6], edges, mode='amplitude')
hht = ndimage.gaussian_filter(hht, 1)

t = np.arange(len(t))

plt.figure(figsize=(width*1.925, height*1.25))

plt.axes([.12, .55, .855, .425], frameon=True)
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

plt.axes([.12, .095, .855, .425], frameon=True)
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


#%% --------------------------------------------------------------------
# Create Figure 7

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
