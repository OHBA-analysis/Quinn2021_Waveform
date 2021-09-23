#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

# %% -----------------------------------------------------
#
# This script runs the simulations and analysis of the noisy 12Hz oscillator
# seen in figures 3, 4 and 5. The oscillation is generated and some general EMD
# and wavelet frequency metrics are computed. The three figures are then
# generated using these variables.

# %% -----------------------------------------------------
# Imports and definitions

import os
import emd
import sails
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats, ndimage
from emd_waveform_utils import config

import matplotlib
matplotlib.rc('font', serif=config['fontname'])

# %% ---------------------------------------------------
# Define systems from Feynman Vol 1 50-6

def linear_system(x, K):
    """ A linear system which scales a signal by a factor"""
    return K * x


def nonlinear_system(x, K, eta=.43, power=2):
    """ A non-linear system which scales a signal by a factor introduces a
    waveform distortion"""
    return K * (x + eta * (x ** power))


# %% ---------------------------------------------------
# Generate simuated data

# Create 60 seconds of data at 12Hz
peak_freq = 12
sample_rate = 512
seconds = 60
noise_std = None
x = emd.utils.ar_simulate(peak_freq, sample_rate, seconds,
                          noise_std=noise_std, random_seed=42, r=.99)
x = x * 1e-5
t = np.linspace(0, seconds, seconds*sample_rate)

# Apply linear and non-linear equations and add noise

x_linear_raw = linear_system(x, K=1)
x_nonlinear_raw = nonlinear_system(x, K=1, eta=2)

x_linear = x_linear_raw + np.random.randn(len(t), 1)*2e-2
x_nonlinear = x_nonlinear_raw + np.random.randn(len(t), 1)*2e-2

# %% ---------------------------------------------------
# Run frequency analyses

# Welch's Periodogram
f, pxx_linear = signal.welch(x_linear[:, 0], fs=sample_rate, nperseg=2048)
f, pxx_nonlinear = signal.welch(x_nonlinear[:, 0], fs=sample_rate, nperseg=2048)

# EMD
sift_config = {'imf_opts': {'sd_thresh': 5e-2},
               'mask_freqs': 120/sample_rate,
               'mask_amp_mode': 'ratio_sig',
               'mask_step_factor': 2.5}

imf_linear = emd.sift.mask_sift(x_linear, **sift_config)
imf_nonlinear = emd.sift.mask_sift(x_nonlinear, **sift_config)

IP_linear, IF_linear, IA_linear = emd.spectra.frequency_transform(imf_linear, sample_rate, 'hilbert')
IP_nonlinear, IF_nonlinear, IA_nonlinear = emd.spectra.frequency_transform(imf_nonlinear, sample_rate, 'hilbert')


# %% --------------------------------------------------
# Cycle analysis

def my_range(x):
    return x.max() - x.min()

def asc2desc(x):
    """Ascending to Descending ratio ( A / A+D )."""
    pt = emd.cycles.cf_peak_sample(x, interp=True)
    tt = emd.cycles.cf_trough_sample(x, interp=True)
    if (pt is None) or (tt is None):
        return np.nan
    asc = pt + (len(x) - tt)
    desc = tt - pt
    return asc / len(x)

def peak2trough(x):
    """Peak to trough ratio ( P / P+T )."""
    des = emd.cycles.cf_descending_zero_sample(x, interp=True)
    if des is None:
        return np.nan
    return des / len(x)

Cl = emd.cycles.Cycles(IP_linear[:, 2])
Cl.compute_cycle_metric('max_amp', IA_linear[:, 2], np.max)
Cl.compute_cycle_metric('max_if', IF_linear[:, 2], np.max)
Cl.compute_cycle_metric('if_range', IF_linear[:, 2], my_range)

Cn = emd.cycles.Cycles(IP_nonlinear[:, 2])
Cn.compute_cycle_metric('max_amp', IA_nonlinear[:, 2], np.max)
Cn.compute_cycle_metric('max_if', IF_nonlinear[:, 2], np.max)
Cn.compute_cycle_metric('if_range', IF_nonlinear[:, 2], my_range)
Cn.compute_cycle_metric('asc2desc', imf_nonlinear[:, 2], asc2desc)
Cn.compute_cycle_metric('peak2trough', imf_nonlinear[:, 2], peak2trough)

conditions = ['is_good==1', 'max_amp>0.04', 'if_range<8', 'max_if<18']
pa_linear, phase_x = emd.cycles.phase_align(IP_linear[:, 2], IF_linear[:, 2],
                                            cycles=Cl.iterate(conditions=conditions))

pa_nonlinear, phase_x = emd.cycles.phase_align(IP_nonlinear[:, 2], IF_nonlinear[:, 2],
                                               cycles=Cn.iterate(conditions=conditions))
df_nonlinear = Cn.get_metric_dataframe(conditions=conditions)

# %% --------------------------------------------------
# Time-frequency transform

# Hilbert-Huang Transform
edges, centres = emd.spectra.define_hist_bins(0, 40, 64)
spec_linear = emd.spectra.hilberthuang_1d(IF_linear, IA_linear, edges, mode='energy')/x_linear.shape[0]
spec_nonlinear = emd.spectra.hilberthuang_1d(IF_nonlinear, IA_nonlinear, edges, mode='energy')/x_nonlinear.shape[0]

# Carrier frequency histogram definition
edges, bins = emd.spectra.define_hist_bins(2, 35, 64, 'linear')

# Compute the 2d Hilbert-Huang transform (power over time x carrier frequency)
hht_linear = emd.spectra.hilberthuang(IF_linear[:, 2], IA_linear[:, 2], edges, mode='amplitude')
hht_nonlinear = emd.spectra.hilberthuang(IF_nonlinear[:, 2], IA_nonlinear[:, 2], edges, mode='amplitude')

# Smooth HHTs to help visualisation
hht_linear = ndimage.gaussian_filter(hht_linear, .5)
hht_nonlinear = ndimage.gaussian_filter(hht_nonlinear, 1)

# Compute 2d wavelet transform
cwt_linear = sails.wavelet.morlet(x_linear[:, 0], bins, sample_rate, normalise='simple', ret_mode='amplitude')
cwt_nonlinear = sails.wavelet.morlet(x_nonlinear[:, 0], bins, sample_rate, normalise='simple', ret_mode='amplitude')

# %% --------------------------------------------------
# FIGURE 3 - Example system with time-frequency transforms


def decorate_ax(ax):
    for tag in ['top', 'right']:
        ax.spines[tag].set_visible(False)


inds = np.arange(7550, 8550)

width = config['3col_width'] / 25.4
height = width

matches = Cn.get_matching_cycles(conditions)
goods = emd._cycles_support.project_cycles_to_samples(matches, Cn.cycle_vect)[:, 0]

plt.figure(figsize=(width*2, height*2))

# Plot time-series
plt.axes([.1, .5, .875, .45], frameon=False)
plt.xticks([])
plt.yticks([])
plt.plot(x_nonlinear[inds]+0.5, 'k')
plt.plot(imf_nonlinear[inds, 2:].sum(axis=1)-0.25, 'g')
plt.text(-50, 1, 'Cycle #', verticalalignment='center', horizontalalignment='right')
plt.text(-50, 0.5, 'Signal', verticalalignment='center', horizontalalignment='right')
plt.text(-50, -.2, 'IMF-3', verticalalignment='center', horizontalalignment='right')


# Instantaneous Phase
ip = IP_nonlinear[inds, 2]
bad_cycles = np.logical_or(np.diff(ip) < -3, goods[inds[:-1]] == False)
bad_cycles = np.r_[bad_cycles, True]
bad_cycles = goods[inds[:-1]] == False
bad_cycles = np.r_[bad_cycles, True]

ip[np.where(np.diff(ip) < -3)[0]] = np.nan
to_plot = ip/15 - 1.15
plt.plot(to_plot)
#to_plot[:np.where(np.isnan(to_plot))[0][17]] = np.nan
to_plot[bad_cycles == False] = np.nan
plt.plot(to_plot, 'r')
mn = np.nanmin(to_plot)
mx = np.nanmax(to_plot)
plt.plot([-25, -25], [mn, mx], 'k')
plt.plot([-35, len(inds)], [mx, mx], color=[.8, .8, .8], linewidth=.5)
plt.plot([-35, len(inds)], [np.mean((mn, mx)), np.mean((mn, mx))], color=[.8, .8, .8], linewidth=.5)
plt.plot([-35, len(inds)], [mn, mn], color=[.8, .8, .8], linewidth=.5)
plt.text(-30, mx, 'pi', verticalalignment='center', horizontalalignment='right')
plt.text(-30, np.mean((mn, mx)), '0', verticalalignment='center', horizontalalignment='right')
plt.text(-30, mn, '-pi', verticalalignment='center', horizontalalignment='right')
plt.text(-105, np.mean((mn, mx)), 'Instantaneous\nPhase (rads)', ha='center', va='center', rotation=0)

# Instantanous Frequency
frange = emd._cycles_support.project_cycles_to_samples(Cn.metrics['if_range'], Cn.cycle_vect)[:, 0]
iif = IF_nonlinear[inds, 2].copy()
#iif[goods==0] = np.nan
iif[bad_cycles] = np.nan
to_plot = iif/20 - 2.15
plt.plot(to_plot)
freq_range = np.array([8, 12, 16])
freq_range_conv = freq_range/20 - 2.2

mn = np.nanmin(to_plot)
mx = np.nanmax(to_plot)
plt.plot([-25, -25], [mn, mx], 'k')
plt.plot([-35, len(inds)], [mx, mx], color=[.8, .8, .8], linewidth=.5)
plt.plot([-35, len(inds)], [np.mean((mn, mx)), np.mean((mn, mx))], color=[.8, .8, .8], linewidth=.5)
plt.plot([-35, len(inds)], [mn, mn], color=[.8, .8, .8], linewidth=.5)
for ii in range(3):
    plt.text(-30, freq_range_conv[ii], '{0}Hz'.format(freq_range[ii]),
             verticalalignment='center', horizontalalignment='right')
plt.text(-105, freq_range_conv[1], 'Instantaneous\nFrequency (Hz)', ha='center', va='center', rotation=0)

# Cycle Boundaries
yl = plt.ylim()
cycle_bounds = np.where(np.diff(Cn.cycle_vect[inds, 0]) > .5)[0]
for ii in range(len(cycle_bounds)):
    plt.plot([cycle_bounds[ii], cycle_bounds[ii]], [-2.2, 1.4], color=[.8, .8, .8], linewidth=.5)
    if ii < len(cycle_bounds)-1:
        plt.text( (cycle_bounds[ii]+cycle_bounds[ii+1])/2, 1, str(ii+1), horizontalalignment='center')
plt.ylim(yl)
plt.xlim(-55, 896)

# Hilbert-Huang Transform
tt = np.linspace(0, len(inds)/sample_rate, len(inds))
plt.axes([.15, .275, .825, .2])
pcm = plt.pcolormesh(tt, bins, hht_nonlinear[:, inds], cmap='hot_r', vmin=0, vmax=.175)
yl = plt.ylim()
for ii in range(len(cycle_bounds)):
    plt.plot([tt[cycle_bounds[ii]], t[cycle_bounds[ii]]], [0, bins[-1]], color=[.8, .8, .8], linewidth=.5)
plt.ylim(yl)
for tag in ['top', 'right']:
    plt.gca().spines[tag].set_visible(False)
plt.gca().set_xticklabels([])
plt.ylabel('Frequency (Hz)')
plt.xlim(0, 1.75)
ax = plt.axes([.97, .285, .015, .18])
cb = plt.colorbar(pcm, cax=ax)
ax.yaxis.set_ticks_position('left')
cb.set_label('Power')

# Wavelet Transform
plt.axes([.15, .05, .825, .2])
pcm = plt.pcolormesh(tt, bins, cwt_nonlinear[:, inds], cmap='hot_r', vmin=0, vmax=.175)
yl = plt.ylim()
for ii in range(len(cycle_bounds)):
    plt.plot([tt[cycle_bounds[ii]], tt[cycle_bounds[ii]]], [0, bins[-1]], color=[.8, .8, .8], linewidth=.5)
plt.ylim(yl)
for tag in ['top', 'right']:
    plt.gca().spines[tag].set_visible(False)
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (seconds)')
plt.xlim(0, 1.75)
ax = plt.axes([.97, .06, .015, .18])
cb = plt.colorbar(pcm, cax=ax)
ax.yaxis.set_ticks_position('left')
cb.set_label('Power')

outname = os.path.join(config['figdir'], 'emd_fig3_simu_decomp.png')
plt.savefig(outname, dpi=300, transparent=True)

# %% --------------------------------------------------
# FIGURE 4 - PHASE ALIGNMENT IN SIMULATION

# Get temporally aligned waveforms and instantanous frequencies
waveform_linear = np.zeros((100, Cl.ncycles))*np.nan
instfreq_linear = np.zeros((100, Cl.ncycles))*np.nan

for ii, inds in Cl.iterate(conditions=conditions):
    waveform_linear[:len(inds), ii] = imf_linear[inds, 2]
    instfreq_linear[:len(inds), ii] = IF_linear[inds, 2]

ctrl_linear = emd.cycles.get_control_points(imf_linear[:, 2], Cl.iterate(conditions=conditions), interp=True)
ctrl_mets_linear = emd.cycles.get_control_point_metrics(ctrl_linear)

waveform_nonlinear = np.zeros((100, Cn.ncycles))*np.nan
instfreq_nonlinear = np.zeros((100, Cn.ncycles))*np.nan

for ii, inds in Cn.iterate(conditions=conditions):
    waveform_nonlinear[:len(inds), ii] = imf_nonlinear[inds, 2]
    instfreq_nonlinear[:len(inds), ii] = IF_nonlinear[inds, 2]

ctrl_nonlinear = emd.cycles.get_control_points(imf_nonlinear[:, 2], Cn.iterate(conditions=conditions), interp=True)
ctrl_mets_nonlinear = emd.cycles.get_control_point_metrics(ctrl_nonlinear)

I = np.argsort(ctrl_nonlinear[:, 4])[::-1]
segments = np.zeros((ctrl_nonlinear.shape[0], 60))*np.nan
for ii in range(ctrl_nonlinear.shape[0]):
    for jj in range(1, ctrl_nonlinear.shape[1]):
        # Round segments to ints for visualisation
        segments[ii, int(np.floor(ctrl_nonlinear[ii, jj-1])):int(np.ceil(ctrl_nonlinear[ii, jj]))] = jj

# Figure start
width = config['2col_width'] / 25.4
height = width

plt.figure(figsize=(width*2, height*2))

# Plot control point segments
plt.axes([.1, .1, .2, .65])
plt.pcolormesh(segments[I, :])
plt.xticks(np.linspace(0, 40, 3))
decorate_ax(plt.gca())
plt.ylabel('Cycles (sorted)')
plt.xticks(np.linspace(0, 0.08*sample_rate, 5), np.linspace(0, 80, 5).astype(int))
plt.xlabel('Time (ms)')
plt.axes([.1, .775, .144, .075], frameon=False)
plt.xticks([])
plt.yticks([])
cols = plt.cm.viridis(np.linspace(0, 1, 4))
for ii in range(4):
    xvals = np.linspace(0, .25)+.25*ii
    plt.plot(xvals, np.sin(2*np.pi*xvals), linewidth=3, color=cols[ii, :])

# Plot control point metrics
plt.axes([.31, .1, .1, .65])
plt.plot(ctrl_mets_nonlinear[0][I], np.arange(len(ctrl_mets_nonlinear[0])), '.')
plt.plot(ctrl_mets_nonlinear[1][I], np.arange(len(ctrl_mets_nonlinear[0])), '+')
plt.plot(np.zeros_like(ctrl_mets_nonlinear[1][I]), np.arange(len(ctrl_mets_nonlinear[0])), 'k', linewidth=.5)
plt.xlim(-.3, .3)
plt.ylim(0, ctrl_nonlinear.shape[0])
plt.yticks([])
decorate_ax(plt.gca())

plt.axes([.31, .775, .1, .15])
plt.hist(ctrl_mets_nonlinear[0], np.linspace(-1, 1), alpha=.5)
plt.hist(ctrl_mets_nonlinear[1], np.linspace(-1, 1), alpha=.5)
plt.xlim(-.3, .3)
plt.xticks(np.linspace(-.25, .25, 3), [])
plt.legend(['Peak/Trough', 'Ascent/Descent'], frameon=False,
           fontsize=8, loc='center', bbox_to_anchor=(0.175, 0.45, 1, 1))
plt.ylim(0, 250)
decorate_ax(plt.gca())
plt.title('Control-Point Ratios')

# Plot temporally aligned instantaneous frequency
plt.axes([.5, .1, .2, .65])
plt.pcolormesh(instfreq_nonlinear[:, I].T)
decorate_ax(plt.gca())
plt.xticks(np.linspace(0, 0.08*sample_rate, 5), np.linspace(0, 80, 5).astype(int))
plt.xlabel('Time (ms)')
plt.xlim(0, 60)

plt.axes([.5, .775, .2, .15])
#plt.plot(instfreq_nonlinear, color=[.8, .8, .8])
plt.plot(np.nanmean(instfreq_nonlinear, axis=1))
decorate_ax(plt.gca())
plt.title('Cycle-Onset Aligned IF')
plt.xlim(0, 60)
plt.xticks(np.linspace(0, 0.08*sample_rate, 5), [])

# Plot phase aligned instantaneous frequency
plt.axes([.75, .1, .2, .65])
pcm = plt.pcolormesh(pa_nonlinear[:, I].T)
plt.xticks(np.arange(5)*12, ['-pi', '-pi/2', '0', 'pi/2', 'pi'])
plt.xlabel('Theta Phase')
plt.yticks(np.arange(6)*100, [])

plt.axes([.75, .775, .2, .15])
#plt.plot(pa_nonlinear[:, :-1], color=[.8, .8, .8])
plt.plot(np.nanmean(pa_nonlinear[:, :-1], axis=1))
plt.xlim(0, 48)
decorate_ax(plt.gca())
plt.xticks(np.arange(5)*12, [])
plt.title('Phase-Aligned IF')

# Inst. freq colourbar
ax = plt.axes([.685, .45, .015, .18])
cb = plt.colorbar(pcm, cax=ax)
ax.yaxis.set_ticks_position('left')
plt.title('Instantaneous\nFrequency (Hz)', fontsize=9)

outname = os.path.join(config['figdir'], 'emd_fig4_simu_phasealign.png')
plt.savefig(outname, dpi=300, transparent=True)

# %% --------------------------------------------------
# FIGURE 4 - PHASE ALIGNMENT IN SIMULATION : REVISED
I2 = I[::5]

width = config['2col_width'] / 25.4
height = config['3col_width'] / 25.4

col_height = 0.45
top_height = 0.3

# Figure start
plt.figure(figsize=(width*3, height*2))

# Plot control point segments
plt.axes([.1, .1, .2, col_height])
#plt.pcolormesh(segments[I2, :])
plt.plot(ctrl_nonlinear[I2, 1], np.arange(len(I2)), '^')
plt.plot(ctrl_nonlinear[I2, 2], np.arange(len(I2)), 'x')
plt.plot(ctrl_nonlinear[I2, 3], np.arange(len(I2)), 'v')
plt.plot(ctrl_nonlinear[I2, 4], np.arange(len(I2)), '.')
plt.legend(['Peak', 'Desc', 'Trough', 'Asc'], frameon=False, loc='center', bbox_to_anchor=(0.4, 0.2, 1, 1))
plt.xticks(np.linspace(0, 64, 5), (np.linspace(0, 125, 5)).astype(int))
plt.xlabel('Time (ms)')
plt.xlim(0, 64)
plt.ylim(0, len(I2))
plt.ylabel('# Cycle (Sorted by duration)')
decorate_ax(plt.gca())

plt.axes([.1, .6, .2, top_height-0.05])
plt.plot((0.5, 0.5), (0, 800), 'k--')
#plt.hist(ctrl_mets_nonlinear[0][I], np.linspace(-1, 1), alpha=.5)
#plt.hist(ctrl_mets_nonlinear[1][I], np.linspace(-1, 1), alpha=.5)
plt.hist(df_nonlinear['peak2trough'].values, np.linspace(0, 1), alpha=0.5)
plt.hist(df_nonlinear['asc2desc'].values, np.linspace(0, 1), alpha=0.5)
#plt.xticks(np.linspace(-.25, .25, 3))
plt.legend(['Sinusoid', 'Peak/Trough', 'Ascent/Descent'], frameon=False,
           fontsize=10, loc='center', bbox_to_anchor=(0.5, 0.4, 1, 1))
decorate_ax(plt.gca())
plt.xlim(1/3, 2/3)
plt.ylim(0, 250)
plt.title('Control-Point Ratios\n')
plt.xlabel('Ratio')
plt.ylabel('Num Cycles')

# Plot temporally aligned instantaneous frequency
plt.axes([.425, .1, .2, col_height])
plt.imshow(instfreq_nonlinear[:64, I2].T, interpolation='nearest', vmin=6, vmax=14, aspect='auto', origin='lower')
decorate_ax(plt.gca())
plt.xticks(np.linspace(0, 64, 5), (np.linspace(0, 125, 5)).astype(int))
plt.xlabel('Time (ms)')
plt.xlim(0, 64)

plt.axes([.425, .6, .2, top_height/2])
mn = np.nanmean(instfreq_nonlinear[:, I], axis=1)
sem = np.nanstd(instfreq_nonlinear[:, I], axis=1)
sem = sem / np.sqrt(np.sum(np.isnan(instfreq_nonlinear[:, I]) == False, axis=1))
plt.errorbar(np.arange(100), mn, yerr=sem, errorevery=4)
decorate_ax(plt.gca())
plt.xticks(np.linspace(0, 64, 5), (np.linspace(0, 125, 5)).astype(int))
plt.xlim(0, 64)
plt.legend(['Avg IF (std-error of mean)'], loc='center', bbox_to_anchor=(0.3, 0.5, 1, 1), frameon=False)
plt.ylabel('Instantaneous\nFrequency (Hz)')

plt.axes([.425, .8, .2, 0.075])
plt.plot(np.nanmean(waveform_nonlinear[:, I], axis=1), 'k')
for tag in ['top', 'right', 'bottom']:
    plt.gca().spines[tag].set_visible(False)
plt.xticks([])
plt.ylim(-0.1, 0.1)
plt.legend(['Avg Waveform'], loc='center', bbox_to_anchor=(0.3, 0.5, 1, 1), frameon=False)
plt.xlim(0, 64)
plt.ylabel(r'Amplitude (a.u.)')
plt.title('Cycle-Onset Alignment\n\n')#\nInstantaneous. Frequency\n(std-error of mean)')

# Plot phase aligned instantaneous frequency
plt.axes([.75, .1, .2, col_height])
pcm = plt.imshow(pa_nonlinear[:, I2].T, interpolation='nearest',  vmin=6, vmax=14, aspect='auto', origin='lower')
decorate_ax(plt.gca())
plt.xticks(np.arange(5)*12, ['-pi', '-pi/2', '0', 'pi/2', 'pi'])
plt.xlabel('Theta Phase (rads)')

plt.axes([.75, .6, .2, top_height/2])
mn = np.nanmean(pa_nonlinear[:, I], axis=1)
sem = np.nanstd(pa_nonlinear[:, I], axis=1) / np.sqrt(I.shape[0])
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
plt.ylim(-200, 200)
plt.legend(['Avg Waveform'], loc='center', bbox_to_anchor=(0.3, 0.5, 1, 1), frameon=False)
plt.ylabel(r'Amplitude (a.u.)')
plt.title('Phase Alignment\n\n')#\nInstantaneous. Frequency\n(std-error of mean)')

# Inst. freq colourbar
ax = plt.axes([.635, .25, .015, .18])
cb = plt.colorbar(pcm, cax=ax)
ax.yaxis.set_ticks_position('left')
plt.title('Instantaneous\nFrequency (Hz)\n', fontsize=12)

outname = os.path.join(config['figdir'], 'emd_fig4_simu_phasealign_revised.png')
plt.savefig(outname, dpi=300, transparent=True)

outname = os.path.join(config['figdir'], 'emd_fig4_simu_phasealign_revised.pdf')
plt.savefig(outname, dpi=300, transparent=True)

# %% --------------------------------------------------
# FIGURE 5 - SHAPE COMPARISON

pa_linear_avg = np.nanmean(pa_linear, axis=1)
fs = np.mean(pa_linear_avg)
lin_phase = emd.spectra.phase_from_freq(pa_linear_avg, 48*fs)
pa_nonlinear_avg = np.nanmean(pa_nonlinear, axis=1)
fs = np.mean(pa_nonlinear_avg)
nonlin_phase = emd.spectra.phase_from_freq(pa_nonlinear_avg, 48*fs)
lin_phase = np.r_[-np.pi, lin_phase]
nonlin_phase = np.r_[-np.pi, nonlin_phase]

cols = [np.array([31, 119, 180])/255, np.array([255, 127, 14])/255]

cmap = 'hot_r'
inds = np.arange(sample_rate*1.6, sample_rate*2.55).astype(int)

width = config['2col_width'] / 25.4
heigh = config['3col_width'] / 25.4

plt.figure(figsize=(width*2, height*2))

# Original time-series
plt.axes([.3, .8, .4, .15], frameon=False)
plt.plot(x_linear_raw[inds], 'k')
plt.xticks([])
plt.yticks([])

# Linear and nonlinear systems
plt.axes([0.05, .6, .4, .25], frameon=False)
plt.plot(x_linear[inds]+.4, 'k')
plt.plot(imf_linear[inds, 2]-.2, color=cols[0])
plt.xticks([])
plt.yticks([])
plt.axes([0.55, .6, .4, .25], frameon=False)
plt.plot(x_nonlinear[inds]+.4, 'k')
plt.plot(imf_nonlinear[inds, 2]-.2, color=cols[1])
plt.xticks([])
plt.yticks([])

# Linear phase-aligned IF
plt.axes([0.05+.1, .35, .2, .2])
plt.plot(pa_linear, color=[.8, .8, .8])
plt.plot(np.nanmean(pa_linear, axis=1), color=cols[0])
plt.xticks(np.arange(4)*12, ['-pi', '-pi/2', '0', 'pi/2', 'pi'])
plt.grid(True)
plt.xlim(0, 48)
plt.ylim(5, 20)
decorate_ax(plt.gca())
plt.title('Phase-Aligned IF')
plt.xlabel('Theta-Phase')

# Non-linear phase-aligned IF
plt.axes([.55+.1, .35, .2, .2])
plt.plot(pa_nonlinear, color=[.8, .8, .8])
plt.plot(np.nanmean(pa_nonlinear, axis=1), color=cols[1])
plt.xticks(np.arange(4)*12, ['-pi', '-pi/2', '0', 'pi/2', 'pi'])
plt.grid(True)
plt.xlim(0, 48)
plt.ylim(5, 20)
decorate_ax(plt.gca())
plt.title('Phase-Aligned IF')
plt.xlabel('Theta-Phase')

# Phase-aligned IF comparison
plt.axes([.075, .05, .2, .175])
plt.plot(np.nanmean(pa_linear, axis=1), color=cols[0])
plt.plot(np.nanmean(pa_nonlinear, axis=1), color=cols[1])
plt.xticks(np.arange(5)*12, ['-pi', '-pi/2', '0', 'pi/2', 'pi'])
plt.grid(True)
decorate_ax(plt.gca())
plt.ylim(9, 15)
plt.title('Phase-Aligned\nAverage IF')

# Phase aligned IF t-test
plt.axes([.4, .05, .2, .175])
t, p = stats.ttest_ind(pa_nonlinear, pa_linear, axis=1)
plt.plot(t)
plt.xticks(np.arange(5)*12, ['-pi', '-pi/2', '0', 'pi/2', 'pi'])
plt.grid(True)
decorate_ax(plt.gca())
plt.xlim(0, 48)
plt.ylabel('t-value')
plt.title('Nonlinear>Linear\nt-test')

# Normalised waveforms
plt.axes([.725, .05, .2, .175])
plt.plot(-np.sin(lin_phase), color=cols[0])
plt.plot(-np.sin(nonlin_phase), color=cols[1])
plt.xticks(np.arange(5)*12, [])
plt.grid(True)
plt.legend(['Linear', 'Nonlinear'])
decorate_ax(plt.gca())
plt.ylim(-1, 1)
plt.xlim(0, 4*12)
plt.yticks(np.linspace(-1, 1, 3))
plt.title('Normalised Waveforms')

outname = os.path.join(config['figdir'], 'emd_fig5_shape_compare.png')
plt.savefig(outname, dpi=300, transparent=True)
