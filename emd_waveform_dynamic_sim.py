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

def asc2desc(x):
    """Ascending to Descending ratio ( A / A+D )."""
    pt = emd.cycles.cf_peak_sample(x, interp=True)
    tt = emd.cycles.cf_trough_sample(x, interp=True)
    if (pt is None) or (tt is None):
        return np.nan
    asc = pt + (len(x) - tt)
    desc = tt - pt
    return asc / len(x)

# %% ---------------------------------------------

# Create 60 seconds of data at 12Hz
peak_freq = 12
sample_rate = 512
seconds = 60
noise_std = None
x = emd.utils.ar_simulate(peak_freq, sample_rate, seconds,
                          noise_std=noise_std, random_seed=42, r=.99)
x = x * 1e-5
t = np.linspace(0, seconds, seconds*sample_rate)

IP, IF, IA = emd.spectra.frequency_transform(x, sample_rate, 'hilbert')
C = emd.cycles.Cycles(IP[:, 0])

def distort_phase(ph, freq, amp, phase):
    dist = np.sin(2*np.pi*freq*np.linspace(0, 1, len(ph)) + phase) * amp
    return ph + dist - dist[0]

from functools import partial
amp = 4/5
templates = [partial(distort_phase, freq=1, amp=0, phase=0),
             partial(distort_phase, freq=1, amp=amp, phase=0),
             partial(distort_phase, freq=1, amp=amp, phase=np.pi)]

dist_phase = IP[:, 0].copy()
np.random.seed(42)
waveform_type = np.random.choice(np.arange(3), C.ncycles)
waveform_vect = emd._cycles_support.project_cycles_to_samples(waveform_type, C.cycle_vect)

tempph = np.zeros((512, 3))
tempx = np.zeros((512, 3))
for ii in range(3):
    tempph[:, ii] = templates[ii](np.linspace(0, np.pi*2, 512))
    tempx[:, ii] = np.sin(tempph[:, ii])

for idx, sl in enumerate(C._slice_cache):
    dist_phase[sl] = templates[waveform_type[idx]](dist_phase[sl])

dist_x = IA[:, 0] * np.sin(dist_phase)


np.random.seed(42)
y = dist_x[:, None] + np.random.randn(len(t), 1)*3.5e-2

# EMD
sift_config = {'imf_opts': {'sd_thresh': 5e-2},
               'mask_freqs': 100/sample_rate,
               'mask_amp_mode': 'ratio_sig',
               'nphases': 24,
               'mask_step_factor': 2.5}

imf = emd.sift.mask_sift(y, **sift_config)
IP, IF, IA = emd.spectra.frequency_transform(imf, sample_rate, 'hilbert')

def mode(x):
    return stats.mode(x)[0][0]

C = emd.cycles.Cycles(IP[:, 2])
C.compute_cycle_metric('max_amp', IA[:, 2], np.max)
C.compute_cycle_metric('max_if', IF[:, 2], np.max)
C.compute_cycle_metric('state', waveform_vect[:, 0], mode)
C.compute_cycle_metric('asc2desc', imf[:, 2], asc2desc)
#C.add_cycle_metric('state', waveform_type)

conditions = ['is_good==1', 'max_amp>0.03', 'max_if<18']
C.pick_cycle_subset(conditions)
pa, phase_x = emd.cycles.phase_align(IP[:, 2], IF[:, 2],
                                            cycles=C.iterate(through='subset'))
df = C.get_metric_dataframe(conditions=conditions)

pc_data = pa.T
cycle_mean = pc_data.mean(axis=1)[:, None]
phase_mean = pc_data.mean(axis=0)[:, None]
pc_data = pc_data - cycle_mean

pca = sails.utils.PCA(pc_data, npcs=10)
df['comp1'] = pca.scores[:, 0]
df['comp2'] = pca.scores[:, 1]
df['comp3'] = pca.scores[:, 2]
comp1 = emd._cycles_support.project_subset_to_samples(pca.scores[:, 0], C.subset_vect, C.cycle_vect)
wst = emd._cycles_support.project_subset_to_samples(df['state'].values, C.subset_vect, C.cycle_vect)
#wst = emd._cycles_support.project_cycles_to_samples(waveform_type, C.cycle_vect)
a2d = emd._cycles_support.project_subset_to_samples(df['asc2desc'].values, C.subset_vect, C.cycle_vect)

pc_proj = np.zeros((48, 2, 10))
wf_proj = np.zeros((49, 2, 10))
val = 15  # PC-score to project
for ii in range(10):
    sc = np.zeros((2, 10))

    sc[0, ii] = val
    sc[1, ii] = -val
    pc_proj[:, :, ii] = pca.project_score(sc).T + phase_mean

    sr = pc_proj[:, 0, ii].mean() * 49
    phase = emd.spectra.phase_from_freq(pc_proj[:, 0, ii], sr, phase_start=0)
    phase = np.r_[0, phase]
    wf_proj[:, 0, ii] = np.sin(phase)

    sr = pc_proj[:, 1, ii].mean() * 49
    phase = emd.spectra.phase_from_freq(pc_proj[:, 1, ii], sr, phase_start=0)
    phase = np.r_[0, phase]
    wf_proj[:, 1, ii] = np.sin(phase)

state_cols = plt.cm.Set3(np.linspace(0, 1, 5))[1:, :]
#state_cols = cols = np.array([[57, 119, 75, 1], [239, 133, 53, 1], [81, 157, 62, 1]])

inds = np.arange(3250, 4150)+500
inds = np.arange(7775, 8550) - 250
starts = np.where(np.diff(C.cycle_vect.T) == 1)[1]
tend = t[inds[-1]] + 0.01

plt.figure(figsize=(18, 5))
plt.axes([0.1, 0.6, 0.8, 0.3])
plt.plot(t[inds], y[inds], 'k')
plt.plot(t[inds], imf[inds, 2], 'r', linewidth=2)
xl = plt.xlim()
yl = plt.ylim()
for ii in range(3):
    tmpx = (wst == ii).astype(float) * 0.35
    tmpx[tmpx == 0] = np.nan
    plt.plot(t[inds], tmpx[inds], linewidth=12, color=state_cols[ii, :], solid_capstyle="butt")
for idx, st in enumerate(starts):
    if (st > inds[0]) and (st < inds[-1]) and idx-179 < 19:
        plt.plot([t[st], t[st]], [yl[0], 0.35], color=[0.8, 0.8, 0.8])
        plt.text((t[st]+t[starts[idx+1]])/2, 0.36, '#{0}'.format(idx-179), ha='center')
plt.ylim(-0.3, 0.35)
for tag in ['top', 'right', 'left', 'bottom']:
    plt.gca().spines[tag].set_visible(False)
plt.xticks([]); plt.yticks([])
plt.xlim(t[inds[0]], tend)
plt.text(14.6, 0.35, 'Cycle Type', va='center')
plt.text(14.6, 0.05, 'Signal')
plt.text(14.6, -0.05, 'IMF', color='r')

plt.axes([0.1, 0.45, 0.8, 0.15])
plt.plot(t[inds], IF[inds, 2])
plt.xlim(xl)
yl = (4, 20)
for st in starts:
    if (st > inds[0]) and (st < inds[-1]):
        plt.plot([t[st], t[st]], [yl[0], yl[1]], color=[0.8, 0.8, 0.8])
plt.ylim(yl)
for tag in ['top', 'right', 'bottom']:
    plt.gca().spines[tag].set_visible(False)
plt.gca().spines['left'].set_bounds(6, 14)
plt.yticks([6, 10, 14])
plt.xticks([])
plt.xlim(t[inds[0]], tend)
plt.ylabel('Instantaneous\nFrequency (Hz)', rotation=0)
plt.gca().yaxis.set_label_coords(-0.07, 0.2)

plt.axes([0.1, 0.2, 0.8, 0.25])
plt.fill_between(t[inds], comp1[inds, 0])
plt.xlim(xl)
yl = (-11, 15) #plt.ylim()
for st in starts:
    if (st > inds[0]) and (st < inds[-1]):
        plt.plot([t[st], t[st]], [yl[0], yl[1]], color=[0.8, 0.8, 0.8])
plt.ylim(yl)
for tag in ['top', 'right']:
    plt.gca().spines[tag].set_visible(False)
plt.gca().spines['left'].set_bounds(-10, 10)
plt.xticks(t[inds[::sample_rate//4]], np.linspace(0, 1.75, 8))
plt.yticks([-10, -5, 0, 5, 10])
plt.plot(np.linspace(tend, tend+0.2, 49), 5*wf_proj[:, 0, 0]+5.1, 'k', linewidth=2, color=state_cols[2, :])
plt.plot(np.linspace(tend, tend+0.2, 49), 5*wf_proj[:, 1, 0]-5.1, 'k', linewidth=2, color=state_cols[1, :])
plt.xlim(t[inds[0]], tend)
plt.ylabel('PCA Motif 1\nScore', rotation=0)
plt.xlabel('Time (seconds)')
plt.gca().yaxis.set_label_coords(-0.07, 0.2)

outname = os.path.join(config['figdir'], 'emd_fig4_dyn_timeseries.png')
plt.savefig(outname, dpi=300, transparent=True)

# %% ----------------------------------------------------------------

plt.figure(figsize=(12, 4))
plt.subplots_adjust(bottom=0.15, wspace=0.35, hspace=0.4)
plt.subplot(231)
plt.plot(np.sin(2*np.pi*np.linspace(0, 1, 49)), 'k:')
plt.plot(wf_proj[:, 0, 0])
plt.plot(wf_proj[:, 1, 0])
plt.xticks(np.linspace(0, 49, 3), [0, 0.5, 1])
plt.legend(['Sinusoid', '+ve score', '-ve score'], frameon=False, fontsize='small')
plt.xlabel('Duration')
plt.ylabel('Amplitude (a.u.)')
for tag in ['top', 'right']:
    plt.gca().spines[tag].set_visible(False)

plt.subplot(234)
plt.plot(pca.components[0, :], 'k')
plt.xticks(np.linspace(0, 48, 5), ['-pi', '-pi/2', '0', 'pi/2', 'pi'])
for tag in ['top', 'right']:
    plt.gca().spines[tag].set_visible(False)
plt.xlabel('Phase')
plt.ylabel('Component Weight')

plt.subplot(132)
for idx, ii in enumerate([1, 0, 2]):
    c = state_cols[idx, :]
    plt.boxplot(df['comp1'].values[df['state'].values == idx], positions=[0.7+0.3*ii],
                patch_artist=True,
                boxprops=dict(facecolor=c, color=c),
                capprops=dict(color=c),
                whiskerprops=dict(color=c),
                flierprops=dict(color=c, markeredgecolor=c),
                medianprops=dict(color=c),
                )
plt.xticks([0.7, 1, 1.3], ['Fast\nAscent', 'Sinusoid', 'Fast\nDescent'])
plt.xlim(0.5, 1.5)
plt.xlabel('Waveform Type')
plt.ylabel('PC-1 Score')
for tag in ['top', 'right']:
    plt.gca().spines[tag].set_visible(False)

plt.subplot(133)
plt.plot(df['asc2desc'], df['comp1'], '.')
plt.xlabel('Ascent to Descent Ratio')
plt.ylabel('PC-1 Score')
for tag in ['top', 'right']:
    plt.gca().spines[tag].set_visible(False)

outname = os.path.join(config['figdir'], 'emd_fig4_dyn_summary.png')
plt.savefig(outname, dpi=300, transparent=True)


z = imf[:, 1:].sum(axis=1)
plt.figure(figsize=(12, 4))
plt.subplots_adjust(bottom=0.2)
plt.plot(z, 'k')
plt.plot(emd.sift.interp_envelope(z, interp_method='splrep'))
plt.plot(emd.sift.interp_envelope(z, interp_method='mono_pchip'))
plt.xlim(1500, 3000)
plt.legend(['Signal', 'Cubic-Spline Envelope', 'Monotonic PCHIP Envelope'], frameon=False)
for tag in ['top', 'right']:
    plt.gca().spines[tag].set_visible(False)
plt.xlabel('Time (samples)')

outname = os.path.join(config['figdir'], 'emd_supp_interpolation.png')
plt.savefig(outname, dpi=300, transparent=True)
