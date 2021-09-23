#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

# %% -----------------------------------------------------
#
# This script loads all six of of the analysed LFP data. It computes the
# average phase-aligned instantaneous frequency profiles and mean-vectors
# before running the PCA and GLM on the phase-aligned data. The results are
# plotted up in figure 9 showing the group average shapes and figure 10 showing
# the PCA-GLM.

# %% -----------------------------------------------------
# Imports and definitions

import os
import sails
import numpy as np
import emd
import h5py
import pandas
import glmtools as glm
from scipy import stats
from emd_waveform_utils import config

import matplotlib
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
matplotlib.rc('font', serif=config['fontname'])


def phase_from_freq(ifrequency, sample_rate, phase_start=-np.pi):
    """
    Compute the instantaneous phase of a signal from its instantaneous phase.

    Parameters
    ----------
    ifrequency : ndarray
        Input array containing the instantaneous frequencies of a signal
    sample_rate : scalar
        The sampling frequency of the data
    phase_start : scalar
         Start value of the phase output (Default value = -np.pi)

    Returns
    -------
    IP : ndarray
        The instantaneous phase of the signal

    """

    iphase_diff = (ifrequency / sample_rate) * (2 * np.pi)

    iphase = phase_start + np.cumsum(iphase_diff, axis=0)

    iphase = np.r_[phase_start, iphase]

    return iphase


def get_cycle_examples():
    from scipy import interpolate
    freq = 8
    seconds = 2/freq
    num_samples = 3124

    time_vect = np.linspace(0, seconds, num_samples)

    phs = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    tim2 = [0, 390-100, 781, 1171+100, 1562]
    tim3 = [0, 390+100, 781, 1171-100, 1562]
    tim4 = [0, 390-150, 781, 1171+150, 1562]

    pin2 = interpolate.interp1d(tim2, phs, kind='quadratic')
    pin3 = interpolate.interp1d(tim3, phs, kind='quadratic')
    pin4 = interpolate.interp1d(tim4, phs, kind='quadratic')

    xx2 = pin2(np.arange(1562))
    xx3 = pin3(np.arange(1562))
    xx4 = pin4(np.arange(1562))

    x = np.zeros((3124, 5))
    x[:, 0] = -np.cos(2*np.pi*8*time_vect)
    x[:, 1] = np.r_[np.sin(xx4), np.sin(xx4), np.sin(xx4)][1171:1171+3124]
    x[:, 2] = np.flipud(x[:, 1])
    x[:, 3] = -np.r_[np.cos(xx2), np.cos(xx2)]
    x[:, 4] = -np.r_[np.cos(xx3), np.cos(xx3)]
    return time_vect, x


def add_circles(ax, waves=False, sines=False,
                wave_len=.5, wave_height=1, offset=3):

    x_polar = np.linspace(0, 2*np.pi, 128)
    factor = np.array([.5, 1])*2
    for f in factor:
        ax.plot(f*np.sin(x_polar), f*np.cos(x_polar), 'k', linewidth=.75)
    ax.plot((-2.5, 2.5), (0, 0), 'k', linewidth=.75)
    ax.plot((0, 0), (-2.5, 2.5), 'k', linewidth=.75)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xticks([])
    ax.set_yticks([])

    tv, canonical_cycles = get_cycle_examples()
    canonical_cycles = canonical_cycles*wave_height
    tv /= 15
    tv2 = np.linspace(0, wave_len, 3124)

    if sines:
        ax.plot(tv2-wave_len/2, canonical_cycles[:, 0]-offset, 'k', linewidth=.5)
        ax.plot(tv2-wave_len/2, canonical_cycles[:, 0]+offset, 'k', linewidth=.5)
        ax.plot(tv2-offset-wave_len/2, canonical_cycles[:, 0], 'k', linewidth=.5)
        ax.plot(tv2+offset-wave_len/2, canonical_cycles[:, 0], 'k', linewidth=.5)

    if waves:
        ax.plot(tv2-wave_len/2, canonical_cycles[:, 3]-offset, 'k')
        ax.plot(tv2-wave_len/2, canonical_cycles[:, 4]+offset, 'k')
        ax.plot(tv2-offset-wave_len/2, canonical_cycles[:, 2], 'k')
        ax.plot(tv2+offset-wave_len/2, canonical_cycles[:, 1], 'k')


def normalised_waveform(infreq, sample_rate):
    sr = infreq.mean() * len(infreq)
    phase_diff = (infreq / sr) * (2 * np.pi)
    phase = np.cumsum(phase_diff, axis=0)
    phase = np.r_[0, phase]
    sine = np.sin(np.linspace(0, 2*np.pi, len(phase)))
    nw = np.sin(phase)

    return nw, sine


def scatter_kde(x, y):
    # Calculate the point density
    xy = np.vstack([x, y])
    z = stats.gaussian_kde(xy)(xy)

    # KDE colouring
    plt.scatter(x, y, c=z, s=5, edgecolor='')


# %% --------------------------------------------
# Load in analysed LFP data

emd.logger.set_up()

imf = []
pa = []
amp = []
dur = []
speed = []
p2t = []
a2d = []

for run, run_name in enumerate(config['recordings']):
    csvfile = os.path.join(config['analysisdir'], run_name + '.csv')
    df = pandas.read_csv(csvfile)
    amp.append(df['max_amp'].values)
    dur.append(df['duration_samples'].values)
    speed.append(df['speed'].values)
    p2t.append(df['peak2trough'].values)
    a2d.append(df['asc2desc'].values)

    hdffile = os.path.join(config['analysisdir'], run_name + '.hdf5')
    F = h5py.File(hdffile, 'r')
    sample_rate = 1250

    imf.append(F['imf'][...])
    pa.append(F['pa'][...])
    F.close()

    if np.isnan(speed[run]).sum() > 0:
        goods = np.where(~np.isnan(speed[run]))[0]
        pa[run] = pa[run][:, goods]
        amp[run] = amp[run][goods]
        dur[run] = dur[run][goods]
        speed[run] = speed[run][goods]

# %% ---------------------------------------------------------
# Create figure 9

plt.figure(figsize=(10, 8))
plt.axes([.1, .45, .2, .5])
linest = [':', ':', '--', '--', '-.', '-.']
for run in range(6):
    plt.plot(pa[run].mean(axis=1), color=[.8, .8, .8], linestyle=linest[run])
plt.plot(np.concatenate(pa, axis=1).mean(axis=1), 'k', linewidth=2)
plt.xlabel('Theta Phase (rads)')
plt.ylabel('Inst. Freq (Hz)')
plt.xticks(np.linspace(0, 48, 5), ['-pi', '-pi/2', '0', 'pi/2', 'pi'])
plt.xlim(0, 48)
for tag in ['top', 'right']:
    plt.gca().spines[tag].set_visible(False)
plt.grid(True)

plt.axes([.1, .1, .2, .24])
phi = np.cos(np.linspace(0, 2*np.pi, 48)) + 1j * np.sin(np.linspace(0, 2*np.pi, 48))
phi2 = 8.828 * phi
plt.plot(phi2.real, phi2.imag, 'k:')
phi1 = np.concatenate(pa, axis=1).mean(axis=1) * phi
plt.plot(phi1.real, phi1.imag, 'k')
plt.xlim(-10, 10)
plt.ylim(-10, 10)
for tag in ['top', 'right']:
    plt.gca().spines[tag].set_visible(False)
plt.plot(0, 0, 'k.')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Frequency (Hz)')
plt.grid(True)

ax = plt.axes([.35, .1, .64, .8], frameon=False)
plt.xticks([])
plt.yticks([])
mv = emd.cycles.mean_vector(np.linspace(0, 2*np.pi, 48), np.concatenate(pa, axis=1))
scatter_kde(mv.real, mv.imag)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
add_circles(plt.gca(), waves=True, wave_height=.33)
plt.plot((-0.1, 0.7), (-0.25, -0.25), 'k')
plt.plot((-0.1, 0.7), (0.25, 0.25), 'k')
plt.plot((-0.1, -0.1), (-0.25, 0.25), 'k')
plt.plot((0.7, 0.7), (-0.25, 0.25), 'k')
plt.text(-1, 0.1, '2Hz', va='bottom', ha='right', fontsize='large')
plt.text(-2, 0.1, '4Hz', va='bottom', ha='right', fontsize='large')

mm = ['o', 'o', '+', '+', '*', '*']
for run in range(6):
    mv = emd.cycles.mean_vector(np.linspace(0, 2*np.pi, 48), pa[run].mean(axis=1)[:, None])
    plt.plot(mv.real, mv.imag, mm[run], color=[0.8, .2, .2])
cbax = plt.axes([.4, .6, .01, .3])
cb = plt.colorbar(ax=ax, cax=cbax)
cb.set_label('Proportion of cycles')

ax = plt.axes([.4, 0.075, .16, .2])
add_circles(plt.gca(), waves=False, wave_height=.33)
plt.xlim(-0.1, 0.7)
plt.ylim(-0.25, 0.25)
for run in range(6):
    mv = emd.cycles.mean_vector(np.linspace(0, 2*np.pi, 48), pa[run].mean(axis=1)[:, None])
    plt.plot(mv.real, mv.imag, mm[run], color=[0.8, .2, .2])

outname = os.path.join(config['figdir'], 'emd_fig9_groupsummary.png')
plt.savefig(outname, dpi=300, transparent=False)

# Run t-tests

base = 'M={0}, SD={1}, t({2})={3}, p={4}'
tt = stats.ttest_1samp(mv.real, 0)
print('Shape space real axis - 1 sample ttest')
print(base.format(mv.real.mean(), mv.real.std(), mv.shape[0]-1, tt.statistic, tt.pvalue))

tt = stats.ttest_1samp(mv.imag, 0)
print('Shape space imaginary axis - 1 sample ttest')
print(base.format(mv.imag.mean(), mv.imag.std(), mv.shape[0]-1, tt.statistic, tt.pvalue))

# %% ---------------------------------------------
# Run PCA on phase-aligned instantaneous frequency


pc_data = np.concatenate(pa, axis=1).T
cycle_mean = pc_data.mean(axis=1)[:, None]
phase_mean = pc_data.mean(axis=0)[:, None]
pc_data = pc_data - cycle_mean

bads, _ = sails.utils.gesd(pc_data.std(axis=1))
goods = bads == False

pca = sails.utils.PCA(pc_data[goods, :], npcs=10)

pc_proj = np.zeros((48, 2, 10))
val = 15  # PC-score to project
for ii in range(10):
    sc = np.zeros((2, 10))

    sc[0, ii] = val
    sc[1, ii] = -val
    pc_proj[:, :, ii] = pca.project_score(sc).T + phase_mean


# %% ---------------------------------------------------------
# OPTIONAL - Compute split-half reproducibility of PCA

run_splits = False  # Default to off as this can take a minute or two

if run_splits:
    nsplits = 500
    half_ind = pc_data.shape[0]//2
    C = np.zeros((3, 10, nsplits))
    evr = np.zeros((2, 10, nsplits))
    for ii in range(nsplits):
        perm = np.random.permutation(pc_data.shape[0])

        p1 = sails.utils.PCA(pc_data[perm[:half_ind], :], npcs=10)
        p2 = sails.utils.PCA(pc_data[perm[-half_ind:], :], npcs=10)

        evr[0, :, ii] = p1.explained_variance_ratio
        evr[1, :, ii] = p2.explained_variance_ratio

        for jj in range(10):

            C[0, jj, ii] = np.corrcoef(pca.components[jj, :], p1.components[jj, :])[0, 1]
            C[1, jj, ii] = np.corrcoef(pca.components[jj, :], p2.components[jj, :])[0, 1]
            C[2, jj, ii] = np.corrcoef(p1.components[jj, :], p2.components[jj, :])[0, 1]
    C = np.abs(C)

    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(hspace=0.3)
    plt.subplot(211)
    h1 = plt.boxplot(evr[0, :, :].T, positions=2*np.arange(10)-0.3, patch_artist=True)
    h2 = plt.boxplot(evr[1, :, :].T, positions=2*np.arange(10)+0.3, patch_artist=True)
    plt.xlim(-1, 19)
    plt.plot((-1, 19), (0.05, 0.05), 'k:')
    plt.yticks([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    plt.xticks(np.arange(10)*2, np.arange(1, 11))
    for ii in range(len(h1['boxes'])):
        h1['boxes'][ii].set_facecolor('red')
        h2['boxes'][ii].set_facecolor('blue')
    plt.title('Variance Explained per PC per split')
    plt.ylabel('Proportion variance explained')
    plt.subplot(212)
    plt.boxplot(C[2, :, :].T, positions=2*np.arange(10))
    plt.xticks(np.arange(10)*2, np.arange(1, 11))
    plt.xlim(-1, 19)
    plt.title('Split-half correlation per PC')
    plt.ylabel('Correlation Coefficient')
    plt.xlabel('Principal Component')


    outname = os.path.join(config['figdir'], 'emd_supp2_pcavalidation.png')
    plt.savefig(outname, dpi=300, transparent=True)


# %% ---------------------------------------------------------
# Compute GLM

shape_glm_config = """
first_level:
  regressors:
    - {name: Mean,        rtype: ConstantRegressor }
    - {name: Speed,       rtype: ParametricRegressor,
                          datainfo: speed, preproc: z }
    - {name: Duration,    rtype: ParametricRegressor,
                          datainfo: duration, preproc: z }
    - {name: Amplitude ,  rtype: ParametricRegressor,
                          datainfo: amplitude, preproc: z }
  contrasts:
    - {name: Mean,         values: 1 0 0 0}
    - {name: Speed,        values: 0 1 0 0}
    - {name: Duration,     values: 0 0 1 0}
    - {name: Amplitude,    values: 0 0 0 1}
"""

glmdata = np.concatenate((pca.scores, np.hstack(p2t)[goods, None], np.hstack(a2d)[goods, None]), axis=1)
data = glm.data.TrialGLMData(data=glmdata,
                             speed=np.concatenate(speed)[goods],
                             amplitude=np.concatenate(amp)[goods],
                             duration=np.concatenate(dur)[goods])

DC = glm.design.DesignConfig(yaml_text=shape_glm_config)
des = DC.design_from_datainfo(data.info)

model = glm.fit.OLSModel(des, data)

perms = [glm.permutations.Permutation(des, data, ind, 500, metric='tstats', nprocesses=6) for ind in range(1, 4)]

tstats = model.tstats[1:, :4]
thresh = np.zeros((3, 4))  # Three predictors and four components
thresh2 = np.zeros((3, 4))  # Three predictors and four components
for ii in range(3):
    thresh[ii, :] = perms[ii].get_thresh(99)[:4]
    thresh2[ii, :] = perms[ii].get_thresh(99.9)[:4]
is_sig = np.abs(tstats) > thresh
is_sig2 = np.abs(tstats) > thresh2


# %% --------------------------------------------------------
# Create figure 10


col1 = [0.085, 0.6, 0.201]  # Green
col2 = [.6, .1, .6]  # Fuchsia

col_centre = [0.865, 0.865, 0.865]  # Grey

R = np.interp(np.linspace(-1, 1), [-1, 0, 1], [col1[0], col_centre[0], col2[0]])
G = np.interp(np.linspace(-1, 1), [-1, 0, 1], [col1[1], col_centre[1], col2[1]])
B = np.interp(np.linspace(-1, 1), [-1, 0, 1], [col1[2], col_centre[2], col2[2]])
A = np.ones_like(R)

GrFu = ListedColormap(np.c_[R, G, B, A], name='GrFu')
cmap = GrFu(np.linspace(0, 1, 128))
barcol = cm.Set1(np.linspace(0, 1, 5))

plt.figure(figsize=(10, 10))
plt.subplots_adjust(top=0.95, right=0.95, hspace=0.5, wspace=0.7)
for ii in range(4):
    plt.subplot(5, 4, ii+1)
    plt.plot(np.sin(np.linspace(0, 2*np.pi)), 'k:')
    sr = pc_proj[:, 0, ii].mean() * 49
    phase = emd.spectra.phase_from_freq(pc_proj[:, 0, ii], sr, phase_start=0)
    plt.plot(np.sin(phase), color=col1, linewidth=2)
    sr = pc_proj[:, 1, ii].mean() * 49
    phase = emd.spectra.phase_from_freq(pc_proj[:, 1, ii], sr, phase_start=0)
    plt.plot(np.sin(phase), color=col2, linewidth=2)
    for tag in ['top', 'right']:
        plt.gca().spines[tag].set_visible(False)
    plt.gca().spines[tag].set_bounds(-1, 1)
    if ii == 0:
        plt.ylabel('Normalised\nWaveform')
    plt.xticks(np.linspace(0, 48, 3), np.linspace(0, 1, 3))
    plt.title('PC: {0} ({1}%)'.format(ii+1, np.round(pca.explained_variance_ratio[ii]*100, 2)))

    plt.subplot(5, 4, ii+5)
    plt.plot(pca.components[ii, :], 'k', linewidth=1.5)
    plt.ylim(-.225, .225)
    for tag in ['top', 'right']:
        plt.gca().spines[tag].set_visible(False)
    if ii == 0:
        plt.ylabel('PC Component')
    plt.xticks(np.linspace(0, 48, 5), ['-pi', '-pi/2', '0', 'pi/2', 'pi'])

    plt.subplot(5, 4, ii+9)
    plt.plot(pc_proj[:, 0, ii], color=col1, linewidth=2)
    plt.plot(pc_proj[:, 1, ii], color=col2, linewidth=2)
    for tag in ['top', 'right']:
        plt.gca().spines[tag].set_visible(False)
    plt.ylim(3, 14)
    plt.yticks(np.arange(4, 14, 2))
    if ii == 0:
        plt.ylabel('Inst. Frequency (Hz)')
    plt.xticks(np.linspace(0, 48, 5), ['-pi', '-pi/2', '0', 'pi/2', 'pi'])

    plt.subplot(5, 4, ii+13)
    d = [data.data[pca.scores[:, ii] > 0, 10], data.data[pca.scores[:, ii] < 0, 10],
         data.data[pca.scores[:, ii] > 0, 11], data.data[pca.scores[:, ii] < 0, 11]]
    h1 = np.histogram(d[0], np.linspace(0, 1))
    h2 = np.histogram(d[1], np.linspace(0, 1))

    plt.barh(h1[1][:-1] + np.abs(np.diff(h1[1]))/2, h1[0]/2, align='center', height=0.07231023, color=col1)
    plt.barh(h2[1][:-1] + np.abs(np.diff(h2[1]))/2, -h2[0]/2, align='center', height=0.07231023, color=col2)

    h1 = np.histogram(d[2], np.linspace(0, 1))
    h2 = np.histogram(d[3], np.linspace(0, 1))

    plt.barh(h1[1][:-1] + np.abs(np.diff(h1[1]))/2, h1[0]/2, align='center', height=0.07231023, left=4000, color=col1)
    plt.barh(h2[1][:-1] + np.abs(np.diff(h2[1]))/2, -h2[0]/2, align='center', height=0.07231023, left=4000, color=col2)
    plt.ylim(.25, .75)
    plt.yticks(np.arange(0.3, 0.8, 0.1))
    plt.plot(plt.gca().get_xlim(), [0.5, 0.5], 'k:')

    plt.xticks([0, 4000], ['P2T', 'A2D'])
    for tag in ['top', 'right', 'bottom']:
        plt.gca().spines[tag].set_visible(False)
    if ii == 0:
        plt.ylabel('Control Point\nRatios')

    plt.subplot(5, 4, ii+17)
    h = plt.bar(np.arange(3), model.tstats[1:, ii], color=barcol[:4, :])
    plt.xticks(np.arange(3), model.contrast_names[1:], rotation=45, ha="right")
    for tag in ['top', 'right']:
        plt.gca().spines[tag].set_visible(False)
    if ii == 0:
        plt.ylabel('T-stats')
    yl = plt.ylim()
    yy = np.max([yl[1], 4])
    for jj in range(3):
        if is_sig[jj, ii]:
            plt.plot(jj, yy*1.1, '*', color=barcol[jj, :])
    plt.ylim(yl[0], yy*1.5)

outname = os.path.join(config['figdir'], 'emd_fig10_pcaglm.png')
plt.savefig(outname, dpi=300, transparent=True)
