#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

#%% -----------------------------------------------------
#
# This script runs the main analysis on 5 minutes of data from a single run.
# The analysis is repeated a specified number of time with jittered mask
# frequencies to assess the robustness of the theta waveform shape to mask
# parameter selection.

#%% -----------------------------------------------------
# Imports and definitions


import os
import emd
import h5py
import logging
import numpy as np

from emd_waveform_utils import config, load_dataset


def run_iter(raw, sample_rate, seconds, sift_config):

    try:
        # Run sift
        imf, mf = emd.sift.mask_sift(raw[:sample_rate*seconds], **sift_config)
    except EMDSiftCovergeError:
        return None

    # Frequency Transform
    IP, IF, IA = emd.spectra.frequency_transform(imf, sample_rate, 'hilbert', smooth_phase=3)

    # Compute cycle statistics - only those needed to find subset
    C = emd.cycles.Cycles(IP[:, 5])
    C.compute_cycle_metric('duration_samples', imf[:, 5], len)
    C.compute_cycle_metric('max_amp', IA[:, 5], np.max)
    C.compute_cycle_metric('speed', speed, np.mean)

    # Extract included subset of cycles
    amp_thresh = np.percentile(IA[:, 5], 25)
    lo_freq_duration = 1250/4
    hi_freq_duration = 1250/12
    conditions = ['is_good==1',
                  f'duration_samples<{lo_freq_duration}',
                  f'duration_samples>{hi_freq_duration}',
                  f'max_amp>{amp_thresh}',
                  'speed>1']

    C.pick_cycle_subset(conditions)

    # phase-aligned waveforms
    pa, phasex = emd.cycles.phase_align(IP[:, 5], IF[:, 5], C.iterate(through='subset'))

    return pa.mean(axis=1)


#%% ----------------------------------------------------
# Main loop

# Load dataset
run = 2
run_name = config['recordings'][run]

logfile = os.path.join(config['analysisdir'], run_name+'_maskjitter.log')
emd.logger.set_up(prefix=run_name, log_file=logfile)
logger = logging.getLogger('emd')

logger.info('STARTING: {0}'.format(run_name))

raw, speed, time, sample_rate = load_dataset(run_name)

# Load sift specification
conf_file = os.path.join(config['basedir'], 'emd_masksift_CA1_config.yml')
sift_config = emd.sift.SiftConfig.from_yaml_file(conf_file)
orig_masks = sift_config['mask_freqs'].copy()

# Specify number of iterations and jitter ranges
niters = 25
mask_jitters = [0.1, 0.2, 0.3]
seconds = 300


# Start main analysis
logger.info('STARTING: sift with original parameters')
pa_orig = run_iter(raw, sample_rate, seconds, sift_config)

pas = np.zeros((48, niters, len(mask_jitters)))

for ii in range(niters):
    for jj in range(len(mask_jitters)):

        logger.info('STARTING: Iteration {0} of {1} with jitter {2}'.format(ii+1, niters, mask_jitters[jj]))

        flag = True
        while flag:
            jitter = np.random.uniform(1-mask_jitters[jj], 1+mask_jitters[jj], len(orig_masks))
            sift_config['mask_freqs'] = orig_masks * jitter

            p = run_iter(raw, sample_rate, seconds, sift_config)
            if p is None:
                logger.info('Iteration failed - trying again with new masks')
                continue
            else:
                flag = False
                pas[:, ii, jj] = p

#%% ----------------------------------------------------
# Summary Figure

phasex = np.linspace(0, 2*np.pi, 48)
titles = ['Manuscript Masks', '10% Mask Jitter', '20% Mask Jitter', '30% Mask Jitter']

plt.figure(figsize=(12,6))

plt.subplot(141)
plt.plot(phasex, pa_orig, 'k', linewidth=2)
plt.ylim(7, 11)
plt.xticks(np.linspace(0, 2*np.pi, 5), ['0', 'pi/2', 'pi', '3pi/2', '2pi'])
for tag in ['top', 'right']:
    plt.gca().spines[tag].set_visible(False)
plt.title(titles[0])
plt.ylabel('Instantaneous Frequency (Hz)')

for ii in range(3):
    plt.subplot(1, 4, ii+2)
    plt.plot(phasex, pas[:,:,ii], color=[0.6, 0.6, 0.6], linewidth=0.5)
    plt.plot(phasex, pas[:,:,ii].mean(axis=1), 'k', linewidth=2)
    plt.ylim(7, 11)
    plt.gca().set_yticklabels([])
    plt.xticks(np.linspace(0, 2*np.pi, 5), ['0', 'pi/2', 'pi', '3pi/2', '2pi'])
    for tag in ['top', 'right']:
        plt.gca().spines[tag].set_visible(False)
    plt.title(titles[ii+1])
    plt.xlabel('Theta Phase (rads)')

outname = os.path.join(config['figdir'], 'emd_supp1_maskjitter.png')
plt.savefig(outname, dpi=300, transparent=True)
