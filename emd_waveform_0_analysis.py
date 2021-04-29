#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

#%% -----------------------------------------------------
#
# This script runs the main analysis used on the six LFP recordings. Each
# datafile is loaded and a mask sift and frequency transform are computed.
# Single cycles are identified in the theta IMF and a range of cycle metrics
# are computed. A subset of cycles are then identified for inclusion in further
# analysis. Finally the zero-crossing aligned temporal waveform and
# instantaneous frequency are saved for each cycle.


#%% -----------------------------------------------------
# Imports and definitions


import os
import emd
import h5py
import logging
import numpy as np

from emd_waveform_utils import config, load_dataset


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


def compute_range(x):
    return x.max() - x.min()


#%% ----------------------------------------------------
# Main loop


for run, run_name in enumerate(config['recordings']):

    logfile = os.path.join(config['analysisdir'], run_name+'.log')
    emd.logger.set_up(prefix=run_name, log_file=logfile)
    logger = logging.getLogger('emd')

    logger.info('STARTING: {0}'.format(run_name))

    raw, speed, time, sample_rate = load_dataset(run_name)

    # Load sift specification
    conf_file = os.path.join(config['basedir'], 'emd_masksift_CA1_config.yml')
    sift_config = emd.sift.SiftConfig.from_yaml_file(conf_file)

    # Run sift
    imf, mf = emd.sift.mask_sift(raw, **sift_config)

    # Frequency Transform
    IP, IF, IA = emd.spectra.frequency_transform(imf, sample_rate, 'hilbert', smooth_phase=3)

    # Compute cycle statistics
    C = emd.cycles.Cycles(IP[:, 5])
    C.compute_cycle_metric('start_sample', np.arange(len(C.cycle_vect)), emd.cycles.cf_start_value)
    C.compute_cycle_metric('stop_sample', imf[:, 5], emd.cycles.cf_end_value)
    C.compute_cycle_metric('peak_sample', imf[:, 5], emd.cycles.cf_peak_sample)
    C.compute_cycle_metric('desc_sample', imf[:, 5], emd.cycles.cf_descending_zero_sample)
    C.compute_cycle_metric('trough_sample', imf[:, 5], emd.cycles.cf_trough_sample)
    C.compute_cycle_metric('duration_samples', imf[:, 5], len)

    C.compute_cycle_metric('max_amp', IA[:, 5], np.max)
    C.compute_cycle_metric('mean_if', IF[:, 5], np.mean)
    C.compute_cycle_metric('range_if', IF[:, 5], compute_range)
    C.compute_cycle_metric('speed', speed, np.mean)
    C.compute_cycle_metric('acc', np.r_[0, np.diff(speed)], np.mean)

    C.compute_cycle_metric('asc2desc', imf[:, 5], asc2desc)
    C.compute_cycle_metric('peak2trough', imf[:, 5], peak2trough)

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
    df = C.get_metric_dataframe(subset=True)

    # phase-aligned waveforms
    pa, phasex = emd.cycles.phase_align(IP[:, 5], IF[:, 5], C.iterate(through='subset'))

    # Compute normalised waveforms
    norm_waveform, sine = emd.cycles.normalised_waveform(pa)

    # ZC-aligned waveforms
    zc_waveform = np.zeros((313, pa.shape[1]))*np.nan
    zc_instfreq = np.zeros((313, pa.shape[1]))*np.nan
    for ii, inds in C.iterate(through='subset'):
        zc_waveform[:len(inds), ii] = imf[inds, 5]
        zc_instfreq[:len(inds), ii] = IF[inds, 5]

    # Save output
    outfile = os.path.join(config['analysisdir'], run_name + '.csv')
    logger.info('Saving cycle-stats to: {0}'.format(outfile))
    df.to_csv(outfile)

    outfile = os.path.join(config['analysisdir'], run_name + '.hdf5')
    logger.info('Saving time-series outputs to: {0}'.format(outfile))
    out = h5py.File(outfile, 'w')

    to_save = ['imf', 'IP', 'IF', 'IA', 'speed',
               'pa', 'norm_waveform', 'zc_waveform', 'zc_instfreq']
    for key in to_save:
        out.create_dataset(key, data=locals()[key])
    out.close()

    logger.info('Processing Completed')
