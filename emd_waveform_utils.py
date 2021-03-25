import os
import emd
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, interpolate

# Please specify the following folder paths in this dictionary.
#   figdir, datadir and analysisdir are optional. If unspecified, the code will
#   look for them within the specified basedir. If a different directory is
#   specified then it will be used.

#config = {'basedir':'/local/path/to/Quinn2021_waveform/',
config = {'basedir': '/Users/andrew/Projects/emd/waveform_methods/dist/',
          'figdir': None,
          'datadir': None,
          'analysisdir': None}

# -----------------------------------------------------------------

config['1col_width'] = 85
config['2col_width'] = 114
config['3col_width'] = 174
config['fontname'] = 'Helvetica'

config['recordings'] = ['mdm81-2311-0128_2',
                        'mdm81-2311-0128_5',
                        'mdm90-1901-0127_2',
                        'mdm90-1901-0127_5',
                        'mdm96-1806-0121_2',
                        'mdm96-1806-0121_6']

config['tetrode_inds'] = [25, 25, 9, 9, 33, 33]


def initialise():

    if os.path.isdir(config['basedir']) is False:
        msg = "Study base directory not found! {0}/".format(config['basedir'])
        msg += "\nPlease specify or check the basedir defined in the emd_waveform_utils.py"
        raise RuntimeError(msg)

    # Check figure directory is defined and make it if not
    if config['figdir'] is None:
        config['figdir'] = os.path.join(config['basedir'], 'figures')
        # Make figures directory inside basedir if undefined
        if os.path.isdir(config['figdir']) is False:
            os.mkdir(config['figdir'])
    elif os.path.isdir(config['figdir']) is False:
        # Don't just make a new directory if user has specified one
        msg = "Specified figdir directory not found! {0}/".format(config['basedir'])
        msg += "\nPlease specify or check the figdir defined in the emd_waveform_utils.py"
        raise RuntimeError(msg)

    # Check analysis directory is defined and make it if not
    if config['analysisdir'] is None:
        config['analysisdir'] = os.path.join(config['basedir'], 'analysis')
        # Make figures directory inside basedir if undefined
        if os.path.isdir(config['analysisdir']) is False:
            os.mkdir(config['analysisdir'])
    elif os.path.isdir(config['analysisdir']) is False:
        # Don't just make a new directory if user has specified one
        msg = "Specified analysisdir directory not found! {0}/".format(config['basedir'])
        msg += "\nPlease specify or check the analysisdir defined in the emd_waveform_utils.py"
        raise RuntimeError(msg)

    # Check data directory exists and data is in the right place inside.
    if config['datadir'] is None:
        config['datadir'] = os.path.join(config['basedir'], 'data')
        # Make figures directory inside basedir if undefined
        if os.path.isdir(config['datadir']) is False:
            msg = "Specified data directory not found! {0}/".format(config['basedir'])
            msg += "\nPlease specify or check the datadir defined in the emd_waveform_utils.py"
            raise RuntimeError(msg)

    for rec in config['recordings']:
        eeg = os.path.join(config['datadir'], rec[:-2], rec + '.eeg')
        D = {}
        if os.path.isfile(eeg):
            D['eeg'] = eeg
        else:
            raise RuntimeError('EEG datafile for {0} is missing! ({1})'.format(rec, eeg))

        whl = os.path.join(config['datadir'], rec[:-2], rec + '.whl')
        if os.path.isfile(whl):
            D['whl'] = whl
        else:
            raise RuntimeError('whl datafile for {0} is missing! ({1})'.format(rec, whl))
        config[rec] = D

    return config

config = initialise()


# -----------------------------------------------------------------


def load_tracking(whl_path, new_len, smoothing=1):
    """Load position data from .whl file"""
    track = np.genfromtxt( whl_path )
    track[track<0] = np.nan
    if smoothing is not None:
            track = ndimage.filters.gaussian_filter1d(track, smoothing, axis=0)

    pixels2bins = 37 / (np.nanmax(track[:,0])-np.nanmin(track[:,0])) * 1/32. * 1250
    velx = np.gradient(track[:,0])
    vely = np.gradient(track[:,1])
    speed = np.sqrt(pow(velx,2)+pow(vely,2))

    # Upsample to match LFP data
    factor = 1250/32
    f = interpolate.interp1d( np.linspace(0,track.shape[0]/factor,track.shape[0]), speed,
                       kind='nearest', bounds_error=False)
    big_speed= f( np.linspace(0,new_len/1250, new_len) )*pixels2bins

    return big_speed


def load_dataset(run_id):
    logger = logging.getLogger('emd')

    inds = np.where([r == run_id for r in config['recordings']])[0][0]

    logger.info('Loading data from: {0}'.format(config[run_id]['eeg']))
    raw = np.fromfile(config[run_id]['eeg'], dtype=np.int16).astype(float)
    raw = raw.reshape(-1,64)[:,config['tetrode_inds'][inds]]
    sample_rate = 1250
    seconds = raw.shape[0] / sample_rate
    time = np.linspace(0,seconds,raw.shape[0])
    logger.info('Loaded {0} seconds of data'.format(seconds))

    logger.info('Loading tracking from: {0}'.format(config[run_id]['whl']))
    speed = load_tracking(config[run_id]['whl'], time.shape[0], smoothing=16)

    return raw, speed, time, sample_rate
