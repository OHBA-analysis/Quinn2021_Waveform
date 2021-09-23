# Within-cycle instantaneous frequency profiles report oscillatory waveform dynamics.
This repository contains the scripts to run the simulations and analysis from:

Quinn, A. J., Lopes-dos-Santos, V., Huang, N., Liang, W.-K., Juan, C., Yeh, J.-R., Nobre, A. C., Dupret, D., & Woolrich, M. W. (2021). Within-cycle instantaneous frequency profiles report oscillatory waveform dynamics. Journal of Neurophysiology. [https://doi.org/10.1152/jn.00201.2021](https://doi.org/10.1152/jn.00201.2021)

and the preprint:

Quinn, A. J., Lopes-dos-Santos, V., Huang, N., Liang, W.-K., Juan, C.-H., Yeh, J.-R., Nobre, A. C., Dupret, D., & Woolrich, M. W. (2021). Within-cycle instantaneous frequency profiles report oscillatory waveform dynamics. Cold Spring Harbor Laboratory. [https://doi.org/10.1101/2021.04.12.439547](https://doi.org/10.1101/2021.04.12.439547)


## Requirements

The LFP data used in this analysis can be freely downloaded from the [MRC BNDU data portal](https://data.mrc.ox.ac.uk/data-set/instantaneous-frequency-profiles-theta-cycles) (requires free registration).

The original analysis used Python 3.7.6 but should run ok in more recent versions. The package requirements to exactly reproduce the the original analysis are detailed in the `requirements.txt` file in the repository. These can be installed using pip:

```
pip install -r requirements.txt
```

or by creating a specific anaconda environment:

```
conda create --name emd_waveform --python 3.7.6 --file requirements.txt
```

## Getting started
Once you have downloaded the data and have a Python environment with the required dependencies installed, you can begin the analyses.

1. Open `emd_waveform_utils.py` and specify the filepaths at the top. `basedir` must be specified but the rest are optional - see the script for more details.
2. Ensure that the right python environment is active.
3. Run `emd_waveform_0_analysis.py`. This will run the EMD analysis on the six datafiles and save the output in the `analysisdir` specifed in `emd_waveform_utils.py`
4. Work through the `emd_waveform_fig*py` scripts in any order. Each script reproduces one or more of the figures in the paper.
