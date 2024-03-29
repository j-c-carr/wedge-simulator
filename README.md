# Wedge Simulator
Code to simulate foreground wedge contamination on 21cmFAST lightcones/coeval boxes


### Example
![Alt Text](https://media.giphy.com/media/nZaB6NvjaESFCriCbt/giphy.gif)

### Quick start
Requirements:
* `python 3.6+`
* `numpy`
* `21cmfast`
* `tqdm`
* `scipy`
* `yaml`
* `h5py`
* `matplotlib` (optional, not required for generating datasets)

A sample exectution of the program can be found in `coeval_wedge_simulator.sh`.
For information about how to run the dataset generators, run
```
python3 wedge-simulator/coeval-simulator/bin/main.py -h
usage: main.py [-h] [--make_coeval_dset] dset_dir dset_name config_file

positional arguments:
  dset_dir            parent directory of dataset
  dset_name           name of dataset. The dataset will be saved to
                      <dset_dir>/<dset_name>.h5
  config_file         filepath to .yml configuration file

optional arguments:
  -h, --help          show this help message and exit
  --make_coeval_dset  generate coeval boxes dataset
```
All `21cmfast` parameters are specified in the `config_file` (see `coeval-simulator/in/` or `lightcone-simulator/in/` for examples). The `dset_dir` should be located somehwere with a lot of memory (`21cmFAST` will store and retrieve information from there).

#### coeval-simulator
Generates a dataset consisting of `21cmFAST` coeval boxes (brightness temperature data and ionized boxes) and the corresponding wedge-filtered boxes. The wedge filtering algorithm is described in [Gagnon-Hartman et. al, 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.504.4716G/abstract). A more elaborate outline of this code can be found on the [wiki](https://github.com/j-c-carr/wedge-simulator/wiki/coeval-simulator-workflow)

#### lightcone-simulator
Generates a dataset consisting of `21cmFAST` lightcones (brightness temperature data and ionized boxes) and the corresponding wedge-filtered lightcones. The wedge filtering algorithm is described in [Prelogović et. al, 2021](https://ui.adsabs.harvard.edu/abs/2021arXiv210700018P/abstract)
