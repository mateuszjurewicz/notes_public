Useful [cheetsheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf).

# Conda Environments

To create a new conda environment with a specific name and python installation:

`conda create --name your_env_name python=3.6.1`

Get a list of all available conda environments, the currently active being marked with `*`:

`conda env list`

To activate a chosen environment (on unix):

`source activate your_env_name`

Show packages installed in the conda environment (pip freeze doesn't refer to the active environment sometimes):

`conda list`

To install packages into the conda environment:

`conda install your_package_name`

To look for packages available via conda distributions. This is where conda failed me as it has no torch-geometric version.

`conda search your_package_name`
