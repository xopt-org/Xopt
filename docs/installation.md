Installing Xopt
===============

Installing `xopt` from the `conda-forge` channel can be achieved by adding `conda-forge` to your channels with:

```shell
conda config --add channels conda-forge
```

Once the `conda-forge` channel has been enabled, `xopt` can be installed with:

```shell
conda install xopt
```

It is possible to list all of the versions of `xopt` available on your platform with:

```shell
conda search xopt --channel conda-forge
```



Developers
==========


Clone this repository:
```shell
git clone https://github.com/xopt-org/xopt.git
```

Create an environment `xopt-dev` with all the dependencies:
```shell
conda env create -f environment.yml
```


Install as editable:
```shell
conda activate xopt-dev
pip install --no-dependencies -e .
```



## Cori (NERSC) setup

```
conda install -c conda-forge xopt
```
Follow instructions to build mpi4py:
https://docs.nersc.gov/programming/high-level-environments/python/
Note that there is a bug in Jupyterhub terminals. Type:
```
module swap PrgEnv-gnu PrgEnv-gnu
```
to get the C compiler activated.
