# HapkeRT

A Python package for Hapke radiative transfer calculations. This is a work in progress!

## Setting up a development environment

Some may prefer to use [Conda](https://docs.conda.io/projects/conda/en/stable/#), but for a straight install you can just do

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

This is enough to work with this project directly, but if you want to run the example notebook you'll need the relevant packages. For example,

```bash
pip install jupyterlab
```

### Building a wheel or sdist

To build a [redistributable package](https://packaging.python.org/en/latest/discussions/package-formats/) (e.g. if you want to upload this to [pypi](https://pypi.org/)), you'll want the following installed:

```bash
pip install build
```

To actually produce an sdist and wheel, run:

```bash
python -m build
```

The results will be in the `dist/` directory.