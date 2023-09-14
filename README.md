# tensorflow-caney

Python package for lots of TensorFlow tools.

[![DOI](https://zenodo.org/badge/471512673.svg)](https://zenodo.org/badge/latestdoi/471512673)
[![CI Workflow](https://github.com/nasa-nccs-hpda/tensorflow-caney/actions/workflows/ci.yml/badge.svg)
![CI to DockerHub](https://github.com/nasa-nccs-hpda/tensorflow-caney/actions/workflows/dockerhub.yml/badge.svg)
![CI to DockerHub Dev](https://github.com/nasa-nccs-hpda/tensorflow-caney/actions/workflows/dockerhub-dev.yml/badge.svg)
![Code style: PEP8](https://github.com/nasa-nccs-hpda/tensorflow-caney/actions/workflows/lint.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Coverage Status](https://coveralls.io/repos/github/nasa-nccs-hpda/tensorflow-caney/badge.svg?branch=main)](https://coveralls.io/github/nasa-nccs-hpda/tensorflow-caney?branch=main)

## Documentation

- Latest: https://nasa-nccs-hpda.github.io/tensorflow-caney

## Objectives

- Library to process remote sensing imagery using GPU and CPU parallelization.
- Machine Learning and Deep Learning image classification and regression.
- Agnostic array and vector-like data structures.
- User interface environments via Notebooks for easy to use AI/ML projects.
- Example notebooks for quick AI/ML start with your own data.

## Installation

The following library is intended to be used to accelerate the development of data science products
for remote sensing satellite imagery, or any other applications. tensorflow-caney can be installed
by itself, but instructions for installing the full environments are listed under the requirements
directory so projects, examples, and notebooks can be run.

Note: PIP installations do not include CUDA libraries for GPU support. Make sure NVIDIA libraries
are installed locally in the system if not using conda/mamba.

### Production Container

```bash
module load singularity
singularity build --sandbox /lscratch/$USER/container/tensorflow-caney docker://nasanccs/tensorflow-caney:latest
```

## Development Container

```bash
module load singularity
singularity build --sandbox /lscratch/$USER/container/tensorflow-caney docker://nasanccs/tensorflow-caney:dev
```

## Why Caney?

"Caney" means longhouse in Taíno.

## Contributors

- Jordan Alexis Caraballo-Vega, jordan.a.caraballo-vega@nasa.gov
- Caleb Spradlin, caleb.s.spradlin@nasa.gov

## Contributing

Please see our [guide for contributing to tensorflow-caney](CONTRIBUTING.md).

## References

- [TensorFlow Advanced Segmentation Models](https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models)
