# Axion model catalogues

<em><font size="4">Python code for computing model catalogues for hadronic (KSVZ) axion models.</font></em>

Developers: Sebastian Hoof, Vaisakh Plakkot\
Maintainer: [Sebastian Hoof](mailto:s.hoof.physics@gmail.com)\
License: [BSD 3-clause license](LICENSE.txt)

### Table of Contents
 - [Results](#results) - Our papers
 - [How to install](#how-to-install) - Learn how to install the code
 - [Get started](#get-started) - Learn how to use our code
 - [How to cite](#how-to-cite) - Guide to acknowledging this work in the literature

## Results

### Partial catalogue for hadronic axion models in standard cosmology
[![arxiv](https://img.shields.io/badge/arXiv-2107.12378_[hep--ph]-B31B1B.svg?style=flat&logo=arxiv&logoColor=B31B1B)](https://arxiv.org/abs/2107.12378)
[![prd](https://img.shields.io/badge/PRD-doi:10.1103/PhysRevD.104.075017-black)](https://doi.org/10.1103/PhysRevD.104.075017)
[![zenodo](https://img.shields.io/badge/Zenodo-doi:10.5281/zenodo.5091706-1682D4.svg?style=flat&logo=Zenodo&logoColor=1682D4)](https://doi.org/10.5281/zenodo.5091706)

In &ldquo;Anomaly Ratio Distributions of Hadronic Axion Models with Multiple Heavy Quarks,&rdquo; we produced the first &ldquo;complete&rdquo; catalogue of hadronic axion models based on previously proposed selection criteria.

### Cosmologically self-consistent hadronic axion models
[![arxiv](https://img.shields.io/badge/arXiv-2412.17896_[hep--ph]-B31B1B.svg?style=flat&logo=arxiv&logoColor=B31B1B)](https://arxiv.org/abs/2412.17896)
[![zenodo](https://img.shields.io/badge/Zenodo-doi:10.5281/zenodo.14524493-1682D4.svg?style=flat&logo=Zenodo&logoColor=1682D4)](https://doi.org/10.5281/zenodo.14524493)

Work in progress...

## How to install

The code is written in Python, so the scripts can be included and run after the following steps:
- Install the required Python packages: `python -m pip install h5py matplotlib numba numpy scipy sympy`.
- Install the MiMeS code, [available on Github](https://github.com/dkaramit/MiMeS/releases/tag/v1.0.0) (tested with v1.0.0).
- Adjust the `mimes_path` variable in [cosmo.py](ksvz_models/cosmo.py) to point to the MiMeS path.

## Get started

Work in progress...

## How to cite

Please cite [[arXiv:2107.12378]](https://arxiv.org/abs/2107.12378) and [[arXiv:2412.17896]](https://arxiv.org/abs/2412.17896), as well as link to this Github repo to acknoweldge our work.

You may also consider using the [BibCom tool](https://github.com/sebhoof/bibcom) to generate a list of references from the arXiv numbers or DOIs.

