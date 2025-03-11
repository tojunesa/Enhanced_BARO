# Smarter Root Cause Analysis: Enhancing BARO with Outlier Filtering and Ranking Refinement

This repo contains our enhancements on the BARO method (https://arxiv.org/abs/2405.09330) and how to evaluated it on the RCAEval dataset (https://arxiv.org/abs/2412.17015).

**Table of Contents** 
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
  * [Reproducibility](#reproducibility)
  * [Licensing](#licensing)


## Prerequisites

We recommend using machines equipped with at least 8 cores, 16GB RAM, and ~50GB available disk space with Ubuntu 22.04 or Ubuntu 20.04, and Python3.10 or above.

## Installation

The `default` environment, which is used for most methods, can be easily installed as follows. Detailed installation instructions for all methods are in [SETUP.md](docs/SETUP.md).


Open your terminal and run the following commands

```bash
sudo apt update -y
sudo apt install -y build-essential \
  libxml2 libxml2-dev zlib1g-dev \
  python3-tk graphviz
```

Clone RCAEval from GitHub

```bash
git clone https://github.com/tojunesa/Enhanced_BARO.git && cd RCAEval
```

Create virtual environment with Python 3.10 (refer [SETUP.md](docs/SETUP.md) to see how to install Python3.10 on Linux)

```bash
python3.10 -m venv env
. env/bin/activate
```

Install RCAEval using pip

```bash
pip install pip==20.0.2
pip install -e .[default]
```

Or, install RCAEval from PyPI

```bash
# Install RCAEval from PyPI
pip install pip==20.0.2
pip install RCAEval[default]
```

Test the installation

```bash
python -m pytest tests/test.py::test_basic
```

Expected output after running the above command (it takes less than 1 minute)

```bash 
$ pytest tests/test.py::test_basic
============================== test session starts ===============================
platform linux -- Python 3.10.12, pytest-7.3.1, pluggy-1.0.0
rootdir: /home/ubuntu/RCAEval
plugins: dvc-2.57.3, hydra-core-1.3.2
collected 1 item                                                                 

tests/test.py .                                                            [100%]

=============================== 1 passed in 3.16s ================================
```

## Reproducibility

We augment the `main.py` in the original RCAEval repo to assist in reproducing the results. This script can be executed using Python with the following syntax: 

```
python main.py [-h] [--dataset DATASET] [--method METHOD] [--]
```

The available options and their descriptions are as follows:

```
options:
  -h, --help            Show this help message and exit
  --dataset DATASET     Choose a dataset. Valid options:
                        [re2-ob, re2-ss, re2-tt, etc.]
  --method METHOD       Choose a method (`causalrca`, `microcause`, `e_diagnosis`, `baro`, `rcd`, `circa`, etc.)
  --outlier             Choose a outlier removal method ("adaptive_z", "iqr", "tukey", "dbscan", "bayesian", "winsorization"). Default is None.
  --ranking             Choose a ranking method ("max", "mean", "med"). Default is "max".
```

For example, in Table 8, enhanced BARO (Tukey's Fences) achieves AC@1 of 0.60, 1.00, 1.00, 0.80, 0.60, and 0.53 for CPU, MEM, DISK, SOCKET, DELAY, and LOSS on the Train Ticket dataset. To reproduce these results, you can run the following commands:

```bash
python  main.py --method baro --dataset re2-tt --outlier tukey 
```


## Licensing

This repository includes code from various sources. We have included their corresponding LICENSE into the [LICENSES](LICENSES) directory:

- **BARO**: Licensed under the [MIT License]. Original source: [BARO GitHub Repository](https://github.com/phamquiluan/baro/blob/main/LICENSE).
- **RCAEval**: Licensed under the [MIT License]. Original source: [RCAEval GitHub Repository](https://github.com/phamquiluan/RCAEval/tree/main/LICENSES).


**For the code implemented by us, we distribute them under the [MIT LICENSE](LICENSE)**.



