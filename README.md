# locus-model

LOCUS aims to automate GeoGuessr with Deep Learning. This repository contains the code for the Deep Learning model.

## Setup

### Setup environment

First create a virtual environment and activate it:

```bash
make create_environment
```

Then install the requirements:

```bash
# to install requirements
make requirements

# to install requirements and dev requirements
make dev_requirements
```

The project currently uses the pytorch nightly build for Python 3.12 compatibility. We will move over to the stable build upon full Python 3.12 support.

### Setup data

The dataset is found on (kaggle)[https://www.kaggle.com/datasets/habedi/large-dataset-of-geotagged-images]. Extract the dataset to the `data/raw/LDoGI` folder.

To preprocess the data, run:

```bash
make data
```

## Usage




## Project structure

The project structure can be seen in the [structure.md](/docs/markdown/structure.md) file.


