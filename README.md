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

## Setup database

install docker

install postgres docker

create network

```bash
    docker network create db
```

create postgres container

```bash
docker run -d --name pgadmin --network=db -p 80:80 -e PGADMIN_DEFAULT_EMAIL={YOUR_USER} -e PGADMIN_DEFAULT_PASSWORD={YOUR_PASSWORD} dpage/pgadmin4
```

run psql

```bash
docker run -it --rm --network=db postgres psql -h locus-db -U postgres

```

run database

```bash
docker run --name locus-db -p 5432:5432 --network=db -v "$PWD:/var/lib/postgresql/data" -e POSTGRES_PASSWORD={YOUR_PASSWORD} -d postgres
```








## Project structure

The project structure can be seen in the [structure.md](/docs/markdown/structure.md) file.


```
