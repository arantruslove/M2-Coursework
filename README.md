# M2 Coursework

Report: https://github.com/arantruslove/M2-Coursework/blob/main/report/M2_Report.pdf
This README contains instructions on how to setup and run the M2 coursework project.

## Installation

Build the Docker image from the root of the repository:

```bash
docker build -t m2 .
```

Start the Docker container and expose `port 8888` for the Jupyter notebooks.

```bash
docker run -d -p 8888:8888 -v ${PWD}:/m2 m2
```

## Notebooks

Access `localhost:8888` on a web browser for the Jupyter server containing the notebooks. The notebooks contain:

- FLOPS breakdown
- Data exploration
- Examples of conversion of numerical trajectories to string and token formats
- Generation of figures used in the report

## FLOPS

The source code used to calculate the FLOPS is located in `m2_utilities/flops.py`.

## Experiments

The code used to run the experiments is found in `experiments/` with the training file in `train.py` and the training configurations in `config.py`.

To run an experiment, first start a bash shell in the container:

```bash
docker exec -it <container_id> bash
```

Then execute this command from the root of the repository with the desired config no:

```bash
python experiments/train.py --config_no=<int>
```

This will run the training and save the models to the `models/` directory.

## Metrics

After running the experiments, the metrics of each model (saved in the `models/` directory) can be evaluated on the validation set by running:

```bash
python experiments/metrics.py --restrict_tokens=True --config_no=<int>
```

## Autogeneration Tools Declaration

ChatGPT was used to assist in the development of this project in the following ways:

1. Generation of all matplotlib figures.
2. Generation of all Google-style docstrings.
