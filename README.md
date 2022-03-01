# EPFL-SKA-SDC2
Code for the SKA Science Data Challenge #2

## Installation
This pipeline requires python 3. The required python libraries can be installed the `setup.sh` script:

```bash
source setup.sh
```

## Structure
<p align="center">
  <img src="https://github.com/etolley/EPFL-SKA-SDC2/blob/dev/doc/pipeline.png" width="450" title="full data processing pipeline">
</p>
The data analysis is done in several steps, as shown in the image above. Each data analysis chain is run by a pipeline script in the `pipelines` directory. The pipelines chain together different modules to read the domain, calculate noise, remove noise, etc. These different reside in the `modules` directory. 

The `utils` directory contains useful python scripts for examining different files and making plots.

## Running
To run an example pipeline, run
`python pipeline.py`

Several pipelines can be run with MPI to automatically distribute domains across different nodes.
For example, to run `pipeline_makedataframe.py` in MPI mode, call:
```bash
sbatch batch_scripts/run_pipeline_makedataframe.sbatch
```
To adapt a pipeline to run in MPI mode one needs to set up tasks with mpi4py and define a submission script in the `batch_scripts` directory.
