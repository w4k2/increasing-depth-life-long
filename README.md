# Increasing Depth of Neural Networks for Life-long Learning

Code and experiemet results for paper Increasing Depth of Neural Networks for Life-longLearning.

Implemented in Pytorch with utilization of avalanche library (https://avalanche.continualai.org/). CUDA version >= 11.0.

Repo is organized as follows:
* the most important python scripts are in root directory, along with conda env file, and some bash scripts for running multiple experiments at once.
* mlruns folder contains mlflow registry with experiements results
* models folder contain source code for architectures used in experiments
* utils folder contain scripts for plotting tables, figures and some custom code


## Environment

To acess reulsts or run experiments please create conda environment:

```
conda env create -f environment.yml
```

This command will create environment ll-increasing-depth. 
If instalation of avalanche-lib via pip was not sucessful, please install it manually:

```
conda activate ll-increasing-depth
pip install avalanche==0.2.0
```

## Experiment results

To look up experiment results please run mlflow user interface server by executing in console:

```
mlflow ui
```

Next open your browser and go to adress http://127.0.0.1:5000/ 
This will site will contain results of all experiments. They are divided into several groups based on what dataset was used.


## Reproduction

To run single experiment with specific method please run:

```
python train_avalanche.py --dataset="cifar100" --method="agem" --n_epochs=10 --lr=0.001
```

List of all parameters avaliable for customization can be obtained by runing:

```
python train_avalanche.py --help
```

Hyperparameters tuning is implemented in `hyperparamters.py`. To run this script please modify this script to specify what method should be used, what device it should run on etc. There is predefined gird of paramters implemented in this file, it can be modified if needed. When ready please run

```
python hyperparamters.py
```