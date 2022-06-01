#! /bin/bash

# NOTE: when running direct all output to log file (controll for exceptions)


# CIFAR10
# python hyperparamters.py --run_name="cumulative hyperparameters w/o pretraining" --experiment "Cifar100" --interactive_logger=0 --method="cumulative" --pretrained=0 --dataset="cifar100" --n_experiences=20 --train_on_experiences=5 --device="cuda:0" --batch_size=32 
# python hyperparamters.py --run_name="ewc hyperparameters w/o pretraining" --experiment "Cifar100" --interactive_logger=0 --method="ewc" --pretrained=0 --dataset="cifar100" --n_experiences=20 --train_on_experiences=5 --device="cuda:0" --batch_size=32 
# python hyperparamters.py --run_name="replay hyperparameters w/o pretraining" --experiment "Cifar100" --interactive_logger=0 --method="replay" --pretrained=0 --dataset="cifar100" --n_experiences=20 --train_on_experiences=5 --device="cuda:0" --batch_size=32 
# python hyperparamters.py --run_name="agem hyperparameters w/o pretraining" --experiment "Cifar100" --interactive_logger=0 --method="agem" --pretrained=0 --dataset="cifar100" --n_experiences=20 --train_on_experiences=5 --device="cuda:0" --batch_size=32 
# python hyperparamters.py --run_name="pnn hyperparameters w/o pretraining" --experiment "Cifar100" --interactive_logger=0 --method="pnn" --pretrained=0 --dataset="cifar100" --n_experiences=20 --train_on_experiences=5 --device="cuda:0" --batch_size=32 
# python hyperparamters.py --run_name="lwf hyperparameters w/o pretraining" --experiment "Cifar100" --interactive_logger=0 --method="lwf" --pretrained=0 --dataset="cifar100" --n_experiences=20 --train_on_experiences=5 --device="cuda:0" --batch_size=32 
# python hyperparamters.py --run_name="mir hyperparameters w/o pretraining" --experiment "Cifar100" --interactive_logger=0 --method="mir" --pretrained=0 --dataset="cifar100" --n_experiences=20 --train_on_experiences=5 --device="cuda:0" --batch_size=32 
# python hyperparamters.py --run_name="ll-stochastic-depth hyperparameters w/o pretraining" --experiment "Cifar100" --interactive_logger=0 --method="ll-stochastic-depth" --pretrained=0 --dataset="cifar100" --n_experiences=20 --train_on_experiences=5 --device="cuda:0" --batch_size=8 

# TinyImageNet
# python hyperparamters.py --run_name="cumulative hyperparameters w/o pretraining" --experiment "TinyImageNet" --interactive_logger=0 --method="cumulative" --pretrained=0 --dataset="tiny-imagenet" --n_experiences=20 --train_on_experiences=5 --device="cuda:0" --batch_size=32 
# python hyperparamters.py --run_name="ewc hyperparameters w/o pretraining" --experiment "TinyImageNet" --interactive_logger=0 --method="ewc" --pretrained=0 --dataset="tiny-imagenet" --n_experiences=20 --train_on_experiences=5 --device="cuda:0" --batch_size=32 
# # python hyperparamters.py --run_name="replay hyperparameters w/o pretraining" --experiment "TinyImageNet" --interactive_logger=0 --method="replay" --pretrained=0 --dataset="tiny-imagenet" --n_experiences=20 --train_on_experiences=5 --device="cuda:0" --batch_size=32 
# python hyperparamters.py --run_name="agem hyperparameters w/o pretraining" --experiment "TinyImageNet" --interactive_logger=0 --method="agem" --pretrained=0 --dataset="tiny-imagenet" --n_experiences=20 --train_on_experiences=5 --device="cuda:0" --batch_size=32 
# python hyperparamters.py --run_name="pnn hyperparameters w/o pretraining" --experiment "TinyImageNet" --interactive_logger=0 --method="pnn" --pretrained=0 --dataset="tiny-imagenet" --n_experiences=20 --train_on_experiences=5 --device="cuda:0" --batch_size=32 
# python hyperparamters.py --run_name="lwf hyperparameters w/o pretraining" --experiment "TinyImageNet" --interactive_logger=0 --method="lwf" --pretrained=0 --dataset="tiny-imagenet" --n_experiences=20 --train_on_experiences=5 --device="cuda:0" --batch_size=32 
# python hyperparamters.py --run_name="mir hyperparameters w/o pretraining" --experiment "TinyImageNet" --interactive_logger=0 --method="mir" --pretrained=0 --dataset="tiny-imagenet" --n_experiences=20 --train_on_experiences=5 --device="cuda:0" --batch_size=32 
# python hyperparamters.py --run_name="ll-stochastic-depth hyperparameters w/o pretraining" --experiment "TinyImageNet" --interactive_logger=0 --method="ll-stochastic-depth" --pretrained=0 --dataset="tiny-imagenet" --n_experiences=20 --train_on_experiences=5 --device="cuda:0" --batch_size=8 

# PMNIST
# python hyperparamters.py --run_name="baseline hyperparameters w/o pretraining" --experiment "PermutedMNIST" --interactive_logger=0 --method="baseline" --pretrained=0 --dataset="permutation-mnist" --n_experiences=20 --train_on_experiences=5 --device="cuda:0" --batch_size=32
# python hyperparamters.py --run_name="ewc hyperparameters w/o pretraining" --experiment "PermutedMNIST" --interactive_logger=0 --method="ewc" --pretrained=0 --dataset="permutation-mnist" --n_experiences=20 --train_on_experiences=5 --device="cuda:0" --batch_size=32
# # python hyperparamters.py --run_name="replay hyperparameters w/o pretraining" --experiment "PermutedMNIST" --interactive_logger=0 --method="replay" --pretrained=0 --dataset="permutation-mnist" --n_experiences=20 --train_on_experiences=5 --device="cuda:0" --batch_size=32
# python hyperparamters.py --run_name="agem hyperparameters w/o pretraining" --experiment "PermutedMNIST" --interactive_logger=0 --method="agem" --pretrained=0 --dataset="permutation-mnist" --n_experiences=20 --train_on_experiences=5 --device="cuda:0" --batch_size=32
# python hyperparamters.py --run_name="pnn hyperparameters w/o pretraining" --experiment "PermutedMNIST" --interactive_logger=0 --method="pnn" --pretrained=0 --dataset="permutation-mnist" --n_experiences=20 --train_on_experiences=5 --device="cuda:0" --batch_size=32
# python hyperparamters.py --run_name="lwf hyperparameters w/o pretraining" --experiment "PermutedMNIST" --interactive_logger=0 --method="lwf" --pretrained=0 --dataset="permutation-mnist" --n_experiences=20 --train_on_experiences=5 --device="cuda:0" --batch_size=32
# # python hyperparamters.py --run_name="mir hyperparameters w/o pretraining" --experiment "PermutedMNIST" --interactive_logger=0 --method="mir" --pretrained=0 --dataset="permutation-mnist" --n_experiences=20 --train_on_experiences=5 --device="cuda:0" --batch_size=32
# python hyperparamters.py --run_name="ll-stochastic-depth hyperparameters w/o pretraining" --experiment "PermutedMNIST" --interactive_logger=0 --method="ll-stochastic-depth" --pretrained=0 --dataset="permutation-mnist" --n_experiences=20 --train_on_experiences=5 --device="cuda:0" --batch_size=8

# CORES50
python hyperparamters.py --run_name="ewc hyperparameters w/o pretraining" --experiment "CORES" --interactive_logger=0 --method="ewc" --pretrained=0 --dataset="cores50" --n_experiences=10 --train_on_experiences=5 --device="cuda:0" --batch_size=32
python hyperparamters.py --run_name="replay hyperparameters w/o pretraining" --experiment "CORES" --interactive_logger=0 --method="replay" --pretrained=0 --dataset="cores50" --n_experiences=10 --train_on_experiences=5 --device="cuda:0" --batch_size=32
python hyperparamters.py --run_name="agem hyperparameters w/o pretraining" --experiment "CORES" --interactive_logger=0 --method="agem" --pretrained=0 --dataset="cores50" --n_experiences=10 --train_on_experiences=5 --device="cuda:0" --batch_size=32
python hyperparamters.py --run_name="pnn hyperparameters w/o pretraining" --experiment "CORES" --interactive_logger=0 --method="pnn" --pretrained=0 --dataset="cores50" --n_experiences=10 --train_on_experiences=5 --device="cuda:0" --batch_size=32
python hyperparamters.py --run_name="lwf hyperparameters w/o pretraining" --experiment "CORES" --interactive_logger=0 --method="lwf" --pretrained=0 --dataset="cores50" --n_experiences=10 --train_on_experiences=5 --device="cuda:0" --batch_size=32
# python hyperparamters.py --run_name="mir hyperparameters w/o pretraining" --experiment "CORES" --interactive_logger=0 --method="mir" --pretrained=0 --dataset="cores50" --n_experiences=10 --train_on_experiences=5 --device="cuda:0" --batch_size=32
python hyperparamters.py --run_name="ll-stochastic-depth hyperparameters w/o pretraining" --experiment "CORES" --interactive_logger=0 --method="ll-stochastic-depth" --pretrained=0 --dataset="cores50" --n_experiences=10 --train_on_experiences=5 --device="cuda:0" --batch_size=8
python hyperparamters.py --run_name="cumulative hyperparameters w/o pretraining" --experiment "CORES" --interactive_logger=0 --method="cumulative" --pretrained=0 --dataset="cores50" --n_experiences=10 --train_on_experiences=5 --device="cuda:0" --batch_size=32
