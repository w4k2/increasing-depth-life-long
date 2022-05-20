#! /bin/bash

# NOTE: when running direct all output to log file (controll for exceptions)


# CIFAR10
# python hyperparamters.py --run_name="cumulative hyperparameters w/ pretraining" --experiment "Cifar100" --interactive_logger=0 --method="cumulative" --pretrained=1 --dataset="cifar100" --n_experiences=20 --train_on_experiences=5 --device="cuda:1" --batch_size=32 
# python hyperparamters.py --run_name="ewc hyperparameters w/ pretraining" --experiment "Cifar100" --interactive_logger=0 --method="ewc" --pretrained=1 --dataset="cifar100" --n_experiences=20 --train_on_experiences=5 --device="cuda:1" --batch_size=32 
# python hyperparamters.py --run_name="replay hyperparameters w/ pretraining" --experiment "Cifar100" --interactive_logger=0 --method="replay" --pretrained=1 --dataset="cifar100" --n_experiences=20 --train_on_experiences=5 --device="cuda:1" --batch_size=32 
# python hyperparamters.py --run_name="agem hyperparameters w/ pretraining" --experiment "Cifar100" --interactive_logger=0 --method="agem" --pretrained=1 --dataset="cifar100" --n_experiences=20 --train_on_experiences=5 --device="cuda:1" --batch_size=32 
# python hyperparamters.py --run_name="pnn hyperparameters w/ pretraining" --experiment "Cifar100" --interactive_logger=0 --method="pnn" --pretrained=1 --dataset="cifar100" --n_experiences=20 --train_on_experiences=5 --device="cuda:1" --batch_size=32 
# python hyperparamters.py --run_name="lwf hyperparameters w/ pretraining" --experiment "Cifar100" --interactive_logger=0 --method="lwf" --pretrained=1 --dataset="cifar100" --n_experiences=20 --train_on_experiences=5 --device="cuda:1" --batch_size=32 
# python hyperparamters.py --run_name="mir hyperparameters w/ pretraining" --experiment "Cifar100" --interactive_logger=0 --method="mir" --pretrained=1 --dataset="cifar100" --n_experiences=20 --train_on_experiences=5 --device="cuda:1" --batch_size=32 
# python hyperparamters.py --run_name="ll-stochastic-depth hyperparameters w/ pretraining" --experiment "Cifar100" --interactive_logger=0 --method="ll-stochastic-depth" --pretrained=1 --dataset="cifar100" --n_experiences=20 --train_on_experiences=5 --device="cuda:1" --batch_size=8 

# TinyImageNet
# python hyperparamters.py --run_name="cumulative hyperparameters w/ pretraining" --experiment "TinyImageNet" --interactive_logger=0 --method="cumulative" --pretrained=1 --dataset="tiny-imagenet" --n_experiences=20 --train_on_experiences=5 --device="cuda:1" --batch_size=32 
# python hyperparamters.py --run_name="ewc hyperparameters w/ pretraining" --experiment "TinyImageNet" --interactive_logger=0 --method="ewc" --pretrained=1 --dataset="tiny-imagenet" --n_experiences=20 --train_on_experiences=5 --device="cuda:1" --batch_size=32 
# # python hyperparamters.py --run_name="replay hyperparameters w/ pretraining" --experiment "TinyImageNet" --interactive_logger=0 --method="replay" --pretrained=1 --dataset="tiny-imagenet" --n_experiences=20 --train_on_experiences=5 --device="cuda:1" --batch_size=32 
# python hyperparamters.py --run_name="agem hyperparameters w/ pretraining" --experiment "TinyImageNet" --interactive_logger=0 --method="agem" --pretrained=1 --dataset="tiny-imagenet" --n_experiences=20 --train_on_experiences=5 --device="cuda:1" --batch_size=32 
# python hyperparamters.py --run_name="pnn hyperparameters w/ pretraining" --experiment "TinyImageNet" --interactive_logger=0 --method="pnn" --pretrained=1 --dataset="tiny-imagenet" --n_experiences=20 --train_on_experiences=5 --device="cuda:1" --batch_size=32 
# python hyperparamters.py --run_name="lwf hyperparameters w/ pretraining" --experiment "TinyImageNet" --interactive_logger=0 --method="lwf" --pretrained=1 --dataset="tiny-imagenet" --n_experiences=20 --train_on_experiences=5 --device="cuda:1" --batch_size=32 
# # python hyperparamters.py --run_name="mir hyperparameters w/ pretraining" --experiment "TinyImageNet" --interactive_logger=0 --method="mir" --pretrained=1 --dataset="tiny-imagenet" --n_experiences=20 --train_on_experiences=5 --device="cuda:1" --batch_size=32 
# python hyperparamters.py --run_name="ll-stochastic-depth hyperparameters w/ pretraining" --experiment "TinyImageNet" --interactive_logger=0 --method="ll-stochastic-depth" --pretrained=1 --dataset="tiny-imagenet" --n_experiences=20 --train_on_experiences=5 --device="cuda:1" --batch_size=8 

# PMNIST
python train_avalanche.py --run_name="cumulative hyperparameters w/ pretraining" --experiment "PermutedMNIST" --interactive_logger=0 --method="cumulative" --pretrained=1 --dataset="permutation-mnist" --n_experiences=20 --train_on_experiences=5 --device="cuda:1" --batch_size=32
python train_avalanche.py --run_name="ewc hyperparameters w/ pretraining" --experiment "PermutedMNIST" --interactive_logger=0 --method="ewc" --pretrained=1 --dataset="permutation-mnist" --n_experiences=20 --train_on_experiences=5 --device="cuda:1" --batch_size=32
# python train_avalanche.py --run_name="replay hyperparameters w/ pretraining" --experiment "PermutedMNIST" --interactive_logger=0 --method="replay" --pretrained=1 --dataset="permutation-mnist" --n_experiences=20 --train_on_experiences=5 --device="cuda:1" --batch_size=32
python train_avalanche.py --run_name="agem hyperparameters w/ pretraining" --experiment "PermutedMNIST" --interactive_logger=0 --method="agem" --pretrained=1 --dataset="permutation-mnist" --n_experiences=20 --train_on_experiences=5 --device="cuda:1" --batch_size=32
python train_avalanche.py --run_name="pnn hyperparameters w/ pretraining" --experiment "PermutedMNIST" --interactive_logger=0 --method="pnn" --pretrained=1 --dataset="permutation-mnist" --n_experiences=20 --train_on_experiences=5 --device="cuda:1" --batch_size=32
python train_avalanche.py --run_name="lwf hyperparameters w/ pretraining" --experiment "PermutedMNIST" --interactive_logger=0 --method="lwf" --pretrained=1 --dataset="permutation-mnist" --n_experiences=20 --train_on_experiences=5 --device="cuda:1" --batch_size=32
# python train_avalanche.py --run_name="mir hyperparameters w/ pretraining" --experiment "PermutedMNIST" --interactive_logger=0 --method="mir" --pretrained=1 --dataset="permutation-mnist" --n_experiences=20 --train_on_experiences=5 --device="cuda:1" --batch_size=32
python train_avalanche.py --run_name="ll-stochastic-depth hyperparameters w/ pretraining" --experiment "PermutedMNIST" --interactive_logger=0 --method="ll-stochastic-depth" --pretrained=1 --dataset="permutation-mnist" --n_experiences=20 --train_on_experiences=5 --device="cuda:1" --batch_size=8
