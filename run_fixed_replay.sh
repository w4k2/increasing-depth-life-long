#! /bin/bash

# with pretraining
python hyperparamters.py --run_name="replay hyperparameters w/ pretraining" --experiment "Cifar100" --interactive_logger=0 --method="replay" --pretrained=1 --dataset="cifar100" --n_experiences=20 --train_on_experiences=5 --device="cuda:0" --batch_size=32 
python hyperparamters.py --run_name="replay hyperparameters w/ pretraining" --experiment "TinyImageNet" --interactive_logger=0 --method="replay" --pretrained=1 --dataset="tiny-imagenet" --n_experiences=20 --train_on_experiences=5 --device="cuda:0" --batch_size=32 
python hyperparamters.py --run_name="replay hyperparameters w/ pretraining" --experiment "PermutedMNIST" --interactive_logger=0 --method="replay" --pretrained=1 --dataset="permutation-mnist" --n_experiences=20 --train_on_experiences=5 --device="cuda:0" --batch_size=32

# w/o pretraining
python hyperparamters.py --run_name="replay hyperparameters w/o pretraining" --experiment "Cifar100" --interactive_logger=0 --method="replay" --pretrained=0 --dataset="cifar100" --n_experiences=20 --train_on_experiences=5 --device="cuda:0" --batch_size=32 
python hyperparamters.py --run_name="replay hyperparameters w/o pretraining" --experiment "TinyImageNet" --interactive_logger=0 --method="replay" --pretrained=0 --dataset="tiny-imagenet" --n_experiences=20 --train_on_experiences=5 --device="cuda:0" --batch_size=32 
python hyperparamters.py --run_name="replay hyperparameters w/o pretraining" --experiment "PermutedMNIST" --interactive_logger=0 --method="replay" --pretrained=0 --dataset="permutation-mnist" --n_experiences=20 --train_on_experiences=5 --device="cuda:0" --batch_size=32
