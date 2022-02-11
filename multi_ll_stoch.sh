#! /bin/bash

python train_avalanche.py --run_name="test" --experiment="Multi" --method="ll-stochastic-depth" --dataset="cifar10-mnist-fashion-mnist" --n_experiences=20 --train_on_experiences=20 --device="cuda:0" --batch_size=128 --num_workers=10 --seed=43 --n_epochs=10 --lr=0.01 --weight_decay=1e-05
python train_avalanche.py --run_name="test" --experiment="Multi" --method="ll-stochastic-depth" --dataset="mnist-fashion-mnist-cifar10" --n_experiences=20 --train_on_experiences=20 --device="cuda:0" --batch_size=128 --num_workers=10 --seed=43 --n_epochs=10 --lr=0.01 --weight_decay=1e-05
python train_avalanche.py --run_name="test" --experiment="Multi" --method="ll-stochastic-depth" --dataset="fashion-mnist-cifar10-mnist" --n_experiences=20 --train_on_experiences=20 --device="cuda:0" --batch_size=128 --num_workers=10 --seed=43 --n_epochs=10 --lr=0.01 --weight_decay=1e-05

python train_avalanche.py --run_name="test" --experiment="Multi" --method="ll-stochastic-depth" --dataset="cifar10-mnist-fashion-mnist" --n_experiences=20 --train_on_experiences=20 --device="cuda:0" --batch_size=128 --num_workers=10 --seed=43 --n_epochs=20 --lr=0.01 --weight_decay=1e-05
python train_avalanche.py --run_name="test" --experiment="Multi" --method="ll-stochastic-depth" --dataset="mnist-fashion-mnist-cifar10" --n_experiences=20 --train_on_experiences=20 --device="cuda:0" --batch_size=128 --num_workers=10 --seed=43 --n_epochs=20 --lr=0.01 --weight_decay=1e-05
python train_avalanche.py --run_name="test" --experiment="Multi" --method="ll-stochastic-depth" --dataset="fashion-mnist-cifar10-mnist" --n_experiences=20 --train_on_experiences=20 --device="cuda:0" --batch_size=128 --num_workers=10 --seed=43 --n_epochs=20 --lr=0.01 --weight_decay=1e-05
