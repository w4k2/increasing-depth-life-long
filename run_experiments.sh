#! /bin/bash

# python train_avalanche.py --run_name="rerun incresing depth another seed" --method="ll-stochastic-depth" --pretrained=1 --dataset="cifar100" --n_experiences=20 --batch_size=8 --n_epochs=30 --lr=0.0008 --weight_decay=0.0001 --seed=45 --prob_begin=1.0 --prob_end=0.5
# python train_avalanche.py --run_name="rerun incresing depth another seed" --method="ll-stochastic-depth" --pretrained=1 --dataset="cifar100" --n_experiences=20 --batch_size=8 --n_epochs=30 --lr=0.0008 --weight_decay=0.0001 --seed=45 --prob_begin=1.0 --prob_end=0.5
# python train_avalanche.py --run_name="rerun incresing depth another seed" --method="ll-stochastic-depth" --pretrained=1 --dataset="cifar100" --n_experiences=20 --batch_size=8 --n_epochs=30 --lr=0.0008 --weight_decay=0.0001 --seed=45 --prob_begin=1.0 --prob_end=0.5
# results: 0.791, 0.794, 0.788
# accuracy average across runs: 0.791

# python train_avalanche.py --run_name="rerun incresing depth another seed 40 epochs" --method="ll-stochastic-depth" --pretrained=1 --dataset="cifar100" --n_experiences=20 --batch_size=8 --n_epochs=40 --lr=0.0008 --weight_decay=0.0001 --seed=45 --prob_begin=1.0 --prob_end=0.5
# python train_avalanche.py --run_name="rerun incresing depth another seed 40 epochs" --method="ll-stochastic-depth" --pretrained=1 --dataset="cifar100" --n_experiences=20 --batch_size=8 --n_epochs=40 --lr=0.0008 --weight_decay=0.0001 --seed=45 --prob_begin=1.0 --prob_end=0.5
# python train_avalanche.py --run_name="rerun incresing depth another seed 40 epochs" --method="ll-stochastic-depth" --pretrained=1 --dataset="cifar100" --n_experiences=20 --batch_size=8 --n_epochs=40 --lr=0.0008 --weight_decay=0.0001 --seed=45 --prob_begin=1.0 --prob_end=0.5
# 0.788, 0.797, 0.793
# 0.7926666

# python train_avalanche.py --run_name="rerun incresing depth another seed 40 epochs lower lr" --method="ll-stochastic-depth" --pretrained=1 --dataset="cifar100" --n_experiences=20 --batch_size=8 --n_epochs=40 --lr=0.0001 --weight_decay=0.0001 --seed=45 --prob_begin=1.0 --prob_end=0.5
# python train_avalanche.py --run_name="rerun incresing depth another seed 40 epochs lower lr" --method="ll-stochastic-depth" --pretrained=1 --dataset="cifar100" --n_experiences=20 --batch_size=8 --n_epochs=40 --lr=0.0001 --weight_decay=0.0001 --seed=45 --prob_begin=1.0 --prob_end=0.5
# python train_avalanche.py --run_name="rerun incresing depth another seed 40 epochs lower lr" --method="ll-stochastic-depth" --pretrained=1 --dataset="cifar100" --n_experiences=20 --batch_size=8 --n_epochs=40 --lr=0.0001 --weight_decay=0.0001 --seed=45 --prob_begin=1.0 --prob_end=0.5
# 0.803, 0.793, 0.797
# 0.79766666666

# python train_avalanche.py --run_name="rerun incresing depth another seed big weight decay" --method="ll-stochastic-depth" --pretrained=1 --dataset="cifar100" --n_experiences=20 --batch_size=8 --n_epochs=30 --lr=0.0008 --weight_decay=0.001 --seed=45 --prob_begin=1.0 --prob_end=0.5
# python train_avalanche.py --run_name="rerun incresing depth another seed big weight decay" --method="ll-stochastic-depth" --pretrained=1 --dataset="cifar100" --n_experiences=20 --batch_size=8 --n_epochs=30 --lr=0.0008 --weight_decay=0.001 --seed=45 --prob_begin=1.0 --prob_end=0.5
# python train_avalanche.py --run_name="rerun incresing depth another seed big weight decay" --method="ll-stochastic-depth" --pretrained=1 --dataset="cifar100" --n_experiences=20 --batch_size=8 --n_epochs=30 --lr=0.0008 --weight_decay=0.001 --seed=45 --prob_begin=1.0 --prob_end=0.5
# 0.789, 0.784, 0.78
# 0.78433333

# python train_avalanche.py --run_name="rerun incresing depth another seed lower prob_begin" --method="ll-stochastic-depth" --pretrained=1 --dataset="cifar100" --n_experiences=20 --batch_size=8 --n_epochs=30 --lr=0.0008 --weight_decay=0.001 --seed=45 --prob_begin=0.8 --prob_end=0.6
# python train_avalanche.py --run_name="rerun incresing depth another seed lower prob_begin" --method="ll-stochastic-depth" --pretrained=1 --dataset="cifar100" --n_experiences=20 --batch_size=8 --n_epochs=30 --lr=0.0008 --weight_decay=0.001 --seed=45 --prob_begin=0.8 --prob_end=0.6
# python train_avalanche.py --run_name="rerun incresing depth another seed lower prob_begin" --method="ll-stochastic-depth" --pretrained=1 --dataset="cifar100" --n_experiences=20 --batch_size=8 --n_epochs=30 --lr=0.0008 --weight_decay=0.001 --seed=45 --prob_begin=0.8 --prob_end=0.6
# 0.795, 0.791, 0.797
# 0.79433333

# python train_avalanche.py --run_name="rerun incresing depth another seed 50 epochs lower lr" --method="ll-stochastic-depth" --pretrained=1 --dataset="cifar100" --n_experiences=20 --batch_size=8 --n_epochs=50 --lr=0.0001 --weight_decay=0.0001 --seed=45 --prob_begin=1.0 --prob_end=0.5
# python train_avalanche.py --run_name="rerun incresing depth another seed 50 epochs lower lr" --method="ll-stochastic-depth" --pretrained=1 --dataset="cifar100" --n_experiences=20 --batch_size=8 --n_epochs=50 --lr=0.0001 --weight_decay=0.0001 --seed=45 --prob_begin=1.0 --prob_end=0.5
# python train_avalanche.py --run_name="rerun incresing depth another seed 50 epochs lower lr" --method="ll-stochastic-depth" --pretrained=1 --dataset="cifar100" --n_experiences=20 --batch_size=8 --n_epochs=50 --lr=0.0001 --weight_decay=0.0001 --seed=45 --prob_begin=1.0 --prob_end=0.5
# 0.796, 0.8, 0.799
# 0.79833333333333