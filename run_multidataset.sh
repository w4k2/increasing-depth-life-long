#! /bin/bash

# python train_avalanche.py --run_name="increasing depth pretraining" --experiment="Multi" --method="ll-stochastic-depth" --pretrained=1 --dataset="5-datasets" --batch_size=8 --n_epochs=50
# python train_avalanche.py --run_name="increasing depth w/o pretraining" --experiment="Multi" --method="ll-stochastic-depth" --pretrained=0 --dataset="5-datasets" --batch_size=8 --n_epochs=50
# python train_avalanche.py --run_name="replay pretraining memsize 500 lr=0.0001" --experiment="Multi" --method="replay" --pretrained=1 --dataset="5-datasets" --batch_size=32 --lr=0.0001 --weight_decay=1e-05 --n_epochs=10 --forgetting_stopping_threshold=1.0
# python train_avalanche.py --run_name="replay w/o pretraining memsize 500 lr=0.0001" --experiment="Multi" --method="replay" --pretrained=0 --dataset="5-datasets" --batch_size=32 --lr=0.0001 --weight_decay=1e-05 --n_epochs=10 --forgetting_stopping_threshold=1.0
# python train_avalanche.py --run_name="ewc w/ pretraining memsize 500 lr=0.0001" --experiment="Multi" --method="ewc" --pretrained=1 --dataset="5-datasets" --batch_size=32 --lr=0.001 --weight_decay=1e-05 --n_epochs=10 --forgetting_stopping_threshold=1.0
# python train_avalanche.py --run_name="ewc w/o pretraining memsize 500 lr=0.0001" --experiment="Multi" --method="ewc" --pretrained=0 --dataset="5-datasets" --batch_size=32 --lr=0.01 --weight_decay=0.0001 --n_epochs=50 --forgetting_stopping_threshold=1.0
# python train_avalanche.py --run_name="pnn w/o pretraining memsize 500 lr=0.0001" --experiment="Multi" --method="pnn" --pretrained=0 --dataset="5-datasets" --batch_size=32 --lr=0.001 --weight_decay=1e-06 --n_epochs=50 --forgetting_stopping_threshold=1.0
python train_avalanche.py --run_name="agem w/o pretraining memsize 500 lr=0.0001" --experiment="Multi" --method="agem" --pretrained=0 --dataset="5-datasets" --batch_size=32 --lr=0.0008 --weight_decay=1e-06 --n_epochs=30 --forgetting_stopping_threshold=1.0
python train_avalanche.py --run_name="agem w/ pretraining memsize 500 lr=0.0001" --experiment="Multi" --method="agem" --pretrained=1 --dataset="5-datasets" --batch_size=32 --lr=0.0008 --weight_decay=1e-06 --n_epochs=50 --forgetting_stopping_threshold=1.0
