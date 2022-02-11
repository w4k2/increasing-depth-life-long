#! /bin/bash



python train_avalanche.py --experiment="TinyImageNet" --method="lwf" --dataset="tiny-imagenet" --n_experiences=20 --train_on_experiences=20 --device="cuda:0" --batch_size=128 --num_workers=10 --seed=43 --n_epochs=20 --lr=0.01 --weight_decay=1e-05 
