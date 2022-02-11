#! /bin/bash



python train_avalanche.py --run_name="lwf final" --experiment="TinyImageNet" --method="lwf" --dataset="tiny-imagenet" --n_experiences=20 --train_on_experiences=20 --device="cuda:0" --batch_size=128 --num_workers=10 --seed=43 --n_epochs=20 --lr=0.01 --weight_decay=1e-05 
python train_avalanche.py --run_name="ewc final" --experiment="TinyImageNet" --method="ewc" --dataset="tiny-imagenet" --n_experiences=20 --train_on_experiences=20 --device="cuda:0" --batch_size=128 --num_workers=10 --seed=43 --n_epochs=5 --lr=0.01 --weight_decay=0.0001
python train_avalanche.py --run_name="ll-stoch-depth final" --experiment="TinyImageNet" --method="ll-stochastic-depth" --dataset="tiny-imagenet" --n_experiences=20 --train_on_experiences=20 --device="cuda:0" --batch_size=128 --num_workers=10 --seed=43 --n_epochs=20 --lr=0.01 --weight_decay=1e-05 
python train_avalanche.py --run_name="pnn final" --experiment="TinyImageNet" --method="pnn" --dataset="tiny-imagenet" --n_experiences=20 --train_on_experiences=20 --device="cuda:0" --batch_size=128 --num_workers=10 --seed=43 --n_epochs=20 --lr=0.01 --weight_decay=1e-06 --image_size=32
python train_avalanche.py --run_name="agem final" --experiment="TinyImageNet" --method="agem" --dataset="tiny-imagenet" --n_experiences=20 --train_on_experiences=20 --device="cuda:0" --batch_size=10 --num_workers=10 --seed=43 --n_epochs=5 --lr=0.0008 --weight_decay=1e-05
python train_avalanche.py --run_name="cumulative final" --experiment="TinyImageNet" --method="cumulative" --dataset="tiny-imagenet" --n_experiences=20 --train_on_experiences=20 --device="cuda:0" --batch_size=128 --num_workers=10 --seed=43 --n_epochs=10 --lr=0.001 --weight_decay=1e-06
python train_avalanche.py --run_name="replay final" --experiment="TinyImageNet" --method="replay" --dataset="tiny-imagenet" --n_experiences=20 --train_on_experiences=20 --device="cuda:0" --batch_size=128 --num_workers=10 --seed=43 --n_epochs=5 --lr=0.001 --weight_decay=0.0001