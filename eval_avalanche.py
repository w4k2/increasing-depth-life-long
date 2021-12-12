import pathlib
import argparse
import torch

from train_avalanche import get_data, get_method, compute_conf_matrix


def main():
    args = parse_args()

    device = torch.device(args.device)
    _, test_stream = get_data(args.dataset, args.seed)
    strategy, _ = get_method(args, device, use_mlflow=False)
    strategy.model = torch.load(args.weights_path)

    res = strategy.eval(test_stream)
    print(res)

    conf_mat = compute_conf_matrix(test_stream, strategy)
    print(conf_mat)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--weights_path', type=pathlib.Path, required=True)
    parser.add_argument('--method', default='ll-stochastic-depth', choices=('baseline', 'll-stochastic-depth', 'ewc'))
    parser.add_argument('--dataset', default='cifar100', choices=('cifar100', 'cifar10', 'mnist', 'permutation-mnist'))
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=20, type=int)
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    args.n_epochs = 0
    args.debug = False
    args.entropy_threshold = 0.7
    return args


if __name__ == '__main__':
    main()
