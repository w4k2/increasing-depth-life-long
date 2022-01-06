from train_avalanche import *
import mlflow


class args:
    run_name = None
    experiment = 'PermutedMNIST'
    method = 'baseline'
    base_model = 'resnet18'
    pretrained = True
    dataset = 'permutation-mnist'
    n_experiences = 50
    device = 'cuda'
    batch_size = 128
    num_workers = 20
    seed = 42
    n_epochs = 20
    image_size = 128
    debug = False
    lr = 0.0001
    weight_decay = 1e-6
    entropy_threshold = 0.7
    nested_run = True


def main():
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(args.experiment)
    if experiment is None:
        id = mlflow.create_experiment(args.experiment)
        experiment = client.get_experiment(id)
    experiment_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=experiment_id):
        for lr in (0.01, 0.001, 0.0008):
            for n_epochs in (1, 5, 10, 20):
                for weight_decay in (1e-4, 1e-5, 1e-6):
                    args.run_name = f'lr={lr}, n_epochs={n_epochs}, weight_decay={weight_decay}'
                    args.lr = lr
                    args.n_epochs = n_epochs
                    args.weight_decay = weight_decay

                    run_experiment(args)


if __name__ == '__main__':
    main()
