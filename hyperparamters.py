from train_avalanche import *
import mlflow
import distutils.util


def main():
    args = parse_args()
    args.nested_run = True
    args.debug = False
    args.train_on_experiences = 3

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(args.experiment)
    if experiment is None:
        id = mlflow.create_experiment(args.experiment)
        experiment = client.get_experiment(id)
    experiment_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=experiment_id, run_name=f'{args.method} hyperparameters'):
        active_run = mlflow.active_run()
        parrent_run_id = active_run.info.run_id
        grid_search(args)

    n_repeats = 5
    args = select_best_paramters(args, client, experiment, parrent_run_id)
    args.train_on_experiences = args.n_experiences
    args.forgetting_stopping_threshold = 1.0

    with mlflow.start_run(experiment_id=experiment_id, run_name=f'{args.method} final'):
        for repeat in range(n_repeats):
            args.run_name = f'{args.method} final run {repeat}'
            args.seed += 1
            run_experiment(args)


def grid_search(args):
    for lr in (0.01, 0.001, 0.0008):
        for n_epochs in (1, 5, 10, 20):
            for weight_decay in (1e-4, 1e-5, 1e-6):
                print(f'{args.method}, lr = {lr}, n_epochs = {n_epochs}, weight_decay = {weight_decay}')
                args.run_name = f'{args.method}, lr={lr}, n_epochs={n_epochs}, weight_decay={weight_decay}'
                args.lr = lr
                args.n_epochs = n_epochs
                args.weight_decay = weight_decay

                run_experiment(args)


def select_best_paramters(args, client, experiment, parrent_run_id):
    experiment_id = experiment.experiment_id
    best_run = select_best(client, parrent_run_id, experiment_id, method='avrg_acc')
    best_parameters = best_run.data.params

    arg_names = list(vars(args).keys())
    for name in arg_names:
        value = best_parameters[name]
        arg_type = type(getattr(args, name))
        if arg_type == int:
            value = int(value)
        elif arg_type == float:
            value = float(value)
        elif arg_type == bool:
            value = distutils.util.strtobool(value)
        else:
            value = arg_type(value)
        setattr(args, name, value)

    print('\nbest args')
    for name, value in vars(args).items():
        print(f'\t{name}: {value}, type = {type(value)}')

    return args


def select_best(client, parrent_run_id, experiment_id, method='avrg_acc'):
    selected_runs = select_runs_with_parent(client, parrent_run_id, experiment_id)

    best_run = None
    if method == 'first_task':
        best_first_task_acc = 0.0
        for run in selected_runs:
            run_metrics = run.data.metrics
            first_task_acc = run_metrics['test_accuracy_task_0']
            if first_task_acc > best_first_task_acc:
                best_first_task_acc = first_task_acc
                best_run = run
    elif method == 'avrg_acc':
        best_avrg_acc = 0.0
        for run in selected_runs:
            run_metrics = run.data.metrics
            avrg_acc = run_metrics['avrg_test_acc']
            if avrg_acc > best_avrg_acc:
                best_avrg_acc = avrg_acc
                best_run = run
    else:
        raise ValueError('Invalid method argument value in select_best call')

    return best_run


def select_runs_with_parent(client, parrent_run_id, experiemnt_id):
    run_infos = client.list_run_infos(str(experiemnt_id))

    selected_runs = []
    for run_info in run_infos:
        run_id = run_info.run_id
        run = client.get_run(run_id)
        run_data = run.data
        if 'mlflow.parentRunId' not in run_data.tags:
            continue
        parent_run = run_data.tags['mlflow.parentRunId']
        if parent_run != parrent_run_id:
            continue
        selected_runs.append(run)

    return selected_runs


if __name__ == '__main__':
    main()
