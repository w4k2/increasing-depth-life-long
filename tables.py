import pathlib
import mlflow
import tabulate


def main():
    method_runs = {
        'upper': ['b16331fd04b740bebd6a72767e3a3e8e', 'd1dc183492d5425792f14d17d3a0e615'],
        'ewc': ['1e53d7c3ca474547811750dc897c0a2e', 'd5b8a1caf38740e6bcd01fb07fcd31c5'],
        'replay': ['3002fc6aa16849788ffd7d5683357403', '5767d9cacf734542bcb97c8dc8dffde1'],
        'a-gem': ['20ef4f666176469fb97a577563bf03f2', 'fe17f4ae261d42b6ae37968bb02345fd'],
        'pnn': ['4d145418eb9d4c6eaf0a305fbe09c479', 'b384bcde400643a8a6505ffd9e41a116'],
        'lwf': [None, '1210e3abce85484b908a465efcf69904'],
        'ours': ['20bb9712c0f64ea588e65906634b4297', '049b9d24be3c4df2b43e0a8b7aeb161b'],
    }

    # client = mlflow.tracking.MlflowClient('///home/pwr/Documents/stochastic-depth-v2/stochastic-depth-data-streams/mlruns/')
    client = mlflow.tracking.MlflowClient('///home/jkozal/Documents/PWr/stochastic_depth/mlruns/')

    table = []
    for name, (pmnist_run_id, cifar100_run_id) in method_runs.items():
        row = list()
        row.append(name)

        if pmnist_run_id is None:
            continue

        acc = get_metrics(pmnist_run_id, client)
        row.append(acc)
        fm = calc_forgetting_measure(pmnist_run_id, client, experiment_id=2)
        row.append(fm)

        acc = get_metrics(cifar100_run_id, client)
        row.append(acc)
        fm = calc_forgetting_measure(cifar100_run_id, client, experiment_id=4)
        row.append(fm)

        table.append(row)

    tab = tabulate.tabulate(table)
    print(tab)
    print("\n\n")

    tab_latex = tabulate.tabulate(table, tablefmt="latex", headers=['method', 'acc', 'FM', 'acc', 'FM'])
    print(tab_latex)


def get_metrics(run_id, client):
    run = client.get_run(run_id)
    run_metrics = run.data.metrics
    acc = run_metrics['avrg_test_acc']
    acc = round(acc, 4)
    return acc


def calc_forgetting_measure(run_id, client, experiment_id):
    run_path = pathlib.Path(f'mlruns/{experiment_id}/{run_id}/metrics/')
    run = client.get_run(run_id)
    num_tasks = run.data.params['n_experiences']
    num_tasks = int(num_tasks)

    fm = 0.0

    for task_id in range(num_tasks):
        filepath = run_path / f'test_accuracy_task_{task_id}'
        task_accs = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                acc_str = line.split()[-2]
                acc = float(acc_str)
                task_accs.append(acc)

        fm += abs(task_accs[-1] - max(task_accs))

    fm = fm / num_tasks
    fm = round(fm, 4)
    return fm


if __name__ == "__main__":
    main()
