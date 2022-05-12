import pathlib
import mlflow
import tabulate
import numpy as np


def main():
    runs_wo_pretraining = {  # columns: name, #parameters, pmnist acc, pmnist FM, cifar100 acc, cifar100 FM, TIN acc, TIN FM x2 (w/o w/ pretraining)
        'Upperbound': [
            [None, None, None, None, None],
            [None, None, None, None, None],
            [None, None, None, None, None],
        ],
        r'EWC \cite{DBLP:journals/corr/KirkpatrickPRVD16}': [
            [None, None, None, None, None],
            [None, None, None, None, None],
            [None, None, None, None, None],
        ],
        r'ER \cite{DBLP:journals/corr/abs-1902-10486}': [
            ['46aae2323aea487aa37e28dde368d2b0', 'eeabf8f4d9614edd8123ffb1751dfd63', '8bf82c9490ef48fe94c9b2577b42b0f6', '40a3970e1ec644bb90ed7a2de4df4353', 'cff2bf5764984fdb9ecfe2adcfc1b945'],
            ['24feac44583844499e9f4908835c80d5', '6c7b8a96cea3433db1dfedeaf05a859a', '739859dcecf14a2fb0790357e20e242d', '65f074de0f1a4031850c0009ef000559', 'b5d020a2ed524fec93abfc02839b2443'],
            ['9fcd5b53d7c14a98a729bd8ed69039dd', 'bc2551f188d14550b2abf7bf02555eb3', '01f10d41a97647e7af6d12d42f5e3d51', '1be6fa11d69641cebba4eb754a8cc4c4', 'a02e9f40606d4ab0b707ac6ffce41aa1'],
        ],
        r'A-GEM \cite{DBLP:journals/corr/abs-1812-00420}': [
            [None, None, None, None, None],
            [None, None, None, None, None],
            [None, None, None, None, None],
        ],
        r'PNN \cite{DBLP:journals/corr/RusuRDSKKPH16}': [
            [None, None, None, None, None],
            [None, None, None, None, None],
            [None, None, None, None, None],
        ],
        r'LWF \cite{DBLP:journals/corr/LiH16e}': [
            [None, None, None, None, None],
            [None, None, None, None, None],
            [None, None, None, None, None],
        ],
        'Ours': [
            [None, None, None, None, None],
            [None, None, None, None, None],
            [None, None, None, None, None],
        ],
    }
    runs_w_pretraining = {  # columns: name, #parameters, pmnist acc, pmnist FM, cifar100 acc, cifar100 FM, TIN acc, TIN FM x2 (w/o w/ pretraining)
        'Upperbound': [
            [None, None, None, None, None],
            [None, None, None, None, None],
            [None, None, None, None, None],
        ],
        r'EWC \cite{DBLP:journals/corr/KirkpatrickPRVD16}': [
            [None, None, None, None, None],
            [None, None, None, None, None],
            [None, None, None, None, None],
        ],
        r'ER \cite{DBLP:journals/corr/abs-1902-10486}': [
            ['7299961ddfc94f7b835c5cbaea7436d9', '6685595dc6074ed48e92ab490ce0a039', '21f469599c4e42218022ae185cb53c3e', 'b6ff39f4c6ee423ba74df902fa2c5271', 'd11dd8a241084bce97511322b7d0a741'],
            ['2f5bc0afbce04d5a8bea033d6a8ef1ba', '7985cab2d91545e0893d705da7be50e5', 'a2d2b5e72ebd4dcbbc9f2cda2f9e81d5', 'd7d92cefe8864d488ee75a543f110d86', '8a0f6b6d07fb4087a5a20b2367fec0d4'],
            [None, None, None, None, None],
        ],
        r'A-GEM \cite{DBLP:journals/corr/abs-1812-00420}': [
            [None, None, None, None, None],
            [None, None, None, None, None],
            [None, None, None, None, None],
        ],
        r'PNN \cite{DBLP:journals/corr/RusuRDSKKPH16}': [
            [None, None, None, None, None],
            [None, None, None, None, None],
            [None, None, None, None, None],
        ],
        r'LWF \cite{DBLP:journals/corr/LiH16e}': [
            [None, None, None, None, None],
            [None, None, None, None, None],
            [None, None, None, None, None],
        ],
        'Ours': [
            [None, None, None, None, None],
            [None, None, None, None, None],
            [None, None, None, None, None],
        ],
    }
    num_parameters = [0, 0, 0, 0, 0, 0, 0]

    # client = mlflow.tracking.MlflowClient('///home/pwr/Documents/stochastic-depth-v2/stochastic-depth-data-streams/mlruns/')
    client = mlflow.tracking.MlflowClient('///home/jkozal/Documents/PWr/stochastic_depth/mlruns/')

    table = []
    for num_param, (name, run_ids) in zip(num_parameters, runs_wo_pretraining.items()):
        row = get_row(client, num_param, name, run_ids)
        table.append(row)

    table.append(['\\hline'])

    for num_param, (name, run_ids) in zip(num_parameters, runs_w_pretraining.items()):
        row = get_row(client, num_param, name, run_ids)
        table.append(row)

    tab = tabulate.tabulate(table)
    print(tab)
    print("\n\n")

    tab_latex = tabulate.tabulate(table, tablefmt="latex", headers=['method', '#parameters', 'acc', 'FM', 'acc', 'FM', 'acc', 'FM', 'acc', 'FM', 'acc', 'FM', 'acc', 'FM'])
    tab_latex = tab_latex.replace('\\textbackslash{}', '\\')
    tab_latex = tab_latex.replace('\\{', '{')
    tab_latex = tab_latex.replace('\\}', '}')
    print(tab_latex)
    print("\n\n")

    mixed_runs = {
        'ER': ['b47c4e4040d84cf8986033542c74a256', '682c1cc4d0a8465a9dd54a1528186a4a'],
        'Ours': ['5833f8eb7c634bb6a8551fb1aa6d8fb7', '0f05d33f6360481eb317de1aadf8c480'],
    }

    table = []
    for name, run_ids in mixed_runs.items():
        row = list()
        row.append(name)

        for run_id in run_ids:
            acc = get_metrics(run_id, client)
            row.append(acc)
            fm = calc_forgetting_measure(run_id, client, experiment_id=6, num_tasks=3)
            row.append(fm)
        table.append(row)

    tab = tabulate.tabulate(table)
    print(tab)
    print("\n\n")

    tab_latex = tabulate.tabulate(table, tablefmt="latex", headers=['method', 'acc', 'FM', 'acc', 'FM'])
    print(tab_latex)
    print("\n\n")


def get_row(client, num_param, name, run_ids):
    row = list()
    row.append(name)
    row.append(num_param)

    for dataset_run_ids, experiment_id in zip(run_ids, (4, 1, 2)):
        avrg_acc, acc_std, avrg_fm, fm_std = calc_average_metrics(dataset_run_ids, client, experiment_id)
        row.append(f'{avrg_acc}±{acc_std}')
        row.append(f'{avrg_fm}±{fm_std}')
    return row


def calc_average_metrics(dataset_run_ids, client, experiment_id):
    if dataset_run_ids[0] == None:
        return '-', '-', '-', '-'

    acc_all = []
    fm_all = []
    for run_id in dataset_run_ids:
        acc = get_metrics(run_id, client)
        acc_all.append(acc)
        fm = calc_forgetting_measure(run_id, client, experiment_id=experiment_id)
        fm_all.append(fm)
    avrg_acc = sum(acc_all) / len(acc_all)
    avrg_acc = round(avrg_acc, 4)
    acc_std = np.array(acc_all).std()
    acc_std = round(acc_std, 4)
    avrg_fm = sum(fm_all) / len(fm_all)
    avrg_fm = round(avrg_fm, 4)
    fm_std = np.array(fm_all).std()
    fm_std = round(fm_std, 4)
    return avrg_acc, acc_std, avrg_fm, fm_std


def get_metrics(run_id, client):
    run = client.get_run(run_id)
    run_metrics = run.data.metrics
    acc = run_metrics['avrg_test_acc']
    return acc


def calc_forgetting_measure(run_id, client, experiment_id, num_tasks=None):
    run_path = pathlib.Path(f'mlruns/{experiment_id}/{run_id}/metrics/')
    if num_tasks is None:
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
    return fm


if __name__ == "__main__":
    main()
