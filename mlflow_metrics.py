import pathlib
import mlflow


def main():
    parent_run_id = '114226043b804273a6780f9d014f1ee7'
    client = mlflow.tracking.MlflowClient('///home/jkozal/Documents/PWr/stochastic_depth/mlruns/')

    run_infos = client.list_run_infos('2')
    best_acc = 0.0
    best_id = None

    for run_info in run_infos:
        run_id = run_info.run_id
        run = client.get_run(run_id)

        run_data = run.data
        if 'mlflow.parentRunId' not in run_data.tags:
            continue
        parent_run = run_data.tags['mlflow.parentRunId']
        if parent_run != parent_run_id:
            continue

        run_metrics = run_data.metrics
        # test_accs = [acc for name, acc in run_metrics.items() if name.startswith('test_accuracy_task_')]
        # if len(test_accs) == 0:
        #     continue
        # test_avrg_acc = sum(test_accs) / len(test_accs)

        test_avrg_acc = average_acc(run_id)
        print(f"{run_data.params['method']}, lr = {run_data.params['lr']}, n_epochs = {run_data.params['n_epochs']}, weight_decay = {run_data.params['weight_decay']} test_avrg_acc = {test_avrg_acc}")

        if test_avrg_acc > best_acc:
            best_acc = test_avrg_acc
            best_id = run_id

    print()
    best_run = client.get_run(best_id)
    run_data = best_run.data
    print(f"best: {run_data.params['method']}, lr = {run_data.params['lr']}, n_epochs = {run_data.params['n_epochs']}, weight_decay = {run_data.params['weight_decay']} test_avrg_acc = {best_acc}")


def average_acc(run_id, max_num_tasks=3):
    run_path = pathlib.Path(f'mlruns/2/{run_id}/metrics/')
    num_tasks = get_num_tasks(run_path, max_num_tasks)
    accs = []
    for i in reversed(list(range(num_tasks))):
        filepath = run_path / f'test_accuracy_task_{i}'
        with open(filepath, 'r') as f:
            file_contents = [line for line in f.readlines()]

        metrics_index = num_tasks - (i+1)
        correct_line = file_contents[metrics_index]
        value_str = correct_line.split()[-2]
        acc = float(value_str)
        accs.append(acc)

    avrg_acc = sum(accs) / len(accs)
    return avrg_acc


def get_num_tasks(run_path, max_num_tasks):
    num_tasks = 0
    for i in range(max_num_tasks):
        filepath = run_path / f'test_accuracy_task_{i}'
        if filepath.exists():
            num_tasks = i
        else:
            break
    return num_tasks


if __name__ == '__main__':
    main()
