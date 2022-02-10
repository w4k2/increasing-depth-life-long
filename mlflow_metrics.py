import mlflow


def main():
    parent_run_id = 'c721cb8698b94e6a8665b2215bc6919f'
    client = mlflow.tracking.MlflowClient('///home/pwr/Documents/stochastic-depth-v2/stochastic-depth-data-streams/mlruns/')

    run_infos = client.list_run_infos('4')
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

        print(run_id)
        continue

        run_metrics = run_data.metrics
        test_accs = [acc for name, acc in run_metrics.items() if name.startswith('test_accuracy_task_')]
        if len(test_accs) == 0:
            continue
        test_avrg_acc = average_acc(test_accs)
        print(f"{run_data.params['method']}, lr = {run_data.params['lr']}, n_epochs = {run_data.params['n_epochs']}, weight_decay = {run_data.params['weight_decay']} test_avrg_acc = {test_avrg_acc}")

        if test_avrg_acc > best_acc:
            best_acc = test_avrg_acc
            best_id = run_id

    # print()
    # best_run = client.get_run(best_id)
    # run_data = best_run.data
    # print(f"best: {run_data.params['method']}, lr = {run_data.params['lr']}, n_epochs = {run_data.params['n_epochs']}, weight_decay = {run_data.params['weight_decay']} test_avrg_acc = {best_acc}")


def average_acc(metrics):
    """compute average (or cumulative accuracy)
    metrics - list or iterable of floats with accuracy values for each task
    """
    avrg_acc = sum(metrics) / len(metrics)
    return avrg_acc


if __name__ == '__main__':
    main()
