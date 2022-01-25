import mlflow


def main():
    client = mlflow.tracking.MlflowClient('///home/jkozal/Documents/PWr/stochastic_depth/mlruns/')

    run_infos = client.list_run_infos('2')
    # print(run_infos)
    for run_info in run_infos:
        run_id = run_info.run_id
        print('run id = ', run_id)
        run = client.get_run(run_id)

        run_data = run.data
        print(run_data)
        print(dir(run_data))

        run_metrics = run_data.metrics
        print(run_metrics)
        # print(run)

        # test_accs = [acc for name, acc in run_metrics.items() if name.startswith('test_accuracy_task_')]
        # if len(test_accs) == 0:
        #     continue
        # test_avrg_acc = average_acc(test_accs)
        # print('test_avrg_acc = ', test_avrg_acc)
        # client.log_metric(run_id, 'avrg_test_acc', test_avrg_acc)

        metric_names = list(run_metrics.keys())
        for name in metric_names:
            if not name.startswith('test_accuracy_task_'):
                continue
            metric_history = client.get_metric_history(run_id, name)
            print(metric_history)

        break


def average_acc(metrics):
    """compute average (or cumulative accuracy)
    metrics - list or iterable of floats with accuracy values for each task
    """
    avrg_acc = sum(metrics) / len(metrics)
    return avrg_acc


if __name__ == '__main__':
    main()
