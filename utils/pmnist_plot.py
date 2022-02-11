import pathlib
import matplotlib.pyplot as plt


def main():
    experiment_id = 2
    runs = {
        'upper': 'b16331fd04b740bebd6a72767e3a3e8e',
        'pnn': '4d145418eb9d4c6eaf0a305fbe09c479',
        'ewc': '1e53d7c3ca474547811750dc897c0a2e',
        'ours': '20bb9712c0f64ea588e65906634b4297',
        'replay': '3002fc6aa16849788ffd7d5683357403',
        'a-gem': '20ef4f666176469fb97a577563bf03f2',
        'lwf': '8fc98af384714e3ba611e44491aff9ab',
    }

    plt.figure(figsize=(10, 4))

    for method, color in zip(('upper', 'pnn', 'ewc', 'replay', 'a-gem', 'lwf', 'ours'), ('gray', 'g', 'm', 'b', 'cyan', 'lime', 'r')):
        acc = get_average_accuracy(experiment_id, runs[method])
        plt.plot(acc, color, label=method)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('tasks')
    plt.ylabel('avrg acc')
    # plt.ylim((0.9, 1.0))
    plt.show()


def get_average_accuracy(experiment_id, run_id):
    metrics_path = pathlib.Path(f'../mlruns/{experiment_id}/{run_id}/metrics/')

    averaged_acc = []

    for task_id in range(50):
        accs = []
        for i in range(task_id, -1, -1):
            acc = load_metric(metrics_path / f'test_accuracy_task_{i}')
            acc = acc[-50+i:]
            accs.append(acc[task_id-i])
        avrg = sum(accs) / len(accs)
        averaged_acc.append(avrg)

    return averaged_acc


def load_metric(path):
    values = []
    with open(path, 'r') as f:
        for line in f.readlines():
            _, acc, _ = line.split()
            acc = float(acc)
            values.append(acc)
    return values


if __name__ == '__main__':
    main()
