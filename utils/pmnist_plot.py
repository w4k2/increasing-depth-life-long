import pathlib
import matplotlib.pyplot as plt


def main():
    experiment_id = 2
    runs = {
        'baseline': '0bcbc3399d8646f7843773322966585a',
        'pnn': '4d145418eb9d4c6eaf0a305fbe09c479',
        'ewc': '1e53d7c3ca474547811750dc897c0a2e',
        'ours': '4655f8b07d984caf8e95bbe4ced7721b',
    }


    plt.figure(figsize=(10, 4))

    for method, color in zip(('baseline', 'pnn', 'ewc', 'ours'), ('gray', 'g', 'm', 'r')):
        acc = get_average_accuracy(experiment_id, runs[method])
        plt.plot(acc, color, label=method)

    plt.legend()
    plt.xlabel('tasks')
    plt.ylabel('avrg acc')
    plt.ylim((0.9, 1.0))
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
