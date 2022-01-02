import pathlib
import matplotlib.pyplot as plt


def main():
    experiment_id = 2
    runs = {
        'baseline': '0bcbc3399d8646f7843773322966585a',
        'ours': '4655f8b07d984caf8e95bbe4ced7721b'
    }

    baseline_acc = get_average_accuracy(experiment_id, runs['baseline'])
    ours_acc = get_average_accuracy(experiment_id, runs['ours'])

    plt.figure(figsize=(10, 4))

    plt.plot(baseline_acc, 'gray', label='baseline')
    plt.plot(ours_acc, 'r', label='ours')

    plt.legend()
    plt.xlabel('tasks')
    plt.ylabel('avrg acc')
    plt.show()


def get_average_accuracy(experiment_id, run_id):
    metrics_path = pathlib.Path(f'./mlruns/{experiment_id}/{run_id}/metrics/')

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
