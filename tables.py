import mlflow
import tabulate



def main():
    method_runs = {
        'upper': [None, 'd1dc183492d5425792f14d17d3a0e615'],
        'ewc': ['1e53d7c3ca474547811750dc897c0a2e', 'd5b8a1caf38740e6bcd01fb07fcd31c5'],
        'replay': [None, '5767d9cacf734542bcb97c8dc8dffde1'],
        'a-gem': ['20ef4f666176469fb97a577563bf03f2', 'fe17f4ae261d42b6ae37968bb02345fd'],
        'pnn': ['4d145418eb9d4c6eaf0a305fbe09c479', 'b384bcde400643a8a6505ffd9e41a116'],
        'lwf': [None, '1210e3abce85484b908a465efcf69904'],
        'ours': [None, None],
    }

    table = [['method', 'acc', 'forgetting', 'acc', 'forgetting']]
    for name, runs in method_runs.items():
        row = list()
        row.append(name)
        acc, forgetting = get_metrics(runs[0], experiment_id='2')
        row.append(acc)
        row.append(forgetting)
        acc, forgetting = get_metrics(runs[1], experiment_id='4')
        row.append(acc)
        row.append(forgetting)
        table.append(row)

    tab = tabulate.tabulate(table)
    print(tab)
    print("\n\n")

    tab_latex = tabulate.tabulate(table, tablefmt="latex")
    print(tab_latex)


def get_metrics(run_id, experiment_id):
    if run_id is None:
        return None, None
    
    client = mlflow.tracking.MlflowClient('///home/pwr/Documents/stochastic-depth-v2/stochastic-depth-data-streams/mlruns/')
    run = client.get_run(run_id)
    run_data = run.data
    run_metrics = run_data.metrics
    acc = run_metrics['avrg_test_acc']
    forgetting = None
    return acc, forgetting

if __name__ == "__main__":
    main()