import pathlib
import mlflow
import tabulate
import numpy as np


def main():
    method_runs = {
        'upper': (['a79786744b2c4f929ba7f3fb1a823f90', '4ee10e91092e4603b447122cf98a6334', '625cafc68b0047b2a872b72d8f0ee881', '43d3c90dac6741b08e443289eeaba697', '189701c4614341e5b464b7407109e705'],
                  ['14ef7161166440029dcfbb92714ed76d', '0e40f2d1eeb34c94aa44d5922dad107f', 'fbdb8d65720d4c3daded14bcdb1ac046', '0be0838d849c4faeb3898a213f20cdfb', '8338d756b72b4868ba52cb0f4aa81bc5']),
        'ewc': (['ef709911eb6742a6805106db9cd03738', 'ca2eaf2d7c3743dcab520aba964fc05a', '9bca462deb814a76811339f90a666789', 'fd18b09b3a624e87bffca7145c8ed796', '7e8a9743f151477385a13d05e1fd770c'],
                ['7701b11e29754a51ac9a89be629afbd1', 'e8f33ec72c484bf7bafca2646b69b078', 'ac675bca906b4d90acbca607fb470cbc', '89c804ed156844f086367c59e81c0a68', 'c13fef34d9594a819c9ef9c14d6f90d1']),
        'lwf': (['570dad156738429e82ddaf70c2bd1624', '72554d211fd345e4983dc18665c7a212', '920abcab11c244a78b04dbada7447e55', '237fb22d33c44b3b9e0f3272ea99cf7b', '167d0f22e5f846fda50952aefc881d14'],
                ['d6a4414e2318479c93c719f747077e72', '31b7032a25c14c56974c9eb7b85def35', 'd5ac0cfe585145d9820a49cb74158b3b', '4f8026e289a74109aa32a1bf9e3ca990', 'e4301cc883274154aa2882004ab2971e']),
        'replay': (['6220b83cb35d4ede9af90103cbf6ac1e', '4cd1dd2ee92c4a08981e1031da1a712f', '5440c6fa05af43e08ffae88164afb2af', '4766eb469c30420586cad0bde411e0ad', '87084c0892374d9899ca3b29b203e4e3'],
                   ['efba57424fc944e099e058f87ecb3b7e', 'dfb34864597f49819e1fd77b2cbb20e7', '6d18fe1df9be458b9355dcbae35fb7de', '5f30558844e5413d8e823361dc24f03f', 'dba2fbfe8fb547668cc81e35f938b618']),
        # 'a-gem': (['f4945f31217d4e18ab6ed579f9f7a101', 'a97cde910e0e490b97afb35b2e4258be', 'e2638d237a904469b6e58bf21fbf154d', '6daaf99b3c214a0397816e2ea85a6b1b', '80664c78e67642d8b5b0a5f572fe9c64'],
        #           ['06096b1d2ca644eba7452a7026365e56', 'cf5a5d65d9c94350942e3ef9925fe3f0', '4c7dbcd0ec4e4d0fae5b9bf5cdeb72d2', '4d9c2ca21f224b8691284f7b3f186f23', 'c5079af556cc41438ac4ca8c733701bb']),
        'a-gem': (['570dad156738429e82ddaf70c2bd1624', '72554d211fd345e4983dc18665c7a212', '920abcab11c244a78b04dbada7447e55', '237fb22d33c44b3b9e0f3272ea99cf7b', '167d0f22e5f846fda50952aefc881d14'],
                  ['dc81833688bd4a1b8b10f9b840c0ac01', 'e32a6d73b43749e3a7942242bfa96787', '58802307de384fc6b7b1d3f12c2cfb64', '793b82462edf48a08716ec7955737a68', '7f0244c4ba1c4d018b1001d325d52462']),
        'pnn': (['2a4360ceb88b454fbda57244b5dd881d', 'd2d775b32ded49b9a88bcf742d186318', 'a524fc0291634859b61dd9e610a5b11b', 'bac3ef60f4734fbfa000cfaf05685c0c', 'd42fad630f6c4a969d0ad07a85e6a86e'],
                ['31290def4f464cb39bfa6ae920d89ca8', 'f8fb7ef4fd174ca8aea541c8264ca93b', '1e137217549449da8fe00ae432bf204f', '5c50c9d2c1d641b7af7ed9904d05cc78', 'b7e15c740dc64694b98ac93ea0e84671']),
        'ours (no pretraining)': (['53a9e79b026843cf9d5c2140e4116c89', '872da44f17074845bd0cfd1b8be8a37d', '8c6f14b52a24432db462cba37cd61de8', '0da85e4b5d144c8fb2bc421c2f24007d', 'e089e19311bd466585869006005ea0cd'],
                                  ['553a1797a7d14f78a14db4d1f684704c', '8cc7343140764aa4b0c13423fe3500c2', '2bbb95a75edf47999dbe33a03500f942', '396c4c9014cb43c9a6cd778cf2aacd77', '816ad64057884529aae8cc62829ba397']),
        'ours': (['c7e4e6d072db4aad8e626d45d7b49315', 'da20306d19934acfb8bfade31ed666c2', '7ea7edfb801e4b48a0acaee56d0c01ba', 'b255bf21c2a240be8fdf4df51062f9ef', 'd6bf3810b47e47609e01b7f17a1da7b9'],
                 ['c900ab6fa19c4b6f992b2e3c087494ba', '5bd4d794e5bb40e8a932abecbc532923', '512df383b20f47b3aa4b17cb3f851c4d', '0b87204af0834fd6ad9028300729c9aa', '5bb3d208f2c947bab385ad799ab66026']),
    }

    # client = mlflow.tracking.MlflowClient('///home/pwr/Documents/stochastic-depth-v2/stochastic-depth-data-streams/mlruns/')
    client = mlflow.tracking.MlflowClient('///home/jkozal/Documents/PWr/stochastic_depth/mlruns/')

    table = []
    for name, (cifar100_runs, tiny_imagenet_runs) in method_runs.items():
        row = list()
        row.append(name)

        accs = [get_metrics(run_id, client) for run_id in cifar100_runs]
        avrg_acc = sum(accs) / len(accs)
        acc_std = np.array(accs).std()
        row.append(f'{round(avrg_acc, 4)}±{round(acc_std, 4)}')
        fms = [calc_forgetting_measure(run_id, client, experiment_id=7) for run_id in cifar100_runs]
        avrg_fm = sum(fms) / len(fms)
        fm_std = np.array(fms).std()
        row.append(f'{round(avrg_fm, 4)}±{round(fm_std, 4)}')

        accs = [get_metrics(run_id, client) for run_id in tiny_imagenet_runs]
        avrg_acc = sum(accs) / len(accs)
        acc_std = np.array(accs).std()
        row.append(f'{round(avrg_acc, 4)}±{round(acc_std, 4)}')
        fms = [calc_forgetting_measure(run_id, client, experiment_id=8) for run_id in tiny_imagenet_runs]
        avrg_fm = sum(fms) / len(fms)
        fm_std = np.array(fms).std()
        row.append(f'{round(avrg_fm, 4)}±{round(fm_std, 4)}')

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
