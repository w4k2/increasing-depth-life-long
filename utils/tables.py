import pathlib
import mlflow
import tabulate
import numpy as np


def main():
    runs_wo_pretraining = {  # columns: name, #parameters, cifar100 acc, cifar100 FM, TIN acc, TIN FM, pmnist acc, pmnist FM
        'Upperbound': [
            ['7299df53d45f442cbd66dd74a2987044', '5e30ad287cb64572bcc7db330a026016', '68037f4b3f2d4bec870dcfcc6c0ae7c4', 'd13c12cc8aa7414f8cc30484e5495c37', 'b1d9d932d05b49aab0b8841d69add04f'],
            ['a4b88115c40044a3847c2cb1b9cd931f', '27c1bf3adf6a4d9e8781124f95935f23', 'fb1a96cec4854691aa5b027c14787b0b', '44c5d5e4152c40ac81e1a69709c1bfdf', '61dbee30d5b34c67ac1a199af85ba6e8'],
            [None, None, None, None, None],
        ],
        r'EWC \cite{DBLP:journals/corr/KirkpatrickPRVD16}': [
            ['76073c529caf428da7f3d68f649209b7', '61fbbec0858c4b9fa1438c09010a97d9', 'ee0377e0883e4a6d8f399a8e3a608d10', 'f4ae4a6986044d588b39f06297714856', '8c9dd72c9c974021abbfce3bd7b490c0'],
            ['3631e37897604ab4845c3cf9cf12b89e', 'a5f8df8c08a74b95a6cfadaee8ef1658', '3b2fd8ec871e4119906e56e0f2a151b5', '5cdd21a24caa4706a8c3fdeca64ef099', 'c92d7e11f7dc4dc6b4d3d36e4cc64b1f'],
            ['c447e17e025947c99352ccb9f6beb68f', 'b7f8b0363c2c44e8b059180c1f988c88', '2e5639de0db1497eaf3ad45983f5627f', '67effaaca59946d1a2c776fd2399bd28', '3defaca4dfc249fbaffe248ba747d990'],
        ],
        r'ER \cite{DBLP:journals/corr/abs-1902-10486}': [
            ['46aae2323aea487aa37e28dde368d2b0', 'eeabf8f4d9614edd8123ffb1751dfd63', '8bf82c9490ef48fe94c9b2577b42b0f6', '40a3970e1ec644bb90ed7a2de4df4353', 'cff2bf5764984fdb9ecfe2adcfc1b945'],
            ['24feac44583844499e9f4908835c80d5', '6c7b8a96cea3433db1dfedeaf05a859a', '739859dcecf14a2fb0790357e20e242d', '65f074de0f1a4031850c0009ef000559', 'b5d020a2ed524fec93abfc02839b2443'],
            ['0625aacb992244dab425cf7a9dd49c8a', '41388987cd22496f887f181c4788981e'],
        ],
        r'A-GEM \cite{DBLP:journals/corr/abs-1812-00420}': [
            ['0c7f7156b6044b52b7ccbbcbca59c649', '0c2d264e07dd4733b2356a424aa1c5fc', '42d13c77ba1a45b4a75f646f716ee3b6', 'cbbb676caf714e3f86a7e5e442282c40', '1f8d7cee06e247d298d1676bdcf2971a'],
            ['06cb3837219c4ae8b7da0f0e392882cd', 'def7013480ce449e9107547a39163289', '578c7dd017074da4b4e7fef482c60d35', 'd99302186bff4c27982a9899b6f0787b', 'a9711018e97f4d5f949d17c5fb2a5a1a'],
            ['4f1beafbe7024afca8e450af7162821c', 'c92b1b96ae6740cca276d43a928344de', 'accc723dd8554d5c92fb509365dd36e6', '0a4e0920917a47118dbbe952ace15cd9', '6cb6338d0c0142b282b9113af7d47063'],
        ],
        r'PNN \cite{DBLP:journals/corr/RusuRDSKKPH16}': [
            ['2de7e46a8c2443ac8587542d470d7154', '46f3944ed802435098ba8c1206bd6ffe', '3f7278e5928e47aeb02e65a8fca46de0', '78a81252967546a4ab9e282d755e7e20', '9400d9e552974abd884ff0af8f8bcb48'],
            ['98231e19162a4be49d6255eb223d0d17', 'c7e5fa154c474884ab786a6935484bff', '95765e3d72934c17868c702d2d887187', '605e92667a4b40cbb3efc2f11e22b359', '30667327abdc44bf928b78bf7ff5b0f4'],
            ['07f9612ec746475bb393a167a5ceb37b', '1ac968b8b0994c3f893f60bedede1325', 'fed9d5665d6548e69da01452221818a0', '12076875711d4d1f91d0c6f1779bcb05', '104e45fd4c244622abfb0bd084f59efc'],
        ],
        r'LWF \cite{DBLP:journals/corr/LiH16e}': [
            ['aa73b91cd0964bfab38c752bb32047c5', '67613a30419c445b862885a7903970b0', 'bda9f4f43c5a45daaae12cb7228dce43', '3a6aec25b4514510a866eaa5a6c495d9', '617278fe12ab40b2a2a42698dbea343c'],
            ['9c233fc60a334974a558d3b631de0feb', '504286b1453b44a19f83219e8d97d4b9', 'fa8f4c5357a24d498cada1c261da921b', '6858810a86094c089b63c3cb4e25463a', '0f6a32659efa48b68d57d27851fc9f4d'],
            ['b3c86633780c469cb55658f8cae0794e', 'b42a755fe901424cb8989651ab27717f', '4af991ae51ca48fb9d15cfac7dc12dc9', '2d78e40e1f294cbcb8aeb321c7a69e3e', 'a50c16e475db4e76894729e712c2ec49'],
        ],
        'Ours': [
            ['375c3e34198d42528207bf6efca58b16', '1e04d877656b4d18b123a568d24a381d', '30f5d1ef87f4430590ca4e48fc803d8d', '110cba2ebc0447b5b5294ed87ae45e7a', '2c820fda1afc4aa1965edbc7c962788e'],
            ['abaa33c8c9ce4e98bc56f65f79239732', '1cf83d2edad24c6ca1848c76b347ed50', '65358873c6394d1ba34feaa4f34a693f', 'd56a160ce9704c85a7c44d6f1837437f', '35f4d728ad554b5f9ab535152671ec27'],
            [None, None, None, None, None],
        ],
    }
    runs_w_pretraining = {  # columns: name, #parameters, cifar100 acc, cifar100 FM, TIN acc, TIN FM, pmnist acc, pmnist FM
        'Upperbound': [
            ['4b19c8293100481eb13e77cb69a74a63', '991355734d0b46018761a01a19c9e2c2', '7daa6e8f75884745a3b4dff2546425f2', 'ed645bd069b14eefb06ef60c8ceed3b5', '6123aacb805f4427a473b0c355ead066'],
            ['0e38b56d2a5e456688629df43930e5c9', 'f8e434da1c184d7eb73245307237de23', '12aa7dad46644cf28e098a1cb62cfa6a', '173e02b046eb4dddb101c699a5982dda', '1d5bb51ac3cf4bcc98b1ce009704cfea'],
            [None, None, None, None, None],
        ],
        r'EWC \cite{DBLP:journals/corr/KirkpatrickPRVD16}': [
            ['6ce642ba84484e4590458a61b73dbbe7', '0688779947fd496a890fd0f2494493e8', '81234edf47b645908d89b4a2ccc2d1b9', '4e96adca217a4c4ea1c65fa8efa12cb0', 'df785c3e21674995a531e7a17e76f53e'],
            ['57a5bf176df8483ab52a8eceaa06675c', 'adbc4d8cead44ef59d3f36db0dc11381', 'a4168977b22c4e92815fb5d213331192', 'bb68f28384b44096b19e88b72a3a50ea', '6671438931b349fa9d363cdf3053703c'],
            ['5bc961430f92499e8caa3ff08313199e', '16431d0d5a744e5a8e9b0b63373500f5', 'f38d472655864b69b0ce508f26f51158', '971bc721d2be496a981bb2daaa06317a', '2286da8b59ae4fd4ab57e0089d699657'],
        ],
        r'ER \cite{DBLP:journals/corr/abs-1902-10486}': [
            ['7299961ddfc94f7b835c5cbaea7436d9', '6685595dc6074ed48e92ab490ce0a039', '21f469599c4e42218022ae185cb53c3e', 'b6ff39f4c6ee423ba74df902fa2c5271', 'd11dd8a241084bce97511322b7d0a741'],
            ['2f5bc0afbce04d5a8bea033d6a8ef1ba', '7985cab2d91545e0893d705da7be50e5', 'a2d2b5e72ebd4dcbbc9f2cda2f9e81d5', 'd7d92cefe8864d488ee75a543f110d86', '8a0f6b6d07fb4087a5a20b2367fec0d4'],
            ['cc172da9f8b047ecabc13e3f018915fc', '4d3db647b34547db85ebab99cefc09c7'],
        ],
        r'A-GEM \cite{DBLP:journals/corr/abs-1812-00420}': [
            ['77b14fad8723461d94ce6acd5407e62a', 'bf3bbf6fb1bb4fa18264ec8066ba4653', '66ab2af44beb4aab9e664d4279304215', '4aebed10ef9a4e9cb92e3e973b4692f4', '913c72664b184b9fa165496534e721c5'],
            ['622b70f5b8084802b0c31718fd56d177', '148031b8f7104b1c90ca4bd88ea09f90', '5a6ae1b854b94e6b895e26244356f7c4', 'dadfcbf7c6be44d69ae45c9a2fa887e1', '657920bc101843da9aa1474506dff755'],
            ['a344256c53224d189977fddb8f4a664f', '6e0662889b214138b1bddd5e0f923eda', '08a32c50df4c4435b18d05f7af335cd6', '082c67356a0f447b9775b239d1e08e2f', '322eb00c0c6b425fb530735de7745e70'],
        ],
        # r'PNN \cite{DBLP:journals/corr/RusuRDSKKPH16}': [
        #     ['c405d5b2d4864260b0b06af7d8ad2357', '864affdda0824ebfb9ad219b0f8508fa', 'f4e8b6c312d448689ea3a07847fa6d74', '9578d0fd7fe14db3a58a820ed3aacae9', 'a97b539a3d3b4db783f9d143609aaf0c'],
        #     ['e41ec60f1a704236aeb43c5aba718956', 'd05fa2da9a9a46ef9cf97069a4507ba4', '3c8c8401983d468fb3b7654ba50ffe15', 'c03e0c1ad9b4449ebe958e012329105b', 'cebfa3fe1d9a4e77867c32cd5e216a83'],
        #     [None, None, None, None, None],
        # ],
        r'LWF \cite{DBLP:journals/corr/LiH16e}': [
            ['b11c67b2b250447386ccf6761069df3e', '5ed58dcddf034f20b3880a00fd3b9c99', 'b4fbf6b731b3478a8cbfa431e6bcbc55', '0721fc99ff1d462589257c7763bf0757', '0c8842ba3b3a48918f5d989bece75ff5'],
            ['8ec8937ed43840d5816ab73137ba691e', 'b67c72e93c9742759fbd931db94889c8', 'fc5f682848bf430eba6e0efeb0eae36a', '12396194a938478e90314aa9764801ae', '198634c44431483f8ff3340b3f480e10'],
            ['65af3e6161774372b12be07d26fcc689', 'b9ffdf7d0fd84033b78f56bf7aa285e6', '630cbe0a131143bd994ca103bc0eae7e', '399d07ae54494206a7d46c60dc6598dc', 'd80fa368aa2444979dc886efee3035a4'],
        ],
        'Ours': [
            ['0213d3aad73e4b9b8a968f6446d8b83d', 'ade72c7c04dd4ba397b2f8bcb72cbeab', 'cb7c412cac6f4b6cb0331ad7ce01c9e3', '37bc9a366c73499ebf4ce17dd65a5c8b', '478400fa86de489c939bba93bb6abcd0'],
            ['8c0cbf9e430b4aefa0f2e4e30cd9b1e4', '085e9692e130475eb1d16e4a88e01fd7', '4158b5c745ce47aea25a17b2ca5cbbd8', 'b665260764d34b39ba0a64cfc3e5e4ca', '4c2a014e5ba34537ac908a22d19f64a5'],
            ['db96bc1bfa5148fd8e6ee55c6524f08b', '362f0c42025542b480832f2a6f6bdc4d', 'c992ec5df36445df876c3271ac5060cd', '74c1b839b953450fbd7f9e7fe8eb5bc8', '17e42a2e1b18434da2c5204ed9c6d039'],
        ],
    }
    num_parameters = [0, 0, 0, 0, 0, 0, 0]

    # client = mlflow.tracking.MlflowClient('///home/pwr/Documents/stochastic-depth-v2/stochastic-depth-data-streams/mlruns/')
    # client = mlflow.tracking.MlflowClient('///home/jedrzejkozal/Documents/stochastic-depth-data-streams/mlruns/')
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

    for dataset_run_ids, experiment_id in zip(run_ids, (4, 1, 9)):
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
