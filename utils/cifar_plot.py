from turtle import color, right
import matplotlib.pyplot as plt
import pathlib
import numpy as np 
import seaborn as sns


def main():
    runs_cifar_wo_pretraining = {
        'Upperbound': ['7299df53d45f442cbd66dd74a2987044', '5e30ad287cb64572bcc7db330a026016', '68037f4b3f2d4bec870dcfcc6c0ae7c4', 'd13c12cc8aa7414f8cc30484e5495c37', 'b1d9d932d05b49aab0b8841d69add04f'],
        'EWC': ['76073c529caf428da7f3d68f649209b7', '61fbbec0858c4b9fa1438c09010a97d9', 'ee0377e0883e4a6d8f399a8e3a608d10', 'f4ae4a6986044d588b39f06297714856', '8c9dd72c9c974021abbfce3bd7b490c0'],
        'ER': ['46aae2323aea487aa37e28dde368d2b0', 'eeabf8f4d9614edd8123ffb1751dfd63', '8bf82c9490ef48fe94c9b2577b42b0f6', '40a3970e1ec644bb90ed7a2de4df4353', 'cff2bf5764984fdb9ecfe2adcfc1b945'],
        'A-GEM': ['0c7f7156b6044b52b7ccbbcbca59c649', '0c2d264e07dd4733b2356a424aa1c5fc', '42d13c77ba1a45b4a75f646f716ee3b6', 'cbbb676caf714e3f86a7e5e442282c40', '1f8d7cee06e247d298d1676bdcf2971a'],
        'PNN': ['2de7e46a8c2443ac8587542d470d7154', '46f3944ed802435098ba8c1206bd6ffe', '3f7278e5928e47aeb02e65a8fca46de0', '78a81252967546a4ab9e282d755e7e20', '9400d9e552974abd884ff0af8f8bcb48'],
        'LWF': ['aa73b91cd0964bfab38c752bb32047c5', '67613a30419c445b862885a7903970b0', 'bda9f4f43c5a45daaae12cb7228dce43', '3a6aec25b4514510a866eaa5a6c495d9', '617278fe12ab40b2a2a42698dbea343c'],
        'Ours': ['375c3e34198d42528207bf6efca58b16', '1e04d877656b4d18b123a568d24a381d', '30f5d1ef87f4430590ca4e48fc803d8d', '110cba2ebc0447b5b5294ed87ae45e7a', '2c820fda1afc4aa1965edbc7c962788e'],
    }
    runs_cifar_w_pretraining = {
        'Upperbound': ['4b19c8293100481eb13e77cb69a74a63', '991355734d0b46018761a01a19c9e2c2', '7daa6e8f75884745a3b4dff2546425f2', 'ed645bd069b14eefb06ef60c8ceed3b5', '6123aacb805f4427a473b0c355ead066'],
        'EWC': ['6ce642ba84484e4590458a61b73dbbe7', '0688779947fd496a890fd0f2494493e8', '81234edf47b645908d89b4a2ccc2d1b9', '4e96adca217a4c4ea1c65fa8efa12cb0', 'df785c3e21674995a531e7a17e76f53e'],
        'ER': ['7299961ddfc94f7b835c5cbaea7436d9', '6685595dc6074ed48e92ab490ce0a039', '21f469599c4e42218022ae185cb53c3e', 'b6ff39f4c6ee423ba74df902fa2c5271', 'd11dd8a241084bce97511322b7d0a741'],
        'A-GEM': ['77b14fad8723461d94ce6acd5407e62a', 'bf3bbf6fb1bb4fa18264ec8066ba4653', '66ab2af44beb4aab9e664d4279304215', '4aebed10ef9a4e9cb92e3e973b4692f4', '913c72664b184b9fa165496534e721c5'],
        'LWF': ['b11c67b2b250447386ccf6761069df3e', '5ed58dcddf034f20b3880a00fd3b9c99', 'b4fbf6b731b3478a8cbfa431e6bcbc55', '0721fc99ff1d462589257c7763bf0757', '0c8842ba3b3a48918f5d989bece75ff5'],
        'Ours': ['0213d3aad73e4b9b8a968f6446d8b83d', 'ade72c7c04dd4ba397b2f8bcb72cbeab', 'cb7c412cac6f4b6cb0331ad7ce01c9e3', '37bc9a366c73499ebf4ce17dd65a5c8b', '478400fa86de489c939bba93bb6abcd0'],
    }
    runs_tinyimagenet_wo_pretraining = {
        'Upperbound': ['a4b88115c40044a3847c2cb1b9cd931f', '27c1bf3adf6a4d9e8781124f95935f23', 'fb1a96cec4854691aa5b027c14787b0b', '44c5d5e4152c40ac81e1a69709c1bfdf', '61dbee30d5b34c67ac1a199af85ba6e8'],
        'EWC': ['3631e37897604ab4845c3cf9cf12b89e', 'a5f8df8c08a74b95a6cfadaee8ef1658', '3b2fd8ec871e4119906e56e0f2a151b5', '5cdd21a24caa4706a8c3fdeca64ef099', 'c92d7e11f7dc4dc6b4d3d36e4cc64b1f'],
        'ER': ['24feac44583844499e9f4908835c80d5', '6c7b8a96cea3433db1dfedeaf05a859a', '739859dcecf14a2fb0790357e20e242d', '65f074de0f1a4031850c0009ef000559', 'b5d020a2ed524fec93abfc02839b2443'],
        'A-GEM': ['06cb3837219c4ae8b7da0f0e392882cd', 'def7013480ce449e9107547a39163289', '578c7dd017074da4b4e7fef482c60d35', 'd99302186bff4c27982a9899b6f0787b', 'a9711018e97f4d5f949d17c5fb2a5a1a'],
        'PNN': ['98231e19162a4be49d6255eb223d0d17', 'c7e5fa154c474884ab786a6935484bff', '95765e3d72934c17868c702d2d887187', '605e92667a4b40cbb3efc2f11e22b359', '30667327abdc44bf928b78bf7ff5b0f4'],
        'LWF': ['9c233fc60a334974a558d3b631de0feb', '504286b1453b44a19f83219e8d97d4b9', 'fa8f4c5357a24d498cada1c261da921b', '6858810a86094c089b63c3cb4e25463a', '0f6a32659efa48b68d57d27851fc9f4d'],
        'Ours': ['abaa33c8c9ce4e98bc56f65f79239732', '1cf83d2edad24c6ca1848c76b347ed50', '65358873c6394d1ba34feaa4f34a693f', 'd56a160ce9704c85a7c44d6f1837437f', '35f4d728ad554b5f9ab535152671ec27'],
    }
    runs_tinyimagenet_w_pretraining = {
        'Upperbound': ['0e38b56d2a5e456688629df43930e5c9', 'f8e434da1c184d7eb73245307237de23', '12aa7dad46644cf28e098a1cb62cfa6a', '173e02b046eb4dddb101c699a5982dda', '1d5bb51ac3cf4bcc98b1ce009704cfea'],
        'EWC': ['57a5bf176df8483ab52a8eceaa06675c', 'adbc4d8cead44ef59d3f36db0dc11381', 'a4168977b22c4e92815fb5d213331192', 'bb68f28384b44096b19e88b72a3a50ea', '6671438931b349fa9d363cdf3053703c'],
        'ER': ['2f5bc0afbce04d5a8bea033d6a8ef1ba', '7985cab2d91545e0893d705da7be50e5', 'a2d2b5e72ebd4dcbbc9f2cda2f9e81d5', 'd7d92cefe8864d488ee75a543f110d86', '8a0f6b6d07fb4087a5a20b2367fec0d4'],
        'A-GEM': ['622b70f5b8084802b0c31718fd56d177', '148031b8f7104b1c90ca4bd88ea09f90', '5a6ae1b854b94e6b895e26244356f7c4', 'dadfcbf7c6be44d69ae45c9a2fa887e1', '657920bc101843da9aa1474506dff755'],
        'LWF': ['8ec8937ed43840d5816ab73137ba691e', 'b67c72e93c9742759fbd931db94889c8', 'fc5f682848bf430eba6e0efeb0eae36a', '12396194a938478e90314aa9764801ae', '198634c44431483f8ff3340b3f480e10'],
        'Ours': ['8c0cbf9e430b4aefa0f2e4e30cd9b1e4', '085e9692e130475eb1d16e4a88e01fd7', '4158b5c745ce47aea25a17b2ca5cbbd8', 'b665260764d34b39ba0a64cfc3e5e4ca', '4c2a014e5ba34537ac908a22d19f64a5'],

    }


    results_list = [runs_cifar_wo_pretraining, runs_cifar_w_pretraining, runs_tinyimagenet_wo_pretraining, runs_tinyimagenet_w_pretraining]
    results_names = ['SplitCifar w/o pretraining', 'SplitCifar w pretraining', 'SplitTinyImagenet w/o pretraining', 'SplitTinyImagenet w pretraining']
    colors = sns.color_palette("husl", 7)
    color_dict = {
        'Upperbound': colors[0],
        'EWC': colors[1],
        'ER': colors[2],
        'A-GEM': colors[3],
        'PNN': colors[4],
        'LWF': colors[5],
        'Ours': colors[6],
    }
    handles = []

    with sns.axes_style("darkgrid"):
        for i, (runs, name) in enumerate(zip(results_list, results_names)):
            num_tasks = 20
            for method_name, run_ids in runs.items():
                all_acc = []
                experiment_id = 4 if 'Cifar' in name else 1
                for run_i in range(5):
                    plot_avrg_acc = get_average_acc(num_tasks, run_ids, run_i, experiment_id)
                    all_acc.append(plot_avrg_acc)
                all_acc = np.array(all_acc)
                acc_avrg_over_runs = np.mean(all_acc, axis=0)
                acc_std = np.std(all_acc, axis=0)

                plt.subplot(2, 2, i+1)
                line_handle = plt.plot(acc_avrg_over_runs, label=method_name, color=color_dict[method_name])
                if i % 2 == 0 and len(handles) < len(color_dict):
                    handles.append(line_handle[0])
                plt.fill_between(list(range(20)), acc_avrg_over_runs-acc_std, acc_avrg_over_runs+acc_std, alpha=0.3, color=color_dict[method_name])

            plt.xticks(list(range(0, 20, 2)))
            plt.xlim(left=0, right=19)
            if i % 2 == 1:
                print(handles)
                plt.legend(handles=handles, labels=list(color_dict.keys()), loc='center left', bbox_to_anchor=(1, 0.5))
            plt.title(name)
            if i > 1:
                plt.xlabel('number of tasks')
            if i % 2 == 0:
                plt.ylabel('average accuracy')
    plt.show()

def get_average_acc(num_tasks, run_ids, run_i, experiment_id):
    run_accs = read_run_acc(run_ids[run_i], experiment_id)
    plot_avrg_acc = []

    for i in range(num_tasks):
        avrg_acc = []
        for j in range(i+1):
            avrg_acc.append(run_accs[j][i])
            i -= 1
        avrg_acc = np.mean(avrg_acc)
        plot_avrg_acc.append(avrg_acc)
    return plot_avrg_acc

def read_run_acc(run_id, experiment_id=4, num_tasks=20):
    run_path = pathlib.Path(f'mlruns/{experiment_id}/{run_id}/metrics/')

    all_tasks_acc = []

    for task_id in range(num_tasks):
        filepath = run_path / f'test_accuracy_task_{task_id}'
        task_accs = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                acc_str = line.split()[-2]
                acc = float(acc_str)
                task_accs.append(acc)
        all_tasks_acc.append(task_accs)
    
    return all_tasks_acc

if __name__ == '__main__':
    main()