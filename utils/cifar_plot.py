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
        'LWF ': ['aa73b91cd0964bfab38c752bb32047c5', '67613a30419c445b862885a7903970b0', 'bda9f4f43c5a45daaae12cb7228dce43', '3a6aec25b4514510a866eaa5a6c495d9', '617278fe12ab40b2a2a42698dbea343c'],
        'Ours': ['375c3e34198d42528207bf6efca58b16', '1e04d877656b4d18b123a568d24a381d', '30f5d1ef87f4430590ca4e48fc803d8d', '110cba2ebc0447b5b5294ed87ae45e7a', '2c820fda1afc4aa1965edbc7c962788e'],
    }
    runs_cifar_w_pretraining = {
        'Upperbound': ['4b19c8293100481eb13e77cb69a74a63', '991355734d0b46018761a01a19c9e2c2', '7daa6e8f75884745a3b4dff2546425f2', 'ed645bd069b14eefb06ef60c8ceed3b5', '6123aacb805f4427a473b0c355ead066'],
        'EWC': ['6ce642ba84484e4590458a61b73dbbe7', '0688779947fd496a890fd0f2494493e8', '81234edf47b645908d89b4a2ccc2d1b9', '4e96adca217a4c4ea1c65fa8efa12cb0', 'df785c3e21674995a531e7a17e76f53e'],
        'ER': ['7299961ddfc94f7b835c5cbaea7436d9', '6685595dc6074ed48e92ab490ce0a039', '21f469599c4e42218022ae185cb53c3e', 'b6ff39f4c6ee423ba74df902fa2c5271', 'd11dd8a241084bce97511322b7d0a741'],
        'A-GEM': ['77b14fad8723461d94ce6acd5407e62a', 'bf3bbf6fb1bb4fa18264ec8066ba4653', '66ab2af44beb4aab9e664d4279304215', '4aebed10ef9a4e9cb92e3e973b4692f4', '913c72664b184b9fa165496534e721c5'],
        'LWF ': ['b11c67b2b250447386ccf6761069df3e', '5ed58dcddf034f20b3880a00fd3b9c99', 'b4fbf6b731b3478a8cbfa431e6bcbc55', '0721fc99ff1d462589257c7763bf0757', '0c8842ba3b3a48918f5d989bece75ff5'],
        'Ours': ['0213d3aad73e4b9b8a968f6446d8b83d', 'ade72c7c04dd4ba397b2f8bcb72cbeab', 'cb7c412cac6f4b6cb0331ad7ce01c9e3', '37bc9a366c73499ebf4ce17dd65a5c8b', '478400fa86de489c939bba93bb6abcd0'],
    }


    results_list = [runs_cifar_wo_pretraining, runs_cifar_w_pretraining]
    results_names = ['SplitCifar w/o pretraining', 'SplitCifar w pretraining']

    with sns.axes_style("darkgrid"):
        for i, (runs, name) in enumerate(zip(results_list, results_names)):
            num_tasks = 20
            for method_name, run_ids in runs.items():
                all_acc = []
                for run_i in range(5):
                    plot_avrg_acc = get_average_acc(num_tasks, run_ids, run_i)
                    all_acc.append(plot_avrg_acc)
                all_acc = np.array(all_acc)
                acc_avrg_over_runs = np.mean(all_acc, axis=0)
                acc_std = np.std(all_acc, axis=0)

                plt.subplot(1, 2, i+1)
                plt.plot(acc_avrg_over_runs, label=method_name)
                plt.fill_between(list(range(20)), acc_avrg_over_runs-acc_std, acc_avrg_over_runs+acc_std, alpha=0.3)

            plt.xticks(list(range(0, 20, 2)))
            plt.xlim(left=0)
            plt.legend()
            plt.title(name)
            plt.xlabel('number of tasks')
            plt.ylabel('average accuracy')
    plt.show()

def get_average_acc(num_tasks, run_ids, run_i):
    run_accs = read_run_acc(run_ids[run_i])
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