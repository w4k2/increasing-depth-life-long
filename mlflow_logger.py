import mlflow
import re

from avalanche.logging import StrategyLogger


class MLFlowLogger(StrategyLogger):
    def __init__(self, run_id=None):
        super().__init__()
        self.run_id = run_id
        if self.run_id == None:
            mlflow.start_run()
            active_run = mlflow.active_run()
            self.run_id = active_run.info.run_id

    def log_single_metric(self, name, value, x_plot):
        with mlflow.start_run(run_id=self.run_id):
            print(f'mlflow logger call, name = {name}, value = {value}, x_plot = {x_plot}')
            metric_name = self.map_metric_name(name)
            print('logging with name = ', metric_name)
            mlflow.log_metric(metric_name, value)

    @staticmethod
    def map_metric_name(name):
        metric_name = None
        if 'Top1_Acc' in name:
            metric_name = 'accuracy'
        elif 'Loss' in name:
            metric_name = 'loss'
        else:
            metric_name = 'unknown'

        phase = None
        if 'train_phase' in name:
            phase = 'train'
        elif 'eval_phase' in name:
            phase = 'test'

        res = re.finditer(r'Exp[0-9]+', name)
        res = list(res)
        # print(list(res))
        if len(res) > 0:
            i = res[0].start()
            # print(dir(i))
            task_id_str = name[i+3:i+6]
            # print('task_id_str = ', task_id_str)
            task_id = int(task_id_str)
            # print('task_id = ', task_id)
            new_name = f'{phase}_{metric_name}_task_{task_id}'
        elif phase == 'train':
            new_name = f'{phase}_{metric_name}_task'
        else:
            new_name = f'avrg_{phase}_{metric_name}'
        return new_name
