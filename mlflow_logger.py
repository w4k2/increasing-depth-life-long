import pathlib
import tempfile
import matplotlib.pyplot as plt
import torch
import seaborn as sn
import re
from avalanche.logging import StrategyLogger
import mlflow
import mlflow.pytorch


class MLFlowLogger(StrategyLogger):
    def __init__(self, run_id=None, experiment_name='Default'):
        super().__init__()
        self.run_id = run_id
        self.experiment_name = experiment_name
        client = mlflow.tracking.MlflowClient()
        self.experiment = client.get_experiment_by_name(experiment_name)
        if self.experiment is None:
            id = mlflow.create_experiment(experiment_name)
            self.experiment = client.get_experiment(id)
        self.experiment_id = self.experiment.experiment_id

        if self.run_id == None:
            with mlflow.start_run(experiment_id=self.experiment_id):
                active_run = mlflow.active_run()
                self.run_id = active_run.info.run_id

    def log_parameters(self, parameters: dict):
        with mlflow.start_run(run_id=self.run_id, experiment_id=self.experiment_id):
            mlflow.log_params(parameters)

    def log_single_metric(self, name, value, x_plot):
        with mlflow.start_run(run_id=self.run_id, experiment_id=self.experiment_id):
            metric_name = self.map_metric_name(name)
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
        if len(res) > 0:
            i = res[0].start()
            task_id_str = name[i+3:i+6]
            task_id = int(task_id_str)
            new_name = f'{phase}_{metric_name}_task_{task_id}'
        elif phase == 'train':
            new_name = f'{phase}_{metric_name}_task'
        else:
            new_name = f'avrg_{phase}_{metric_name}'
        return new_name

    def log_conf_matrix(self, matrix: torch.Tensor):
        plt.figure()
        ax = sn.heatmap(matrix, annot=False)
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = pathlib.Path(tmpdir) / f"test_confusion_matrix.jpg"
            plt.savefig(save_path)
            with mlflow.start_run(run_id=self.run_id, experiment_id=self.experiment_id):
                mlflow.log_artifact(save_path, f'test_confusion_matrix')

    def log_model(self, model: torch.nn.Module):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = pathlib.Path(tmpdir) / 'model.pth'
            torch.save(model, model_path)
            with mlflow.start_run(run_id=self.run_id, experiment_id=self.experiment_id):
                mlflow.log_artifact(model_path, 'model')
