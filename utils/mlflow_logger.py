import urllib
import yaml
import os
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
    def __init__(self, run_id=None, experiment_name='Default', nested=False, run_name=None):
        super().__init__()
        self.run_id = run_id
        self.experiment_name = experiment_name
        client = mlflow.tracking.MlflowClient()
        self.experiment = client.get_experiment_by_name(experiment_name)
        if self.experiment is None:
            id = mlflow.create_experiment(experiment_name)
            self.experiment = client.get_experiment(id)
        self.experiment_id = self.experiment.experiment_id
        self.nested = nested

        if self.run_id == None:
            with mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name, nested=nested):
                active_run = mlflow.active_run()
                self.run_id = active_run.info.run_id

    def log_parameters(self, parameters: dict):
        with mlflow.start_run(run_id=self.run_id, experiment_id=self.experiment_id, nested=self.nested):
            mlflow.log_params(parameters)

    def log_single_metric(self, name, value, x_plot):
        with mlflow.start_run(run_id=self.run_id, experiment_id=self.experiment_id, nested=self.nested):
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
            new_name = f'{phase}_{metric_name}'
        else:
            new_name = f'{phase}_{metric_name}'
        return new_name

    def log_conf_matrix(self, matrix: torch.Tensor):
        plt.figure()
        ax = sn.heatmap(matrix, annot=False)
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')

        with SwapArtifactUri(self.experiment_id, self.run_id):
            with tempfile.TemporaryDirectory() as tmpdir:
                save_path = pathlib.Path(tmpdir) / f"test_confusion_matrix.jpg"
                plt.savefig(save_path)
                with mlflow.start_run(run_id=self.run_id, experiment_id=self.experiment_id, nested=self.nested):
                    mlflow.log_artifact(save_path, f'test_confusion_matrix')

    def log_model(self, model: torch.nn.Module):
        with SwapArtifactUri(self.experiment_id, self.run_id):
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = pathlib.Path(tmpdir) / 'model.pth'
                torch.save(model, model_path)
                with mlflow.start_run(run_id=self.run_id, experiment_id=self.experiment_id, nested=self.nested):
                    mlflow.log_artifact(model_path, 'model')

    def log_avrg_accuracy(self):
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(self.run_id)
        run_metrics = run.data.metrics
        test_accs = [acc for name, acc in run_metrics.items() if name.startswith('test_accuracy_task_')]
        test_avrg_acc = sum(test_accs) / len(test_accs)
        client.log_metric(self.run_id, 'avrg_test_acc', test_avrg_acc)


class SwapArtifactUri:
    def __init__(self, experiment_id, run_id):
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.artifact_uri = None

    def __enter__(self):
        repo_path = repo_dir()
        meta_path = repo_path / 'mlruns' / f'{self.experiment_id}' / f'{self.run_id}' / 'meta.yaml'

        run_meta = self.load_meta(meta_path)

        self.artifact_uri = run_meta['artifact_uri']
        run_meta['artifact_uri'] = f'file://{repo_path}/mlruns/{self.experiment_id}/{self.run_id}/artifacts'
        with open(meta_path, 'w') as file:
            yaml.safe_dump(run_meta, file)

    def __exit__(self, exc_type, exc_value, exc_tb):
        repo_path = repo_dir()
        meta_path = repo_path / 'mlruns' / f'{self.experiment_id}' / f'{self.run_id}' / 'meta.yaml'

        run_meta = self.load_meta(meta_path)
        run_meta['artifact_uri'] = self.artifact_uri
        with open(meta_path, 'w') as file:
            yaml.safe_dump(run_meta, file)

    def load_meta(self, meta_path):
        with open(meta_path, 'r') as file:
            run_meta = yaml.safe_load(file)
        return run_meta


def repo_dir():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = pathlib.Path(repo_dir)
    if type(repo_dir) == pathlib.WindowsPath:
        repo_dir = pathlib.Path(*repo_dir.parts[1:]).as_posix()
    return repo_dir.parent
