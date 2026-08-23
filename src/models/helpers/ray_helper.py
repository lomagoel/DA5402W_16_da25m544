from src.models.helpers.model_trainer import ModelTrainer
import ray
from ray import tune
from ray.air.integrations.mlflow import setup_mlflow


class RayWorker():
    def __init__(self, run_name, mlflow_config):
        self.mlflow = setup_mlflow(
            tracking_uri=mlflow_config['tracking_uri'], 
            experiment_name=mlflow_config['experiment_name'],
            run_name=run_name
        )
        
    def _log(self, report_metrics, final_metrics, epoch):
        tune.report(report_metrics)
        self.mlflow.log_metrics(final_metrics, step=epoch)

    def tune(self, model, train_dataset, validation_dataset, tune_config, num_classes, epochs=3):
        trainer = ModelTrainer(model, tune_config['batch_size'], tune_config['optim'], num_classes)
        trainer.train(train_dataset, validation_dataset, epochs=epochs, tune_callback=lambda r,f,e: self._log(r, f, e))
        self.mlflow.end_run()


class RayDriver():
    def __init__(self, model, resources, mlflow_config, num_classes):
        ray.init()
        self.model = model
        self.resources = resources
        self.mlflow_config = mlflow_config
        self.num_classes = num_classes

    def _worker_job(self, tune_config, data):
        context = ray.tune.get_context()
        run_name = f'ray_worker_{context.get_trial_id()}' 
        worker = RayWorker(run_name, self.mlflow_config)
        worker.tune(self.model, *data, tune_config, self.num_classes)

    def tune(self, data, search_space, num_samples=5):
        tuner = tune.Tuner(
            tune.with_resources(tune.with_parameters(self._worker_job, data=data), self.resources),
            param_space=search_space,
            tune_config=tune.TuneConfig(num_samples=num_samples)
        )

        results = tuner.fit()
        best_config = results.get_best_result(metric='val_f1_score', mode='max').config

        return best_config

    def shutdown(self):
        ray.shutdown()


def uniform_space(start, end):
    return tune.loguniform(start, end)

def random_choice(choices):
    return tune.choice(choices)