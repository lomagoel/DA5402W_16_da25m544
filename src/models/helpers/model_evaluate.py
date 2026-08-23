import mlflow
from mlflow import MlflowClient


def eval_best_model_version(model_name, mlflow_tracking_uri):

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()

    model_details = client.get_registered_model(model_name)
    latest_version_info = model_details.latest_versions[-1]
    print(f'Latest Version: {latest_version_info.version}')

    model_aliases = model_details.aliases

    if 'best' not in model_aliases:
        client.set_registered_model_alias(model_name, 'best', latest_version_info.version)
    else:
        best_version_num = model_aliases['best']
        best_version = client.get_model_version(model_name, best_version_num)
        best_run = client.get_run(best_version.run_id).data
        latest_run = client.get_run(latest_version_info.run_id).data

        best_score = best_run.metrics.get('test_f1_score', -1)
        latest_score = latest_run.metrics.get('test_f1_score', -1)
        print(f'best model f1_score: {best_score}')
        print(f'latest model f1_score: {latest_score}')
        
        if latest_score > best_score:
            client.set_registered_model_alias(model_name, 'best', latest_version_info.version)
