import mlflow
from mlflow import MlflowClient


MLFLOW_TRACKING_URI = 'http://34.47.164.107:5000'

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

def set_classifier(model_version):
    classifier = mlflow.register_model(
        model_uri=model_version.source,
        name='classifier'
    )
    client.set_registered_model_alias(
        name='classifier',
        alias="prod",
        version=classifier.version
    )

best_resnet18 = client.get_model_version_by_alias(
    name='resnet18',
    alias='best'
)

best_mobilenet_small = client.get_model_version_by_alias(
    name='mobilenet_v3_small',
    alias='best'
)

resnet18_f1_score = client.get_run(best_resnet18.run_id).data.metrics.get('test_f1_score', -1)
mobilenet_small_f1_score = client.get_run(best_mobilenet_small.run_id).data.metrics.get('test_f1_score', -1)

print(f'resnet18 f1_score: {resnet18_f1_score}')
print(f'mobilenet f1_score: {mobilenet_small_f1_score}')

if resnet18_f1_score > mobilenet_small_f1_score:
    set_classifier(best_resnet18)
else:
    set_classifier(best_mobilenet_small)