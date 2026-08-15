import torch
import torch.nn as nn
import torch.optim as optim
from src.models.helpers.data_helper import dataloader
from torchmetrics import MeanMetric, MetricCollection, Accuracy, Precision, Recall, F1Score, AUROC


class ModelTrainer():
    def __init__(self, model, batch_size, optim_config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), **optim_config)
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self, train_dataset, val_dataset, label_to_idx,epochs=3, tune_callback=None):

        self.num_classes = len(label_to_idx)
        train_loader = dataloader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = dataloader(val_dataset, batch_size=self.batch_size, shuffle=False)

        train_loss_metric = MeanMetric().to(self.device)
        val_loss_metric = MeanMetric().to(self.device)
        val_metrics = MetricCollection({
            'val_f1_score': F1Score(task='multiclass', num_classes=self.num_classes, average='macro'),
            'val_accuracy_top1': Accuracy(task='multiclass', num_classes=self.num_classes, top_k=1),            
            'val_accuracy_top5': Accuracy(task='multiclass', num_classes=self.num_classes, top_k=5),
            'val_precision': Precision(task='multiclass', num_classes=self.num_classes, average='macro'),
            'val_recall': Recall(task='multiclass', num_classes=self.num_classes, average='macro'),
            'val_roc_auc': AUROC(task='multiclass', num_classes=self.num_classes, average='macro')
        }).to(self.device)

        for epoch in range(epochs):

            self.model.train()
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss_metric.update(loss, weight=images.size(0))

            self.model.eval()
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    val_loss = self.loss_fn(outputs, labels)

                    val_loss_metric.update(val_loss, weight=images.size(0))
                    val_metrics.update(preds=outputs, target=labels)


            train_loss = train_loss_metric.compute()
            val_loss = val_loss_metric.compute()
            metrics = val_metrics.compute()

            loss_metrics = {
                'train_loss': train_loss.item(),
                'val_loss': val_loss.item()
            }

            final_metrics = self._handle_metrics(loss_metrics, metrics)
            if tune_callback:
                tune_callback({**loss_metrics, 'val_f1_score': metrics['val_f1_score'].item()}, final_metrics, epoch)

            train_loss_metric.reset()
            val_loss_metric.reset()
            val_metrics.reset()



    def evaluate(self, data):
        test_loader = dataloader(data, batch_size=self.batch_size, shuffle=False)

        test_loss_metric = MeanMetric().to(self.device)
        test_metrics = MetricCollection({
            'test_f1_score': F1Score(task='multiclass', num_classes=self.num_classes, average='macro'),
            'test_accuracy_top1': Accuracy(task='multiclass', num_classes=self.num_classes, top_k=1),
            'test_accuracy_top5': Accuracy(task='multiclass', num_classes=self.num_classes, top_k=5),
            'test_precision': Precision(task='multiclass', num_classes=self.num_classes, average='macro'),
            'test_recall': Recall(task='multiclass', num_classes=self.num_classes, average='macro'),
            'test_roc_auc': AUROC(task='multiclass', num_classes=self.num_classes, average='macro')
        }).to(self.device)

        self.model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                test_loss = self.loss_fn(outputs, labels)

                test_loss_metric.update(test_loss, weight=images.size(0))
                test_metrics.update(preds=outputs, target=labels)

            test_loss = test_loss_metric.compute()
            metrics = test_metrics.compute()

            loss_metrics = {
                'test_loss': test_loss.item()
            }

            final_metrics = self._handle_metrics(loss_metrics, metrics)

            return final_metrics


    def _handle_metrics(self, metrics_obj, metrics):
        for key, value in metrics.items():
            metrics_obj[key] = value.item()

        return metrics_obj
