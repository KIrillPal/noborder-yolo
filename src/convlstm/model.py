import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torchmetrics import Accuracy, Precision, Recall, F1Score
import mlflow

from src.convlstm.loss import FocalLoss


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(pl.LightningModule):
    def __init__(
        self,
        input_channels,
        hidden_dims,
        kernel_sizes,
        num_classes,
        num_layers,
        config,
        learning_rate=3e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.num_layers = num_layers
        self.hidden_dims = hidden_dims
        self.kernel_sizes = kernel_sizes
        self.learning_rate = learning_rate
        self.config = config
        
        # Создание ConvLSTM слоев
        self.conv_lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            input_dim = input_channels if i == 0 else hidden_dims[i-1]
            self.conv_lstm_layers.append(
                ConvLSTMCell(
                    input_dim=input_dim,
                    hidden_dim=hidden_dims[i],
                    kernel_size=kernel_sizes[i]
                )
            )
        
        # Классификатор
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_dims[-1], num_classes)
        
        # Потери
        self.loss = self._configure_loss()
        
        # Метрики
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.train_precision = Precision(task='multiclass', num_classes=num_classes, average='macro')
        self.val_precision = Precision(task='multiclass', num_classes=num_classes, average='macro')
        self.train_recall = Recall(task='multiclass', num_classes=num_classes, average='macro')
        self.val_recall = Recall(task='multiclass', num_classes=num_classes, average='macro')
        self.train_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.val_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.val_outputs = []
        
    def _configure_loss(self):
        if self.config['class_balancing']['use_focal_loss']:
            return FocalLoss(self.config)
        else:
            return nn.CrossEntropyLoss()

    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        
        # Инициализация скрытых состояний
        hidden_states = []
        for i in range(self.num_layers):
            hidden_states.append(self.conv_lstm_layers[i].init_hidden(batch_size, (H, W)))
        
        # Обработка временной последовательности
        for t in range(time_steps):
            input_tensor = x[:, t, :, :, :]
            
            for layer_idx in range(self.num_layers):
                h, c = self.conv_lstm_layers[layer_idx](
                    input_tensor=input_tensor,
                    cur_state=hidden_states[layer_idx]
                )
                hidden_states[layer_idx] = (h, c)
                input_tensor = h  # Передача выхода как входа следующему слою
        
        # Извлечение признаков из последнего слоя
        last_hidden = hidden_states[-1][0]
        pooled = self.global_pool(last_hidden).squeeze(-1).squeeze(-1)
        return self.fc(pooled)

    def _log_metrics(self, phase, loss, preds, targets):
        accuracy = getattr(self, f'{phase}_accuracy')(preds, targets)
        precision = getattr(self, f'{phase}_precision')(preds, targets)
        recall = getattr(self, f'{phase}_recall')(preds, targets)
        f1 = getattr(self, f'{phase}_f1')(preds, targets)
        
        # Логирование в MLflow
        mlflow.log_metrics({
            f'{phase}_loss': loss.item(),
            f'{phase}_accuracy': accuracy.item(),
            f'{phase}_precision': precision.item(),
            f'{phase}_recall': recall.item(),
            f'{phase}_f1': f1.item()
        }, step=self.global_step)
        
        # Логирование в Lightning
        self.log_dict({
            f'{phase}_loss': loss,
            f'{phase}_accuracy': accuracy,
            f'{phase}_precision': precision,
            f'{phase}_recall': recall,
            f'{phase}_f1': f1
        }, prog_bar=True, logger=True)

        
        if phase == 'val':
            print(f"Validation Loss: {loss.item()}, Accuracy: {accuracy.item()}, Precision: {precision.item()}, Recall: {recall.item()}, F1: {f1.item()}")

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        self._log_metrics('train', loss, preds, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        
        self.val_outputs.append({
            'loss': loss,
            'preds': preds,
            'targets': y
        })
        
        return {'loss': loss, 'preds': preds, 'targets': y}

    def on_validation_epoch_end(self):
        
        # Агрегируем результаты по всем батчам
        avg_loss = torch.stack([x['loss'] for x in self.val_outputs]).mean()
        preds = torch.cat([x['preds'] for x in self.val_outputs])
        targets = torch.cat([x['targets'] for x in self.val_outputs])
        
        # Логируем метрики
        self._log_metrics('val', avg_loss, preds, targets)
        
        # Очищаем результаты для следующей эпохи
        self.val_outputs.clear()
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()


    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)