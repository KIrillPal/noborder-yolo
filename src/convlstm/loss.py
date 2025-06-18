import torch.nn.functional as F
import torch.nn as nn
import torch

class FocalLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config['class_balancing']
        self.alpha = torch.tensor(self.config['focal_alpha'])
        self.gamma = self.config['focal_gamma']
        self.num_classes = len(self.config['focal_alpha'])
        
        # Проверки конфига
        self._validate_config()
        
    def _validate_config(self):
        assert self.config['use_focal_loss'], "Focal Loss отключен в конфиге"
        assert len(self.config['focal_alpha']) > 0, "Не заданы alpha для классов"
        assert all(a > 0 for a in self.config['focal_alpha']), "Alpha должны быть > 0"
        assert self.gamma >= 0, "Gamma должен быть >= 0"
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes) - raw logits от модели
            targets: (batch_size,) - ground truth классы (0..num_classes-1)
        Returns:
            torch.Tensor: Focal loss значение
        """
        alpha = self.alpha.to(inputs.device)
        
        # Вычисляем кросс-энтропию
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Получаем вероятности для правильных классов
        p_t = torch.exp(-ce_loss)
        
        # Вычисляем focal loss
        focal_loss = alpha[targets] * (1 - p_t)**self.gamma * ce_loss
        
        return focal_loss.mean()

    def extra_repr(self):
        return f"alpha={self.alpha.tolist()}, gamma={self.gamma}, num_classes={self.num_classes}"