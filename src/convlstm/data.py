import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from PIL import Image
from torchvision import transforms as T
import random
from tqdm import tqdm

NO_OBJECT = -1

class SequenceDataset(Dataset):
    def __init__(self, root_dir, config, mode='train'):
        self.config = config
        self.mode = mode
        self.aug_cfg = config['augmentation']
        
        # Параметры из конфига
        self.seq_length = config['data']['seq_length']
        self.img_size = config['data']['img_size']
        
        # Базовые преобразования
        self.base_transform = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
        ])
        
        self.groups = self._load_groups(root_dir)
        self.aug_transform = self._create_augmentations()

    def _create_augmentations(self):
        augs = []
        cfg = self.aug_cfg
            
        if cfg['color_jitter']['enabled']:
            augs.append(
                T.RandomApply([
                    T.ColorJitter(
                        brightness=cfg['color_jitter']['brightness'],
                        contrast=cfg['color_jitter']['contrast'],
                        saturation=cfg['color_jitter']['saturation'],
                        hue=cfg['color_jitter']['hue']
                    )],
                    p=cfg['color_jitter']['p']
                ))
        
        if cfg['vflip']['enabled']:
            augs.append(
                T.RandomVerticalFlip(p=cfg['vflip']['p'])
            )
        
        if cfg['hflip']['enabled']:
            augs.append(
                T.RandomHorizontalFlip(p=cfg['hflip']['p'])
            )
            
        return T.Compose(augs)

    def _load_groups(self, root_dir):
        groups = []
        for group_dir in glob.glob(os.path.join(root_dir, '*')):
            if not os.path.isdir(group_dir):
                continue
                
            frames = []
            for f in glob.glob(os.path.join(group_dir, '*.png')):
                filename = os.path.basename(f)
                parts = filename.split('_')
                if len(parts) < 2:
                    continue
                    
                frame_num = int(parts[0])
                class_label = int(parts[1].split('.')[0])
                frames.append((frame_num, class_label, f))
            
            frames.sort(key=lambda x: x[0])
            groups.append(frames)
            
        return groups
    
    def _load_image(self, path):
        if path is None:  # Пропущенный кадр
            return torch.zeros((3, self.img_size, self.img_size))
        return self.base_transform(Image.open(path).convert('RGB'))
    
    def _apply_augmentations(self, sequence):
        if self.mode != 'train':
            return sequence
        
        state = torch.get_rng_state()

        result = []
        for img in sequence:
            torch.set_rng_state(state)
            img = self.aug_transform(img)
            result.append(img)
        
        return result
    
    def __len__(self):
        return len(self.groups)
    
    def __getitem__(self, idx):
        frames = self.groups[idx]
        max_frame = max(f[0] for f in frames)
        
        sequence = []
        labels = []
        for i in range(max_frame + 1):
            # Ищем кадр с нужным номером
            frame = next((f for f in frames if f[0] == i), None)
            
            if frame is None:  # Пропущенный кадр
                img = self._load_image(None)
                label = NO_OBJECT
            else:
                img = self._load_image(frame[2])
                label = frame[1]
                
            sequence.append(img)
            labels.append(label)
        
        # Применение аугментаций
        sequence = self._apply_augmentations(sequence)
        
        # Обрезка/дополнение до нужной длины
        if len(sequence) < self.seq_length:
            padding = [torch.zeros_like(sequence[0])] * (self.seq_length - len(sequence))
            sequence += padding
        else:
            sequence = sequence[:self.seq_length]
            
        # Основной класс последовательности (самый частый валидный класс)
        valid_labels = [l for l in labels if l != NO_OBJECT]
        class_label = max(set(valid_labels), key=valid_labels.count) if valid_labels else NO_OBJECT    
        
        return torch.stack(sequence), int(class_label - NO_OBJECT)


class ConvLSTMDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config['hyp']['batch_size']
        self.img_size = config['data']['img_size']
        self.seq_length = config['data']['seq_length']
        
    def setup(self, stage=None):
        self.train_ds = SequenceDataset(
            self.config['data']['train'],
            config=self.config,
            mode='train'
        )
        
        self.val_ds = SequenceDataset(
            self.config['data']['val'],
            config=self.config,
            mode='val'
        )
        
        self.test_ds = SequenceDataset(
            self.config['data']['test'],
            config=self.config,
            mode='test'
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def _count_classes(self, dataset):
        """Подсчёт количества элементов каждого класса"""
        counts = {}
        for group in dataset.groups:
            # Берём последний валидный класс в последовательности
            valid_labels = [f[1] for f in group if f[1] != NO_OBJECT]
            class_label = max(set(valid_labels), key=valid_labels.count) if valid_labels else NO_OBJECT  
            if class_label not in counts:
                counts[class_label] = 0
            counts[class_label] += 1
        return counts

    def print_class_distribution(self):
        """Вывод статистики по классам"""
        print("\nClass Distribution:")
        for phase in ['train', 'val', 'test']:
            ds = getattr(self, f'{phase}_ds', None)
            if ds:
                counts = self._count_classes(ds)
                total = sum(counts.values())
                print(f"\n{phase.upper()} (total={total}):")
                for cls, count in sorted(counts.items()):
                    print(f"  Class {cls}: {count} samples ({count/total:.1%})")
