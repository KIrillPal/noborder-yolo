import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import mlflow
import yaml
import argparse
from pathlib import Path
import os
import numpy as np
from datetime import datetime
from contextlib import nullcontext
import shutil

from src.convlstm.model import ConvLSTM
from src.convlstm.data import ConvLSTMDataModule

####################
# Global constants #
####################

RUN_CONFIG_ARTIFACT = 'run_config'
BEST_MODEL_PATH = 'best_model.ckpt'

###########################
# Supplementary functions #
###########################

def print_abort(message: str):
    print(message)
    print("Aborting...")
    exit(1)

def check_data_dirs(config: dict):
    for part in ['train', 'val', 'test']:
        data_dir = Path(config['data'][part])
        if not data_dir.exists():
            print_abort(f"Data directory {data_dir} does not exist")

def setup_experiment(config: dict):
    # Создание директории эксперимента
    exp_dir = Path(config['store_exp_to']) / config['experiment']
    exp_dir.mkdir(parents=True, exist_ok=True)

##################
# Main functions #
##################

def train(config_path: str):
    # Загрузка конфигурации
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Инициализация MLflow
    mlflow.set_tracking_uri(config['mlflow_uri'])
    mlflow.set_experiment(config['experiment'])

    # Проверка данных
    check_data_dirs(config)
    setup_experiment(config)

    # Инициализация модели
    model = ConvLSTM(
        input_channels=config['model']['input_channels'],
        hidden_dims=config['model']['hidden_dims'],
        kernel_sizes=config['model']['kernel_sizes'],
        num_layers=config['model']['num_layers'],
        num_classes=config['model']['num_classes'],
        learning_rate=config['hyp']['learning_rate'],
        config=config
    )

    # Инициализация данных
    dm = ConvLSTMDataModule(config)
    dm.setup()
    dm.print_class_distribution()

    # Конфигурация обучения
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(config['store_exp_to']) / config['experiment'],
        filename=BEST_MODEL_PATH,
        monitor='val_loss',
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=config['hyp']['epochs'],
        accelerator='gpu' if config['device'] != 'cpu' else 'cpu',
        devices=[config['device']],
        callbacks=[checkpoint_callback],
        deterministic=(config['hyp']['seed'] is not None),
        enable_progress_bar=not config['debug']
    )

    # Начало обучения
    with mlflow.start_run():
        # Логирование параметров
        mlflow.log_params(config['model'])
        mlflow.log_params(config['hyp'])
        mlflow.log_artifact(config_path, RUN_CONFIG_ARTIFACT)

        # Обучение модели
        trainer.fit(model, dm)

        # Логирование модели
        mlflow.pytorch.log_model(model, "model")

        # Сохранение лучшей модели
        shutil.copy(checkpoint_callback.best_model_path, 
                   Path(config['store_exp_to']) / config['experiment'] / BEST_MODEL_PATH)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ConvLSTM model')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    train(args.config)