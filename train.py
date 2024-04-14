import config
import platform
import tensorflow as tf
import tensorflow_addons as tfa
import os
from model import get_pretrain_model as get_model

from preprocess.PT_DatasetGenerator import PT_DatasetGenerator


curr_dataset = config.pt_dataset

#
#   _______           _         _
#  |__   __|         (_)       (_)
#     | | _ __  __ _  _  _ __   _  _ __    __ _
#     | || '__|/ _` || || '_ \ | || '_ \  / _` |
#     | || |  | (_| || || | | || || | | || (_| |
#     |_||_|   \__,_||_||_| |_||_||_| |_| \__, |
#                                          __/ |
#                                         |___/
#

def train():



    # 1. Build Model
    checkpoint_path = config.model_path
    model = get_model(checkpoint_path=None)

    # 2. Get Optimizer
    optimizer, jit_compile = get_optimizer()

    # 3. Compile Model
    model.compile(optimizer=optimizer, jit_compile=jit_compile)

    # 4. Get Datasets
    train_dataset, val_dataset = get_dataset()
    # train_dataset = train_dataset.take(200)

    # train_dataset = train_dataset.rebatch(128)
    # val_dataset = val_dataset.rebatch(128)

    # 5. Get Checkpoints
    checkpoints = get_checkpoints()


    # 6. Train Model
    history = model.fit(
        train_dataset,
        epochs=config.epochs,
        validation_data=val_dataset,
        callbacks=checkpoints
    )






#
#   _    _        _
#  | |  | |      | |
#  | |__| |  ___ | | _ __    ___  _ __  ___
#  |  __  | / _ \| || '_ \  / _ \| '__|/ __|
#  | |  | ||  __/| || |_) ||  __/| |   \__ \
#  |_|  |_| \___||_|| .__/  \___||_|   |___/
#                   | |
#                   |_|
#



def get_dataset():

    dataset_generator = PT_DatasetGenerator(curr_dataset)
    train_dataset, val_dataset = dataset_generator.load_datasets()

    return train_dataset, val_dataset


def get_optimizer():
    jit_compile = False
    learning_rate = 0.001
    learning_rate = tf.keras.optimizers.schedules.CosineDecay(
        0.0,
        10000,
        alpha=0.1,
        warmup_target=learning_rate,
        warmup_steps=500
    )
    optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)
    if config.mixed_precision is True:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    return optimizer, jit_compile



def get_checkpoints():
    checkpoints = []
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=config.model_path,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )
    checkpoints.append(model_checkpoint)
    return checkpoints




def train_distributed():
    with config.mirrored_strategy.scope():
        train()


if __name__ == "__main__":
    if config.distributed is False:
        train()
    else:
        train_distributed()




