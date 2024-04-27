import config
import platform
import tensorflow as tf
import tensorflow_addons as tfa
import os
from model.callbacks.EvalsCallback import EvalsCallback
from preprocess.PTP_DatasetGenerator import PTP_DatasetGenerator

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
from model import get_pretrain_model_a2 as get_model

# curr_dataset = config.pt_dataset
curr_dataset = os.path.join(config.datasets_dir, 'dataset-arch2-lc0')

# save_model = config.model_path
save_model = os.path.join(config.weights_dir, 'chess-gpt-a5')

load_model = None
# load_model = curr_model


def train():
    # 1. Build Model
    model = get_model(checkpoint_path=load_model)

    # 2. Get Optimizer
    optimizer, jit_compile = get_optimizer()

    # 3. Compile Model
    model.compile(optimizer=optimizer, jit_compile=jit_compile)

    # 4. Get Datasets
    train_dataset, val_dataset = get_dataset()

    # 5. Get Checkpoints
    checkpoints = get_checkpoints(save_model)

    # 6. Train Model
    if config.distributed is True:
        history = model.fit(
            train_dataset,
            epochs=config.epochs,
            validation_data=val_dataset,
            steps_per_epoch=config.epoch_steps,
            validation_steps=config.val_steps,
            callbacks=checkpoints
        )
    else:
        history = model.fit(
            train_dataset,
            epochs=config.epochs,
            validation_data=val_dataset,
            callbacks=checkpoints
        )


    # Save history to file
    history_path = os.path.join(config.results_dir, 'history.txt')
    with open(history_path, 'w') as f:
        f.write(str(history.history))


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
    dataset_generator = PTP_DatasetGenerator(curr_dataset)
    train_dataset, val_dataset = dataset_generator.load_datasets()

    if config.distributed is True:
        train_dataset = train_dataset.rebatch(config.global_batch_size, drop_remainder=True)
        val_dataset = val_dataset.rebatch(config.global_batch_size, drop_remainder=True)
        train_dataset = train_dataset.repeat(10)
        val_dataset = val_dataset.repeat(10)
        train_dataset = config.mirrored_strategy.experimental_distribute_dataset(train_dataset)
        val_dataset = config.mirrored_strategy.experimental_distribute_dataset(val_dataset)
        print('-- Distributed Training Enabled --')

    return train_dataset, val_dataset


def get_optimizer():

    jit_compile = False


    # learning_rate = 0.0005  # --> 0.00005
    # learning_rate = tf.keras.optimizers.schedules.CosineDecay(
    #     0.0,
    #     100000,
    #     alpha=0.1,
    #     warmup_target=learning_rate,
    #     warmup_steps=1000
    # )
    learning_rate = 0.0005

    # learning_rate = 0.0005  # --> 0.0005
    # learning_rate = tf.keras.optimizers.schedules.CosineDecay(
    #     0.0,
    #     10000,
    #     alpha=0.1,
    #     warmup_target=learning_rate,
    #     warmup_steps=400
    # )
    # learning_rate = 0.0005


    optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)
    # optimizer = tfa.optimizers.LAMB(learning_rate=learning_rate)

    if config.mixed_precision is True:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    return optimizer, jit_compile


def get_checkpoints(checkpoint_path):
    checkpoints = []
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_model,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )
    checkpoints.append(model_checkpoint)
    # eval_callback = EvalsCallback(2000, checkpoint_path)
    # checkpoints.append(eval_callback)
    return checkpoints


def train_distributed():
    with config.mirrored_strategy.scope():
        train()


if __name__ == "__main__":
    if config.distributed is False:
        train()
    else:
        train_distributed()
