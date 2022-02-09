import os 
import json
import logging
from functools import reduce
from datetime import datetime
from typing import Callable, Tuple, List

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbCallback

from eoflow.models.metrics import MCCMetric
from eoflow.models.segmentation_base import segmentation_metrics
from eoflow.models.losses import JaccardDistanceLoss, TanimotoDistanceLoss

from eoflow.models.segmentation_unets import ResUnetA

from fd.tf_viz_utils import ExtentBoundDistVisualizationCallback
from fd.training import TrainingConfig, get_dataset
from fd.utils import prepare_filesystem

logging.getLogger('tensorflow').disabled = True
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

n_classes = 2
batch_size = 8
n_samples = 15000 # rough estimate

training_config = TrainingConfig(
    bucket_name='',
    aws_access_key_id='',
    aws_secret_access_key='',
    aws_region='',
    wandb_id=None, # change this with your wandb account 
    #npz_folder='input-data/patchlets_npz',
    npz_folder='input-data/patchletsJRCC_npz',
    #metadata_path='input-data/patchlet-info.csv',
    metadata_path='input-data/patchlet-infoJRCC.csv',
    model_folder='input-data/niva-cyl-models',
    model_s3_folder='input-data',
    #chkpt_folder='input-data/pre-trained-model/checkpoints',
    input_shape=(256, 256, 4),
    n_classes=n_classes,
    batch_size=batch_size,
    #iterations_per_epoch=n_samples//batch_size,
    iterations_per_epoch=10,
    #num_epochs=10,
    num_epochs=5,
    model_name='resunet-a',
    reference_names=['extent','boundary','distance'],
    augmentations_feature=['flip_left_right', 'flip_up_down', 'rotate', 'brightness'],
    augmentations_label=['flip_left_right', 'flip_up_down', 'rotate'],
    normalize='to_medianstd',
    n_folds=3,
    model_config={
        'learning_rate': 0.0001,
        'n_layers': 3,
        'n_classes': n_classes,
        'keep_prob': 0.8,
        'features_root': 32,
        'conv_size': 3,
        'conv_stride': 1,
        'dilation_rate': [1, 3, 15, 31],
        'deconv_size': 2,
        'add_dropout': True,
        'add_batch_norm': False,
        'use_bias': False,
        'bias_init': 0.0,
        'padding': 'SAME',
        'pool_size': 3,
        'pool_stride': 2,
        'prediction_visualization': True,
        'class_weights': None
    }
)

#if training_config.wandb_id is not None:
#    !wandb login {training_config.wandb_id}  # EOR

ds_folds = [get_dataset(training_config, fold=fold, augment=True, randomize=True,
                        num_parallel=200, npz_from_s3=False) 
            for fold in range(1, training_config.n_folds+1)]


ds_fold_ex = ds_folds[0].batch(training_config.batch_size)

example_batch = next(iter(ds_fold_ex))

feats = example_batch[0]
lbls = example_batch[1]

feats['features'].shape, lbls['extent'].shape, lbls['boundary'].shape, lbls['distance'].shape 

fig, axs = plt.subplots(nrows=4, ncols=4, sharex='all', sharey='all', figsize=(20, 20))

for nb in np.arange(4):
    axs[nb][0].imshow(feats['features'].numpy()[nb][...,[2,1,0]])
    axs[nb][1].imshow(lbls['extent'].numpy()[nb][..., 1])
    axs[nb][2].imshow(lbls['boundary'].numpy()[nb][..., 1])
    axs[nb][3].imshow(lbls['distance'].numpy()[nb][..., 1])
    
plt.tight_layout()

import os 
import json
import logging
from functools import reduce
from datetime import datetime
from typing import Callable, Tuple, List

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbCallback

from eoflow.models.metrics import MCCMetric
from eoflow.models.segmentation_base import segmentation_metrics
from eoflow.models.losses import JaccardDistanceLoss, TanimotoDistanceLoss

from eoflow.models.segmentation_unets import ResUnetA

from fd.tf_viz_utils import ExtentBoundDistVisualizationCallback
from fd.training import TrainingConfig, get_dataset
from fd.utils import prepare_filesystem

logging.getLogger('tensorflow').disabled = True
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

n_classes = 2
batch_size = 8
n_samples = 15000 # rough estimate

training_config = TrainingConfig(
    bucket_name='',
    aws_access_key_id='',
    aws_secret_access_key='',
    aws_region='',
    wandb_id=None, # change this with your wandb account 
    #npz_folder='input-data/patchlets_npz',
    npz_folder='input-data/patchletsJRCC_npz',
    #metadata_path='input-data/patchlet-info.csv',
    metadata_path='input-data/patchlet-infoJRCC.csv',
    model_folder='input-data/niva-cyl-models',
    model_s3_folder='input-data',
    #chkpt_folder='input-data/pre-trained-model/checkpoints',
    input_shape=(256, 256, 4),
    n_classes=n_classes,
    batch_size=batch_size,
    #iterations_per_epoch=n_samples//batch_size,
    iterations_per_epoch=10,
    #num_epochs=10,
    num_epochs=5,
    model_name='resunet-a',
    reference_names=['extent','boundary','distance'],
    augmentations_feature=['flip_left_right', 'flip_up_down', 'rotate', 'brightness'],
    augmentations_label=['flip_left_right', 'flip_up_down', 'rotate'],
    normalize='to_medianstd',
    n_folds=3,
    model_config={
        'learning_rate': 0.0001,
        'n_layers': 3,
        'n_classes': n_classes,
        'keep_prob': 0.8,
        'features_root': 32,
        'conv_size': 3,
        'conv_stride': 1,
        'dilation_rate': [1, 3, 15, 31],
        'deconv_size': 2,
        'add_dropout': True,
        'add_batch_norm': False,
        'use_bias': False,
        'bias_init': 0.0,
        'padding': 'SAME',
        'pool_size': 3,
        'pool_stride': 2,
        'prediction_visualization': True,
        'class_weights': None
    }
)

#if training_config.wandb_id is not None:
#    !wandb login {training_config.wandb_id}  # EOR

ds_folds = [get_dataset(training_config, fold=fold, augment=True, randomize=True,
                        num_parallel=200, npz_from_s3=False) 
            for fold in range(1, training_config.n_folds+1)]


ds_fold_ex = ds_folds[0].batch(training_config.batch_size)

example_batch = next(iter(ds_fold_ex))

feats = example_batch[0]
lbls = example_batch[1]

feats['features'].shape, lbls['extent'].shape, lbls['boundary'].shape, lbls['distance'].shape 

fig, axs = plt.subplots(nrows=4, ncols=4, sharex='all', sharey='all', figsize=(20, 20))

for nb in np.arange(4):
    axs[nb][0].imshow(feats['features'].numpy()[nb][...,[2,1,0]])
    axs[nb][1].imshow(lbls['extent'].numpy()[nb][..., 1])
    axs[nb][2].imshow(lbls['boundary'].numpy()[nb][..., 1])
    axs[nb][3].imshow(lbls['distance'].numpy()[nb][..., 1])
    
plt.tight_layout()

del ds_fold_ex

def initialise_model(config: TrainingConfig, chkpt_folder: str = None):
    """ Initialise ResUnetA model 
    
    If an existing chekpoints directory is provided, the existing weights are loaded and 
    training starts from existing state
    """
    mcc_metric = MCCMetric(default_n_classes=n_classes, default_threshold=.5)
    mcc_metric.init_from_config({'n_classes': n_classes})
    
    model = ResUnetA(training_config.model_config)
    
    model.build(dict(features=[None] + list(training_config.input_shape)))
    
    model.net.compile(
        loss={'extent':TanimotoDistanceLoss(from_logits=False),
              'boundary':TanimotoDistanceLoss(from_logits=False),
              'distance':TanimotoDistanceLoss(from_logits=False)},
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=training_config.model_config['learning_rate']),
        # comment out the metrics you don't care about
        metrics=[segmentation_metrics['accuracy'](),
                 tf.keras.metrics.MeanIoU(num_classes=training_config.n_classes)])
    
    if chkpt_folder is not None:
        model.net.load_weights(f'{chkpt_folder}/model.ckpt')
        
    return model


def initialise_callbacks(config: TrainingConfig, 
                         fold: int) -> Tuple[str, List[Callable]]:
    """ Initialise callbacks used for logging and saving of models """
    now = datetime.now().isoformat(sep='-', timespec='seconds').replace(':', '-')
    model_path = f'{training_config.model_folder}/{training_config.model_name}_fold-{fold}_{now}'

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    logs_path = os.path.join(model_path, 'logs')
    checkpoints_path = os.path.join(model_path, 'checkpoints', 'model.ckpt')


    # Tensorboard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_path,
                                                          update_freq='epoch',
                                                          profile_batch=0)

    # Checkpoint saving callback
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoints_path,
                                                             save_best_only=True,
                                                             save_freq='epoch',
                                                             save_weights_only=True)

    full_config = dict(**training_config.model_config, 
                       iterations_per_epoch=training_config.iterations_per_epoch, 
                       num_epochs=training_config.num_epochs, 
                       batch_size=training_config.batch_size,
                       model_name=f'{training_config.model_name}_{now}'
                      )

    # Save model config 
    with open(f'{model_path}/model_cfg.json', 'w') as jfile:
        json.dump(training_config.model_config, jfile)

    # initialise wandb if used
    if training_config.wandb_id:
        wandb.init(config=full_config, 
                   name=f'{training_config.model_name}-leftoutfold-{fold}',
                   project="field-delineation", 
                   sync_tensorboard=True)
        
    callbacks = [tensorboard_callback, 
                 checkpoint_callback, 
                ] + ([WandbCallback()] if training_config.wandb_id is not None else [])
    
    return model_path, callbacks 

folds = list(range(training_config.n_folds))

folds_ids_list = [(folds[:nf] + folds[1 + nf:], [nf]) for nf in folds]
folds_ids_list

np.random.seed(training_config.seed)

models = []
model_paths = []

for training_ids, testing_id in folds_ids_list:
    
    left_out_fold = testing_id[0]+1
    print(f'Training model for left-out fold {left_out_fold}')
    
    # Create datasets for this fold, 3 folds for training, 1 for validation
    fold_val = np.random.choice(training_ids)
    folds_train = [tid for tid in training_ids if tid != fold_val]
    print(f'Trai folds {folds_train}, Val fold: {fold_val}, Test fold: {testing_id[0]}')
    
    ds_folds_train = [ds_folds[tid] for tid in folds_train]
    ds_train = reduce(tf.data.Dataset.concatenate, ds_folds_train)
        
    ds_val = ds_folds[fold_val]
    
    ds_val = ds_val.batch(training_config.batch_size)
    
    ds_train = ds_train.batch(training_config.batch_size)
    ds_train = ds_train.repeat()
    
    # Get model
    model = initialise_model(training_config, chkpt_folder=training_config.chkpt_folder)
    
    # Set up callbacks to monitor training
    model_path, callbacks = initialise_callbacks(training_config, 
                                                 fold=left_out_fold)
    
    print(f'\tTraining model, writing to {model_path}')

    model.net.fit(ds_train, 
                  validation_data=ds_val,
                  epochs=training_config.num_epochs,
                  steps_per_epoch=training_config.iterations_per_epoch,
                  callbacks=callbacks)

    models.append(model)
    model_paths.append(model_path)
    
    del fold_val, folds_train, ds_train, ds_val

from fs.copy import copy_dir

#filesystem = prepare_filesystem(training_config)

import shutil

for model_path in model_paths:
    model_name = os.path.basename(model_path)
    print('model_name:', model_name)
    #
    # os.makedirs(f'{training_config.model_s3_folder}/{model_name}', recreate=True)
    m_pth = f'{training_config.model_s3_folder}/{model_name}'
    print('m_pth: ', m_pth)
    if not os.path.exists(m_pth):
        os.makedirs(m_pth)
    else:
        shutil.rmtree(m_pth)
        os.makedirs(m_pth)

    # copy_dir(training_config.model_folder,
    #          f'{model_name}',
    #          filesystem,
    #          f'{training_config.model_s3_folder}/{model_name}')

    copy_dir(training_config.model_folder,
             f'{model_name}',
             f'{training_config.model_s3_folder}',
             f'{model_name}')

weights = [model.net.get_weights() for model in models]

avg_weights = list()

for weights_list_tuple in zip(*weights):
    avg_weights.append(np.array([np.array(weights_).mean(axis=0) 
                        for weights_ in zip(*weights_list_tuple)]))

avg_model = ResUnetA(training_config.model_config)
    
avg_model.build(dict(features=[None] + list(training_config.input_shape)))

avg_model.net.compile(
        loss={'extent':TanimotoDistanceLoss(from_logits=False),
              'boundary':TanimotoDistanceLoss(from_logits=False),
              'distance':TanimotoDistanceLoss(from_logits=False)},
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=training_config.model_config['learning_rate']),
        # comment out the metrics you don't care about
        metrics=[segmentation_metrics['accuracy'](),
                 tf.keras.metrics.MeanIoU(num_classes=training_config.n_classes)])

avg_model.net.set_weights(avg_weights)

now = datetime.now().isoformat(sep='-', timespec='seconds').replace(':', '-')
model_path = f'{training_config.model_folder}/{training_config.model_name}_avg_{now}'

if not os.path.exists(model_path):
    os.makedirs(model_path)

checkpoints_path = os.path.join(model_path, 'checkpoints', 'model.ckpt')

# Save model config 
with open(f'{model_path}/model_cfg.json', 'w') as jfile:
    json.dump(training_config.model_config, jfile)

avg_model.net.save_weights(checkpoints_path)

test_batch = next(iter(ds_folds[1].batch(batch_size)))

model = models[1]

predictions = model.net.predict(test_batch[0]['features'].numpy())

n_images = 8

fig, axs = plt.subplots(nrows=n_images, ncols=5, 
                        sharex='all', sharey='all', 
                        figsize=(15, 3*n_images))

for nb in np.arange(n_images):
    axs[nb][0].imshow(test_batch[0]['features'].numpy()[nb][...,[2,1,0]])
    axs[nb][1].imshow(predictions[0][nb][..., 1])
    axs[nb][2].imshow(predictions[1][nb][..., 1])
    axs[nb][3].imshow(predictions[2][nb][..., 1])
    axs[nb][4].imshow(test_batch[1]['extent'].numpy()[nb][..., 1])
    
plt.tight_layout()


predictions = avg_model.net.predict(test_batch[0]['features'].numpy())

n_images = 8

fig, axs = plt.subplots(nrows=n_images, ncols=5, 
                        sharex='all', sharey='all', 
                        figsize=(15, 3*n_images))

for nb in np.arange(n_images):
    axs[nb][0].imshow(test_batch[0]['features'].numpy()[nb][...,[2,1,0]])
    axs[nb][1].imshow(predictions[0][nb][..., 1])
    axs[nb][2].imshow(predictions[1][nb][..., 1])
    axs[nb][3].imshow(predictions[2][nb][..., 1])
    axs[nb][4].imshow(test_batch[1]['extent'].numpy()[nb][..., 1])
    
plt.tight_layout()

for _, testing_id in folds_ids_list:
    
    left_out_fold = testing_id[0]+1
    print(f'Evaluating model on left-out fold {left_out_fold}')
    
    model = models[testing_id[0]]
    model.net.evaluate(ds_folds[testing_id[0]].batch(training_config.batch_size))
    
    print(f'Evaluating average model on left-out fold {left_out_fold}')
    avg_model.net.evaluate(ds_folds[testing_id[0]].batch(training_config.batch_size))
    
    print('\n\n')
