import os
import json
import time
import fire
import hypertune
import pandas as pd
from fastai.vision.all import *

timestamp = time.strftime("%Y%m%d-%H%M%S")
TEST_THR_DEFAULT=0.5

def train_evaluate(job_dir: str = None,
                   training_dataset_path: str = None,
                   training_images_path: str = None,
                   model_version: str = None,
                   TAG_COLUMN: str = 'tag',
                   RESIZE_VALUE: int = None,
                   METHOD: str = None,
                   AUG_SIZE: int = None,
                   FREEZE_EPOCHS: int = None,
                   BASE_LR: float = None,
                   EPOCHS:int = None,
                   ARCH_RESNET: str = 'resnet50'
                   ):
    """

    :param job_dir: Directory to export model and metrics
    :param training_dataset_path: GCS path for csv with labelled data includes at least [image_id, tag]
    :param training_images_path: GCS path to images 
    :param model_version: model version
    :param TAG_COLUMN: column name in training data csv including labels
    :param RESIZE_VALUE: 
    :param METHOD: 
    :param AUG_SIZE:
    :param FREEZE_EPOCHS: 
    :param BASE_LR: 
    :param EPOCHS: 
    :param ARCH_RESNET: resnet architecture, defaults to 'resnet50'
    """
    
    if torch.cuda.is_available():
        print('GPU available')
    
    ## GCS gets mounted when the instance starts using gcsfuse. All prefixes exist under /gcs/
    image_dir_train=training_images_path.replace('gs://', '/gcs/')
    job_dir_gcs=job_dir.replace('gs://', '/gcs/')
    job_dir_gcs_export = f'{job_dir_gcs}/{model_version}_{timestamp}'
    os.mkdir(job_dir_gcs_export)
    
    print('Reading csv ...')
    df_train = pd.read_csv(training_dataset_path)

    print('Loading and transforming images...')
    ## define datablock
    img_block_01 = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   splitter=ColSplitter('is_valid'),
                   get_x=ColReader('image_id', pref=f'{image_dir_train}/', suff=''),
                   get_y=ColReader(TAG_COLUMN, label_delim=','),
                   item_tfms=Resize(RESIZE_VALUE, method = METHOD), 
                   batch_tfms=aug_transforms(size=AUG_SIZE, do_flip = False))
    
    # load images and transform
    img_dls_01 = img_block_01.dataloaders(df_train)
    
    print('Starting training...')
    fscore = F1ScoreMulti()
    learn_01 = cnn_learner(dls = img_dls_01, arch = eval(ARCH_RESNET), metrics = [accuracy_multi, fscore])
    learn_01.fine_tune(
        EPOCHS, base_lr=BASE_LR, freeze_epochs=FREEZE_EPOCHS,
        cbs = [SaveModelCallback(monitor='accuracy_multi', fname = 'model', with_opt = True)]
        )
    print('Training completed.')
    
    ## dict to add metrics
    metrics_log={}
    
    ## Get final model metrics
    # here accuracy_multi is maximized (can be changed to valid_loss or any other metric)
    accuracy_multi_train = learn_01.recorder.metrics[0].value.item()
    metrics_log['accuracy_multi_train'] = accuracy_multi_train
    
    print("Saving metrics")
    with open(f'{job_dir_gcs_export}/metrics_log.json', "w") as f:
        json.dump(metrics_log, f)

    ## save model
    print("Exporting metrics")
    learn_01.version = model_version  # add model version to saved model
    learn_01.resize_value = RESIZE_VALUE # add resize value
    model_name = f'model_{model_version}.pkl'

    print(f'Saving model to {job_dir_gcs_export}/{model_name}')
    learn_01.export(f'{job_dir_gcs_export}/{model_name}')

if __name__ == "__main__":
    fire.Fire(train_evaluate)