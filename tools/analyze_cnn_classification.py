"""
Analyzes CNN classification performance
Author: Jeff Mahler
"""
import argparse
import IPython
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from autolab_core import YamlConfig
from perception.models import ClassificationCNN

from dexnet.learning import ClassificationResult, TensorDataset

def analyze_classification_performance(model_dir, config):
    # read params
    #plotting_config = config['plotting']

    # read training config
    training_config_filename = os.path.join(model_dir, 'training_config.yaml')
    training_config = YamlConfig(training_config_filename)

    # read training params
    dataset_name = training_config['dataset']
    x_names = training_config['x_names']
    y_name = training_config['y_name']
    batch_size = training_config['training']['batch_size']
    iterator_config = training_config['data_iteration']
    x_name = x_names[0]

    # read dataset
    dataset = TensorDataset.open(dataset_name)

    # read dataset splits
    indices_filename = os.path.join(model_dir, 'splits.npz')
    indices = np.load(indices_filename)['arr_0'].tolist()
    train_indices = indices['train']
    val_indices = indices['val']
    num_train = train_indices.shape[0]
    num_val = val_indices.shape[0]
    train_indices.sort()
    val_indices.sort()

    # load cnn
    logging.info('Loading model %s' %(model_dir))
    cnn = ClassificationCNN.open(model_dir)

    # evaluate on dataset
    logging.info('Evaluating training performance')
    train_pred_probs, train_labels = cnn.evaluate_on_dataset(dataset, indices=train_indices)
    train_pred_labels = np.argmax(train_pred_probs, axis=1)

    logging.info('Evaluating validation performance')
    val_pred_probs, val_labels = cnn.evaluate_on_dataset(dataset, indices=val_indices)
    val_pred_labels = np.argmax(val_pred_probs, axis=1)

    # compute classification results
    train_result = ClassificationResult([train_pred_probs], [train_labels])
    val_result = ClassificationResult([val_pred_probs], [val_labels])

    analysis_dir = os.path.join(model_dir, 'analysis')
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)
    train_result_filename = os.path.join(analysis_dir, 'train_result.cres')
    val_result_filename = os.path.join(analysis_dir, 'val_result.cres')
    train_result.save(train_result_filename)
    val_result.save(val_result_filename)

    IPython.embed()
    exit(0)

    # plot
    plt.figure(figsize=figsize)

    plt.clf()
    precision, recall, taus = result.precision_recall_curve()
    plt.plot(recall, precision, linewidth=line_width, color=color, linestyle=style, label=experiment_name)
    plt.xlabel('Recall', fontsize=font_size)
    plt.ylabel('Precision', fontsize=font_size)
    plt.title('Precision-Recall Curve', fontsize=font_size)
    handles, plt_labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, plt_labels, loc='best', fontsize=legend_font_size)
    plt.savefig(os.path.join(output_dir, 'precision_recall.pdf'), dpi=config['dpi'])

    plt.clf()
    plt.xlabel('FPR', fontsize=font_size)
    plt.ylabel('TPR', fontsize=font_size)
    plt.title('Receiver Operating Characteristic', fontsize=font_size)
    handles, plt_labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, plt_labels, loc='best', fontsize=legend_font_size)
    plt.savefig(os.path.join(output_dir, 'roc.pdf'), dpi=config['dpi'])

    IPython.embed()
    
if __name__ == '__main__':
    # set logging 
    logging.getLogger().setLevel(logging.INFO)

    # read args
    parser = argparse.ArgumentParser(description='Fine-tune a Classification CNN trained on ImageNet on a custom image dataset using TensorFlow')
    parser.add_argument('model_dir', type=str, default=None, help='directory of the model to use')
    parser.add_argument('config_filename', type=str, default=None, help='path to the configuration file to use')
    args = parser.parse_args()
    model_dir = args.model_dir
    config_filename = args.config_filename

    # check valid inputs
    if not os.path.exists(model_dir):
        raise ValueError('Model %s does not exist!' %(model_dir))
    if not os.path.exists(config_filename):
        raise ValueError('Config file %s does not exist!' %(config_filename))
    
    # read config
    config = YamlConfig(config_filename)

    # analyze
    analyze_classification_performance(model_dir, config)
    
