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

def analyze_classification_performance(model_dir, config, dataset_path=None):
    # read params
    plotting_config = config['plotting']
    figsize = plotting_config['figsize']
    font_size = plotting_config['font_size']
    line_width = plotting_config['line_width']
    colors = plotting_config['colors']
    dpi = plotting_config['dpi']

    # read training config
    training_config_filename = os.path.join(model_dir, 'training_config.yaml')
    training_config = YamlConfig(training_config_filename)

    # read training params
    indices_filename = None
    if dataset_path is None:
        dataset_path = training_config['dataset']
        indices_filename = os.path.join(model_dir, 'splits.pkl')
    _, dataset_name = os.path.split(dataset_path)
    x_names = training_config['x_names']
    y_name = training_config['y_name']
    batch_size = training_config['training']['batch_size']
    iterator_config = training_config['data_iteration']
    x_name = x_names[0]

    # set analysis
    analysis_dir = os.path.join(model_dir, 'analysis')
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    # read dataset
    dataset = TensorDataset.open(dataset_path)

    # read dataset splits
    if indices_filename is None:
        splits = {dataset_name: np.arange(dataset.num_datatpoints)}
    else:
        splits = np.load(indices_filename)['arr_0'].tolist()

    # load cnn
    logging.info('Loading model %s' %(model_dir))
    cnn = ClassificationCNN.open(model_dir)

    # evaluate on dataset
    results = {}
    for split_name, indices in splits.iteritems():
        logging.info('Evaluating performance on split: %s' %(split_name))

        # predict
        pred_probs, true_labels = cnn.evaluate_on_dataset(dataset, indices=indices)
        pred_labels = np.argmax(pred_probs, axis=1)

        # compute classification results
        result = ClassificationResult([pred_probs], [true_labels])
        results[split_name] = result

        # analysis
        result_filename = os.path.join(analysis_dir, '%s.cres' %(split_name))
        result.save(result_filename)

    # plot
    colormap = plt.get_cmap('tab10')
    num_colors = 9

    plt.figure(figsize=(figsize, figsize))

    plt.clf()
    for i, split_name in enumerate(splits.keys()):
        result = results[split_name]
        precision, recall, taus = result.precision_recall_curve()
        color = colormap(float(colors[i%num_colors]) / num_colors)
        plt.plot(recall, precision, linewidth=line_width, color=colors[i], linestyle=style, label=split_name)
    plt.xlabel('Recall', fontsize=font_size)
    plt.ylabel('Precision', fontsize=font_size)
    plt.title('Precision-Recall Curve', fontsize=font_size)
    handles, plt_labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, plt_labels, loc='best', fontsize=legend_font_size)
    plt.savefig(os.path.join(analysis_dir, '%s_precision_recall.pdf' %(dataset_name)), dpi=dpi)

    plt.clf()
    for i, split_name in enumerate(splits.keys()):
        result = results[split_name]
        fpr, tpr, taus = result.roc_curve()
        color = colormap(float(colors[i%num_colors]) / num_colors)
        plt.plot(fpr, tpr, linewidth=line_width, color=colors[i], linestyle=style, label=split_name)
    plt.xlabel('FPR', fontsize=font_size)
    plt.ylabel('TPR', fontsize=font_size)
    plt.title('Receiver Operating Characteristic', fontsize=font_size)
    handles, plt_labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, plt_labels, loc='best', fontsize=legend_font_size)
    plt.savefig(os.path.join(analysis_dir, '%s_roc.pdf' %(dataset_name)), dpi=dpi)

if __name__ == '__main__':
    # set logging 
    logging.getLogger().setLevel(logging.INFO)

    # read args
    parser = argparse.ArgumentParser(description='Fine-tune a Classification CNN trained on ImageNet on a custom image dataset using TensorFlow')
    parser.add_argument('model_dir', type=str, default=None, help='directory of the model to use')
    parser.add_argument('config_filename', type=str, default=None, help='path to the configuration file to use')
    parser.add_argument('--dataset_path', type=str, default=None, help='directory of the dataset to evaluate on')
    args = parser.parse_args()
    model_dir = args.model_dir
    dataset_path = args.dataset_path
    config_filename = args.config_filename

    # check valid inputs
    if not os.path.exists(model_dir):
        raise ValueError('Model %s does not exist!' %(model_dir))
    if not os.path.exists(config_filename):
        raise ValueError('Config file %s does not exist!' %(config_filename))
    
    # read config
    config = YamlConfig(config_filename)

    # analyze
    analyze_classification_performance(model_dir, config, dataset_path=dataset_path)
    
