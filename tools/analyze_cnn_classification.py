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

import autolab_core.utils as utils
from autolab_core import YamlConfig
from perception.models import ClassificationCNN
from perception import ColorImage

from dexnet.learning import ClassificationResult, TensorDataset

def analyze_classification_performance(model_dir, config, dataset_path=None):
    # read params
    predict_batch_size = config['batch_size']
    randomize = config['randomize']
    
    plotting_config = config['plotting']
    figsize = plotting_config['figsize']
    font_size = plotting_config['font_size']
    legend_font_size = plotting_config['legend_font_size']
    line_width = plotting_config['line_width']
    colors = plotting_config['colors']
    dpi = plotting_config['dpi']
    style = '-'

    class_remapping = None
    if 'class_remapping' in config.keys():
        class_remapping = config['class_remapping']

    # read training config
    training_config_filename = os.path.join(model_dir, 'training_config.yaml')
    training_config = YamlConfig(training_config_filename)

    # read training params
    indices_filename = None
    if dataset_path is None:
        dataset_path = training_config['dataset']
        indices_filename = os.path.join(model_dir, 'splits.npz')
    dataset_prefix, dataset_name = os.path.split(dataset_path)
    if dataset_name == '':
        _, dataset_name = os.path.split(dataset_prefix)
    x_names = training_config['x_names']
    y_name = training_config['y_name']
    batch_size = training_config['training']['batch_size']
    iterator_config = training_config['data_iteration']
    x_name = x_names[0]

    # set analysis dir
    analysis_dir = os.path.join(model_dir, 'analysis')
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    # setup log file
    experiment_log_filename = os.path.join(analysis_dir, '%s_analysis.log' %(dataset_name))
    if os.path.exists(experiment_log_filename):
        os.remove(experiment_log_filename)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    hdlr = logging.FileHandler(experiment_log_filename)
    hdlr.setFormatter(formatter)
    logging.getLogger().addHandler(hdlr)

    # setup plotting
    plt.figure(figsize=(figsize, figsize))

    # read dataset
    dataset = TensorDataset.open(dataset_path)

    # read dataset splits
    if indices_filename is None:
        splits = {dataset_name: np.arange(dataset.num_datapoints)}
    else:
        splits = np.load(indices_filename)['arr_0'].tolist()

    # load cnn
    logging.info('Loading model %s' %(model_dir))
    cnn = ClassificationCNN.open(model_dir)

    # save examples
    logging.info('Saving examples of each class')
    all_labels = np.arange(cnn.num_classes)
    label_counts = {}
    [label_counts.update({l:0}) for l in all_labels]
    for tensor_ind in range(dataset.num_tensors):
        tensor = dataset.tensor(y_name, tensor_ind)
        for label in tensor:
            label_counts[label] += 1

    d = utils.sqrt_ceil(cnn.num_classes)
    plt.clf()
    for i, label in enumerate(all_labels):
        tensor_ind = 0
        label_found = False
        while not label_found and tensor_ind < dataset.num_tensors:
            tensor = dataset.tensor(y_name, tensor_ind)
            ind = np.where(tensor.arr == label)[0]
            if ind.shape[0] > 0:
                ind = ind[0] + dataset.datapoints_per_tensor * (tensor_ind)
                label_found = True

            tensor_ind += 1
        
        if not label_found:
            continue
        datapoint = dataset[ind]
        example_im = datapoint[x_name]
        
        plt.subplot(d,d,i+1)
        plt.imshow(example_im[:,:,:3].astype(np.uint8))
        plt.title('Class %03d: %.3f%%' %(label, float(label_counts[label]) / dataset.num_datapoints), fontsize=3)
        plt.axis('off')
    plt.savefig(os.path.join(analysis_dir, '%s_classes.pdf' %(dataset_name)))

    # evaluate on dataset
    results = {}
    for split_name, indices in splits.iteritems():
        logging.info('Evaluating performance on split: %s' %(split_name))

        # predict
        if randomize:
            pred_probs, true_labels = cnn.evaluate_on_dataset(dataset, indices=indices, batch_size=predict_batch_size)
            pred_labels = np.argmax(pred_probs, axis=1)
        else:
            true_labels = []
            pred_labels = []
            pred_probs = []
            for datapoint in dataset:
                im = ColorImage(datapoint['color_ims'].astype(np.uint8)[:,:,:3])
                true_label = datapoint['stp_labels']
                pred_prob = cnn.predict(im)
                pred_label = np.argmax(pred_prob, axis=1)
                true_labels.append(true_label)
                pred_labels.append(pred_label)
                pred_probs.append(pred_prob.ravel())
                
                """
                if class_remapping is not None:
                    true_label = class_remapping[true_label]
                plt.figure()
                plt.imshow(im.raw_data)
                plt.title('T: %d, P: %d' %(true_label, pred_label))
                plt.show()
                """
            true_labels = np.array(true_labels)
            pred_labels = np.array(pred_labels)
            pred_probs = np.array(pred_probs)
                
        # apply optional class re-mapping
        if class_remapping is not None:
            new_true_labels = np.zeros(true_labels.shape)
            for orig_label, new_label in class_remapping.iteritems():
                new_true_labels[true_labels==orig_label] = new_label
            true_labels = new_true_labels

        # compute classification results
        result = ClassificationResult([pred_probs], [true_labels])
        results[split_name] = result

        # print stats
        logging.info('SPLIT: %s' %(split_name))
        logging.info('Acc: %.3f' %(result.accuracy))
        logging.info('AP: %.3f' %(result.ap_score))
        logging.info('AUC: %.3f' %(result.auc_score))

        # save confusion matrix
        confusion = result.confusion_matrix.data
        plt.clf()
        plt.imshow(confusion, cmap=plt.cm.gray, interpolation='none')
        plt.locator_params(nticks=cnn.num_classes)
        plt.xlabel('Predicted', fontsize=font_size)
        plt.ylabel('Actual', fontsize=font_size)
        plt.savefig(os.path.join(analysis_dir, '%s_confusion.pdf' %(split_name)), dpi=dpi)

        # save analysis
        result_filename = os.path.join(analysis_dir, '%s.cres' %(split_name))
        result.save(result_filename)

    # plot
    colormap = plt.get_cmap('tab10')
    num_colors = 9

    plt.clf()
    for i, split_name in enumerate(splits.keys()):
        result = results[split_name]
        precision, recall, taus = result.precision_recall_curve()
        color = colormap(float(colors[i%num_colors]) / num_colors)
        plt.plot(recall, precision, linewidth=line_width, color=color, linestyle=style, label=split_name)
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
        plt.plot(fpr, tpr, linewidth=line_width, color=color, linestyle=style, label=split_name)
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
    
