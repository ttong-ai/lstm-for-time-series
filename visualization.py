# -*- coding: utf-8 -*-
# visualization.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import os
from sklearn.metrics import roc_curve


def visualize_results(g, results, save_figure=False, save_path='results'):
    """
    Input:
        g: a LSTM or GRU or RNN instance after training
        results: returned results from training
    """
    if g.cell_type == 'LSTM':
        kernel_weights = results['cell_states']['lstm_kernel_weights']
    elif g.cell_type == 'GRU':
        kernel_weights = results['cell_states']['gru_candidate_weights']
    elif g.cell_type == 'RNN':
        kernel_weights = results['cell_states']['rnn_kernel_weights']
    else:
        assert False
    fc_weights = results['fc_states']['weights']

    all_corr_is = results['all_corr_is']
    all_corr_oos = results['all_corr_oos']
    all_actual_is = results['all_actual_is']
    all_actual_oos = results['all_actual_oos']
    all_predicted_is = results['all_predicted_is']
    all_predicted_oos = results['all_predicted_oos']
    all_epochs = results['all_epochs']
    all_losses_per_epoch = results['all_losses_per_epoch']
    all_corr_oos_per_epoch = results['all_corr_oos_per_epoch']

    fig = plt.figure(figsize=(10, 6))
    plt.plot(all_corr_is, label='in-sample corr')
    plt.plot(all_corr_oos, label='out-of-sample corr')
    plt.xlabel('iterations', fontsize=15)
    plt.ylabel('Pearson Correlation', fontsize=15)
    plt.title('In-sample and Out-of-sample Corr through Iterations', fontsize=15)
    plt.legend(loc=0)
    plt.tight_layout()

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True)
    ax[0].plot(all_epochs[:], all_losses_per_epoch[:], 'r', label='loss')
    # ax[0].set_xlabel('epochs')
    ax[0].set_ylabel('MSE Loss')
    ax[1].plot(all_epochs[:], all_corr_oos_per_epoch[:], 'r', label='corr_oos')
    ax[1].set_xlabel('epochs')
    ax[1].set_ylabel('Pearson Correlation')
    plt.tight_layout()

    plt.figure(figsize=(8, 7))
    plt.axvline(x=0.0, color='k', linestyle='--')
    plt.axhline(y=0.0, color='k', linestyle='--')
    plt.scatter(all_actual_is[-1000:], all_predicted_is[-1000:])
    # plt.plot([-0.2, 0.2], [-0.2, 0.2], 'r--');
    plt.xlim(-0.22, 0.22)
    # plt.ylim(-0.22, 0.22);
    plt.xlabel('actual')
    plt.ylabel('predicted')
    plt.title('in-sample fit')
    plt.tight_layout()

    plt.figure(figsize=(8, 7))
    plt.axvline(x=0.0, color='k', linestyle='--')
    plt.axhline(y=0.0, color='k', linestyle='--')
    plt.scatter(all_actual_oos[-1000:], all_predicted_oos[-1000:])
    # plt.plot([-0.2, 0.2], [-0.2, 0.2], 'r--')
    plt.xlim(-0.22, 0.22)
    # plt.ylim(-0.22, 0.22)
    plt.xlabel('actual')
    plt.ylabel('predicted')
    plt.title('in-sample fit')
    plt.tight_layout()

    # Visualization of LSTM cell internal weights
    plt.figure(figsize=(24, 6));
    # cmap = sns.diverging_palette(250, 5, sep=1, as_cmap=True)
    sns.heatmap(kernel_weights, center=0, cmap="bwr", xticklabels=10, yticklabels=10);
    # sns.heatmap(kernel_weights, vmin=-2.0, vmax=2.0, center=0, cmap="bwr", xticklabels=10, yticklabels=10);
    plt.tight_layout();
    if save_figure:
        plt.savefig(os.path.join(save_path, 'kernel_weights.png'))

    y_binary = [1 if i >= 0 else -1 for i in all_actual_oos]
    pred_binary = [1 if i >= 0 else -1 for i in all_predicted_oos]
    fpr, tpr, th = roc_curve(y_binary, pred_binary)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.title('Receiver operating characteristic example', fontsize=15)
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.figure(figsize=(22, 6))
    # plt.plot(np.array(all_predicted), color='b', label='predicted')
    plt.plot(np.array(all_actual_oos)[-300:], color='c', label='actual')
    plt.plot(np.array(all_predicted_oos)[-300:], color='b', label='predicted', alpha=0.5)
    plt.legend(loc=0)
    plt.tight_layout()

    plt.figure(figsize=(12, 6))
    # plt.plot(np.array(all_predicted), color='b', label='predicted')
    plt.plot(np.array(all_actual_oos)[-1500:].cumsum(), color='c', label='actual')
    plt.plot(np.array(all_predicted_oos)[-1500:].cumsum(), color='b', label='predicted', alpha=0.5)
    plt.legend(loc=0)
    plt.tight_layout()

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=False)
    all_actual_oos_range = max(all_actual_oos.max(), np.abs(all_actual_oos.min()))
    all_predicted_oos_range = max(all_predicted_oos.max(), np.abs(all_predicted_oos.min()))
    ax[0].hist(all_actual_oos, bins=11)
    ax[0].set_xlim([-all_actual_oos_range, all_actual_oos_range])
    ax[1].hist(all_predicted_oos, bins=11)
    ax[1].set_xlim([-all_predicted_oos_range, all_predicted_oos_range])
    plt.tight_layout()
