import matplotlib.pyplot as plt 
import numpy as np

def plot_prediction(energy_ff, pred_residual, energy_true):
    fig, ax = plt.subplots()
    energy_pred = energy_ff+pred_residual

    min_energy = -10+min([np.min(energy_ff), np.min(energy_pred), np.min(energy_true)])
    max_energy = +10+max([np.max(energy_ff), np.max(energy_pred), np.max(energy_true)])

    axis = np.array([min_energy, max_energy])
    ax.fill_between(axis, axis-2.5, axis+2.5, facecolor='yellow')
    ax.plot(axis, axis, c='k')
    ax.plot(energy_ff, energy_true, '.', c='b', label='ff', markersize=5)
    ax.plot(energy_pred, energy_true, '.', c='r', label='ml', markersize=5)
    for i in range(len(energy_pred)):
        ax.plot([energy_ff[i], energy_pred[i]], [energy_true[i], energy_true[i]], '--', linewidth=0.5, c='k')

    ax.set_xlim(min_energy, max_energy)
    ax.set_ylim(min_energy, max_energy)
    ax.set_xlabel('prediction')
    ax.set_ylabel('true')
    ax.legend()
    fig.set_size_inches(12, 12)
    return fig
