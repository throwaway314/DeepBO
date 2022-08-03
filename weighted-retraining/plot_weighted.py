import numpy as np
import matplotlib.pyplot as plt
import os
from lib import file_io as fio
from phc import level_set
import tensorflow as tf


import matplotlib
matplotlib.rcParams.update({'font.size': 12, 'xtick.labelsize': 8, 'ytick.labelsize': 8})
import matplotlib.gridspec as gridspec
# plt.style.use('ggplot')
# plt.rcParams['axes.facecolor'] = 'w'


def min_so_far(y):
    new_array = np.zeros(y.shape)
    for i in range(len(y)):
        new_array[i] = np.min(y[:i+1])
    return new_array


def load_trials(dir_name, n):
    y_new_arr = []
    data = None
    for i in range(n):
        data = np.loadtxt(os.path.join(dir_name, 'eval%d' % i), skiprows=1)
        y_new = min_so_far(data[:, 1])
        if y_new.size < 505:
            y_new = np.append(y_new, np.ones(505-y_new.size)*y_new[-1])
        y_new_arr.append(y_new)
    return data[:, 0], np.array(y_new_arr)


def load_trials_best(dir_name, n):
    x_best = []
    for i in range(n):
        x_i = []
        with open(os.path.join(dir_name, 'report%d' % i)) as f:
            lines = f.readlines()[31:-1]
            line = lines[0].split()
            nums = line[4:]
            x_i.extend([float(i) for i in nums])
            for line in lines[1:]:
                x_i.extend([float(i) for i in line.split()])
        x_best.append(x_i)
    return np.array(x_best)


fig = plt.figure(figsize=(7, 3))


def plot_error(n_data, best_y, label, color=None, linestyle='-'):
    n_trials = np.shape(best_y)[0]
    se = np.std(best_y, axis=0) / np.sqrt(n_trials)  # Standard error
    plt.plot(n_data, np.mean(best_y, axis=0), label=label, color=color, linestyle=linestyle)
    plt.fill_between(n_data, np.mean(best_y, axis=0) - se,
                     np.mean(best_y, axis=0) + se, alpha=0.1,
                     facecolor=color)


colors = [[94, 60, 153], [44, 123, 182]]
colors = np.array(colors) / 255
colors = [[215, 25, 28], [44, 123, 182]]
colors = np.array(colors) / 255

data = None

plt.subplot(1, 2, 1)

rand = []
for i in range(1,6):
    data = np.load('logs/opt/chem-alpha/seed%d/gp/iter0/data.npz' % i)
    data = -min_so_far(data['y_train'])
    rand.append(data.ravel())
plot_error(range(505), rand, label='Random', color='k', linestyle='dotted')

# plt.legend()
plt.xlabel('$N$')
plt.ylabel('$y_{best}$')
plt.title(r'$\alpha$')
plt.xlim([0, 500])

plt.subplot(1, 2, 2)

rand = []
for i in range(1,6):
    data = np.load('logs/opt/chem-mingapalpha/seed%d/gp/iter0/data.npz' % i)
    data = -min_so_far(data['y_train'])
    rand.append(data.ravel())
plot_error(range(505), rand, label='Random', color='k', linestyle='dotted')


plt.xlabel('$N$')
plt.ylabel('$y_{best}$')
plt.title(r'$\alpha-\epsilon$')
plt.xlim([0, 500])

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))


plt.tight_layout()
plt.show()

