# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:29:32 2019

@author: kuangen
"""
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# plt.rcParams['pdf.fonttype'] = 42
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus']=False

import pandas as pd
import openpyxl
import numpy as np
import glob
import xlrd
import cv2
import sys
# sys.path.insert(0,'../')
from code.pytorch.utils.utils import *
from scipy import stats, signal
from time import time
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets


""" ================================= Plot result ===================================== """
YLABEL = ['Fx(N)', 'Fy(N)', 'Fz(N)', 'Mx(Nm)', 'My(Nm)', 'Mz(Nm)']
Title = ["X axis force", "Y axis force", "Z axis force",
         "X axis moment", "Y axis moment", "Z axis moment"]
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
        'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
        'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']
""" ================================================================================= """

FONT_SIZE = 20
# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20 # 16


def plot_force_and_moment(path_2, path_3):

    font_size_new=30
    V_force = np.load(path_2)
    V_state = np.load(path_3)

    plt.figure(figsize=(20, 10), dpi=300)
    plt.tight_layout(pad=3, w_pad=1., h_pad=0.5)
    plt.subplots_adjust(left=0.08, bottom=0.10, right=0.98, top=0.98, wspace=0.23, hspace=0.23)

    plt.subplot(1, 2, 1)
    # plt.title("Forces and Moments", fontsize=FONT_SIZE)
    plt.plot(V_force[:, : 6], linewidth=2.75)
    plt.xlabel("Steps", fontsize=font_size_new)
    plt.ylabel("Forces(N) / Moments(10XNm)", fontsize=font_size_new)
    plt.legend(labels=['$F_x$', '$F_y$', '$F_z$', '$M_x$', '$M_y$', '$M_z$'], loc='best', fontsize=font_size_new)
    plt.xticks(fontsize=font_size_new)
    plt.yticks(fontsize=font_size_new)
    plt.grid()

    plt.subplot(1, 2, 2)
    # plt.title("Search Result of State", fontsize=FONT_SIZE)
    # plt.plot(V_state[:, 6:] - [539.8759, -39.7005, 193, 179.8834, 1.3056,  -5.4893])
    plt.plot(V_state[:, 6:] - V_state[0, 6:], linewidth=2.75)
    plt.xlabel("Steps", fontsize=font_size_new)
    plt.ylabel("Position(mm) / Orientation$(\circ)$", fontsize=font_size_new)
    plt.legend(labels=['$P_x$', '$P_y$', '$P_z$', '$O_x$', '$O_y$', '$O_z$'], loc='best', fontsize=font_size_new)
    plt.xticks(fontsize=font_size_new)
    plt.yticks(fontsize=font_size_new)
    plt.grid()

    plt.savefig('./figure/pdf/impedance_controller_single_episode_force_moment.pdf')
    plt.savefig('./figure/jpg/impedance_controller_single_episode_force_moment.jpg')

    plt.show()


def plot_six_action(path_2, path_3):

    font_size_new=30
    V_force = np.load(path_2)
    V_state = np.load(path_3)

    plt.figure(figsize=(20, 10), dpi=300)
    plt.tight_layout(pad=3, w_pad=1., h_pad=0.5)
    plt.subplots_adjust(left=0.08, bottom=0.10, right=0.98, top=0.98, wspace=0.23, hspace=0.23)

    plt.subplot(1, 2, 1)
    # plt.title("Search Result of Force", fontsize=font_size_new)
    plt.plot(V_force[:, :], linewidth=2.75)
    plt.xlabel("Steps", fontsize=font_size_new)
    plt.ylabel("Action(mm/$\circ$)", fontsize=font_size_new)
    plt.legend(labels=['$\Delta_x$', '$\Delta_y$', '$\Delta_z$', '$\Theta_x$', '$\Theta_y$', '$\Theta_z$'], loc='best', fontsize=font_size_new)
    plt.xticks(fontsize=font_size_new)
    plt.yticks(fontsize=font_size_new)
    plt.grid()

    plt.subplot(1, 2, 2)
    # plt.title("Search Result of State", fontsize=FONT_SIZE)
    # plt.plot(V_state[:, 6:] - [539.8759, -39.7005, 193, 179.8834, 1.3056,  -5.4893])
    plt.plot(V_state[:, 6:] - V_state[0, 6:], linewidth=2.75)
    plt.xlabel("Steps", fontsize=font_size_new)
    plt.ylabel("Position(mm) / Orientation$(\circ)$", fontsize=font_size_new)
    plt.legend(labels=['$P_x$', '$P_y$', '$P_z$', '$O_x$', '$O_y$', '$O_z$'], loc='best', fontsize=font_size_new)
    plt.xticks(fontsize=font_size_new)
    plt.yticks(fontsize=font_size_new)
    plt.grid()

    plt.savefig('./figure/pdf/impedance_controller_single_episode_actions.pdf')
    plt.savefig('./figure/jpg/impedance_controller_single_episode_actions.jpg')

    plt.show()


def plot_learning_force_and_moment(path_2, path_3, name):

    font_size = 34
    V_force = np.load(path_2)
    V_state = np.load(path_3)

    initial_position = np.array([539.88427, -38.68679, 190.03184, 179.88444 - 180, 1.30539, 0.21414])

    high = np.array([40, 40, 0, 5, 5, 5, 542, -36, 192, 5, 5, 5])
    low = np.array([-40, -40, -40, -5, -5, -5, 538, -42, 188, -5, -5, -5])
    scale = high - low
    length = 22
    v_forces = np.zeros([len(V_force), length, 6])
    for i in range(len(V_force)):
        for j in range(length):
            for m in range(6):
                v_forces[i, j, m] = np.array(V_force[i])[j, m]

    v_states = np.zeros([len(V_state), length, 6])
    for i in range(len(V_state)):
        for j in range(length):
            for m in range(6):
                v_states[i, j, m] = np.array(V_state[i])[j, m+6]
    mean_force = np.mean(v_forces, axis=0)
    std_force = np.std(v_forces, axis=0)
    mean_state = np.mean(v_states, axis=0)
    std_state = np.std(v_states, axis=0)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    plt.figure(figsize=(24, 10), dpi=100)

    # plt.tight_layout(pad=3, w_pad=1., h_pad=0.5)
    plt.subplots_adjust(left=0.08, bottom=0.10, right=0.98, top=0.88, wspace=0.4, hspace=0.23)
    plt.subplot(1, 2, 1)
    for num in range(6):
        if num > 2:
            plt.plot((mean_force[:, num] * scale[num] + low[num]) * 10, linewidth=2.75)
            plt.fill_between(np.arange(len(mean_force[:, 0])),
                             ((mean_force[:, num] - std_force[:, num]) * scale[num] + low[num]) * 10,
                             ((mean_force[:, num] + std_force[:, num]) * scale[num] + low[num]) * 10, alpha=0.3)
        else:
            plt.plot(mean_force[:, num] * scale[num] + low[num], linewidth=2.75)
            plt.fill_between(np.arange(len(mean_force[:, 0])),
                             (mean_force[:, num] - std_force[:, num]) * scale[num] + low[num],
                             (mean_force[:, num] + std_force[:, num]) * scale[num] + low[num], alpha=0.3)
        # plt.plot(mean_force[:, num] * scale[num] + low[num], linewidth=2.)
        # plt.fill_between(np.arange(len(mean_force[:, 0])),
        #                  (mean_force[:, num] - std_force[:, num]) * scale[num] + low[num],
        #                  (mean_force[:, num] + std_force[:, num]) * scale[num] + low[num], alpha=0.3)

    # plt.xlabel("Steps", fontsize=30)
    # plt.ylabel("Forces$(N)$ / Moments$(10XNm)$", fontsize=30)
    # plt.legend(labels=['$F_x$', '$F_y$', '$F_z$', '$M_x$', '$M_y$', '$M_z$'], loc='lower right', fontsize=30)
    # plt.xticks(fontsize=30)
    # plt.yticks(fontsize=30)

    # chinese
    plt.xlabel("装配步数", fontsize=font_size)
    plt.ylabel("接触力(N)/接触力矩(10XNm)", fontsize=font_size)
    plt.legend(labels=['$F_x$', '$F_y$', '$F_z$', '$M_x$', '$M_y$', '$M_z$'], loc=2, bbox_to_anchor=(0.12, 1.15), borderaxespad=0., fontsize=30, ncol=3)

    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    plt.subplot(1, 2, 2)
    for num in range(6):
        plt.plot(mean_state[:, num] * scale[num + 6] + low[num + 6] - initial_position[num], linewidth=2.75)
        plt.fill_between(np.arange(len(mean_state[:, 0])),
                         (mean_state[:, num] - std_state[:, num]) * scale[num + 6] + low[num + 6] - initial_position[num],
                         (mean_state[:, num] + std_state[:, num]) * scale[num + 6] + low[num + 6] - initial_position[num], alpha=0.3)
    # plt.xlabel("Steps", fontsize=30)
    # plt.ylabel("Position$(mm)$ / Orientation$(\circ)$", fontsize=30)
    # plt.legend(labels=['$P_x$', '$P_y$', '$P_z$', '$O_x$', '$O_y$', '$O_z$'], loc='lower right', fontsize=30)
    # plt.xticks(fontsize=30)
    # plt.yticks(fontsize=30)

    # chinese
    plt.xlabel("装配步数", fontsize=font_size)
    plt.ylabel("双轴位置(mm)/姿态($\circ$)", fontsize=font_size)
    plt.legend(labels=['$P_x$', '$P_y$', '$P_z$', '$O_x$', '$O_y$', '$O_z$'], loc=2, bbox_to_anchor=(0.12, 1.15), borderaxespad=0., fontsize=30, ncol=3)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    # plt.savefig('./figure/pdf/chinese_' + name + '.pdf')
    plt.show()


def plot_chinese_learning_force_and_moment(path_2, path_3, name):

    font_size = 30
    V_force = np.load(path_2)
    V_state = np.load(path_3)

    initial_position = np.array([539.88427, -38.68679, 190.03184, 179.88444 - 180, 1.30539, 0.21414])

    high = np.array([40, 40, 0, 5, 5, 5, 542, -36, 192, 5, 5, 5])
    low = np.array([-40, -40, -40, -5, -5, -5, 538, -42, 188, -5, -5, -5])
    scale = high - low
    length = 38
    v_forces = np.zeros([len(V_force), length, 6])
    for i in range(len(V_force)):
        for j in range(length):
            for m in range(6):
                v_forces[i, j, m] = np.array(V_force[i])[j, m]

    v_states = np.zeros([len(V_state), length, 6])
    for i in range(len(V_state)):
        for j in range(length):
            for m in range(6):
                v_states[i, j, m] = np.array(V_state[i])[j, m+6]
    mean_force = np.mean(v_forces, axis=0)
    std_force = np.std(v_forces, axis=0)
    mean_state = np.mean(v_states, axis=0)
    std_state = np.std(v_states, axis=0)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    plt.figure(figsize=(10, 8), dpi=300)

    # plt.tight_layout(pad=3, w_pad=1., h_pad=0.5)
    plt.tight_layout(pad=4.9, w_pad=1., h_pad=1.)
    plt.subplots_adjust(left=0.12, bottom=0.12, right=0.98, top=0.88, wspace=0.23, hspace=0.23)
    plt.subplot(1, 1, 1)
    for num in range(6):
        if num > 2:
            plt.plot((mean_force[:, num] * scale[num] + low[num]) * 10, linewidth=2.75)
            plt.fill_between(np.arange(len(mean_force[:, 0])),
                             ((mean_force[:, num] - std_force[:, num]) * scale[num] + low[num]) * 10,
                             ((mean_force[:, num] + std_force[:, num]) * scale[num] + low[num]) * 10, alpha=0.3)
        else:
            plt.plot(mean_force[:, num] * scale[num] + low[num], linewidth=2.75)
            plt.fill_between(np.arange(len(mean_force[:, 0])),
                             (mean_force[:, num] - std_force[:, num]) * scale[num] + low[num],
                             (mean_force[:, num] + std_force[:, num]) * scale[num] + low[num], alpha=0.3)
        # plt.plot(mean_force[:, num] * scale[num] + low[num], linewidth=2.)
        # plt.fill_between(np.arange(len(mean_force[:, 0])),
        #                  (mean_force[:, num] - std_force[:, num]) * scale[num] + low[num],
        #                  (mean_force[:, num] + std_force[:, num]) * scale[num] + low[num], alpha=0.3)

    # plt.xlabel("Steps", fontsize=30)
    # plt.ylabel("Forces$(N)$ / Moments$(10XNm)$", fontsize=30)
    # plt.legend(labels=['$F_x$', '$F_y$', '$F_z$', '$M_x$', '$M_y$', '$M_z$'], loc='lower right', fontsize=30)
    # plt.xticks(fontsize=30)
    # plt.yticks(fontsize=30)

    # chinese
    plt.xlabel("装配步数", fontsize=font_size)
    plt.ylabel("接触力(N)/接触力矩(10XNm)", fontsize=font_size)
    plt.legend(labels=['$F^x$', '$F^y$', '$F^z$', '$M^x$', '$M^y$', '$M^z$'], loc=2, bbox_to_anchor=(0.1, 1.15), borderaxespad=0., fontsize=30, ncol=3)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    # plt.subplot(1, 2, 2)

    # for num in range(6):
    #     plt.plot(mean_state[:, num] * scale[num + 6] + low[num + 6] - initial_position[num], linewidth=2.75)
    #     plt.fill_between(np.arange(len(mean_state[:, 0])),
    #                      (mean_state[:, num] - std_state[:, num]) * scale[num + 6] + low[num + 6] - initial_position[num],
    #                      (mean_state[:, num] + std_state[:, num]) * scale[num + 6] + low[num + 6] - initial_position[num], alpha=0.3)

    # # plt.xlabel("Steps", fontsize=30)
    # # plt.ylabel("Position$(mm)$ / Orientation$(\circ)$", fontsize=30)
    # # plt.legend(labels=['$P_x$', '$P_y$', '$P_z$', '$O_x$', '$O_y$', '$O_z$'], loc='lower right', fontsize=30)
    # # plt.xticks(fontsize=30)
    # # plt.yticks(fontsize=30)
    #
    # chinese
    # plt.xlabel("装配步数", fontsize=font_size)
    # plt.ylabel("双轴位置(mm)/姿态($\circ$)", fontsize=font_size)
    # plt.legend(labels=['$P^x$', '$P^y$', '$P^z$', '$O^x$', '$O^y$', '$O^z$'], loc=2, bbox_to_anchor=(0.1, 1.15), borderaxespad=0., fontsize=30, ncol=3)
    # plt.xticks(fontsize=font_size)
    # plt.yticks(fontsize=font_size)

    plt.savefig('./figure/pdf/single_chinese_' + name + '_force_moment.pdf')
    # plt.savefig('./figure/pdf/single_chinese_' + name + '_position_orientation.pdf')
    # plt.show()


def plot_raw_data(path_1):
    data = np.load(path_1)
    force_m = np.zeros((len(data), 12))

    plt.figure(figsize=(20, 20), dpi=100)
    plt.tight_layout(pad=3, w_pad=0.5, h_pad=1.0)
    plt.subplots_adjust(left=0.065, bottom=0.1, right=0.995, top=0.9, wspace=0.2, hspace=0.2)
    plt.title("True Data")
    for j in range(len(data)):
        force_m[j] = data[j, 0]
    k = -1
    for i in range(len(data)):
        if data[i, 1] == 0:
            print("===========================================")
            line = force_m[k+1:i+1]
            print(line)
            k = i
            for j in range(6):
                plt.subplot(2, 3, j + 1)
                plt.plot(line[:, j])
                # plt.plot(line[:, 0])

                if j == 1:
                    plt.ylabel(YLABEL[j], fontsize=17.5)
                    plt.xlabel('steps', fontsize=20)
                    plt.xticks(fontsize=15)
                    plt.yticks(fontsize=15)

                else:
                    plt.ylabel(YLABEL[j], fontsize=20)
                    plt.xlabel('steps', fontsize=20)
                    plt.xticks(fontsize=15)
                    plt.yticks(fontsize=15)
        i += 1
    plt.savefig('raw_data_random_policy1.jpg')


def plot_compare(fuzzy_path, none_fuzzy_path, figure_path):

    reward_fuzzy = np.load(fuzzy_path)
    reward_none_fuzzy = np.load(none_fuzzy_path)

    # dfs = pd.read_excel(fuzzy_path)
    # reward_fuzzy = dfs.values.astype(np.float)[:, 0]
    # dfs = pd.read_excel(none_fuzzy_path)
    # reward_none_fuzzy = dfs.values.astype(np.float)[:, 0]

    plt.figure(figsize=(10, 8), dpi=300)
    plt.subplot(1, 1, 1)
    plt.tight_layout(pad=4.9, w_pad=1., h_pad=1.)
    plt.subplots_adjust(left=0.165, bottom=0.15, right=0.98, top=0.98, wspace=0.23, hspace=0.22)

    CHINESE = False

    if CHINESE:
        plt.plot(np.arange(len(reward_none_fuzzy) - 1), np.array(reward_fuzzy[1:]), color='b', linewidth=2.5,
                 label='DDPG')
        plt.plot(np.arange(len(reward_none_fuzzy) - 1), np.array(reward_none_fuzzy[1:]), color='r', linewidth=2.5,
                 label='基于环境预测知识优化DDPG')
        plt.ylabel('每回合装配累计奖励值', fontsize=34)
        plt.xlabel('训练回合数', fontsize=34)
    else:
        plt.plot(np.array(reward_fuzzy[1:223]), color='b',
                 linewidth=5., label='Typical DQN')
        plt.plot(np.array(reward_none_fuzzy[1:223]), color='r',
                 linewidth=5., label='FLDVTSP—DQN')
        plt.ylabel('Episode Reward', fontsize=34)
        plt.xlabel('Episodes', fontsize=34)

    plt.yticks(fontsize=34)
    plt.xticks(fontsize=34)
    # plt.ylabel('每回合装配步数', fontsize=FONT_SIZE)
    plt.legend(fontsize=30, loc='upper left')

    # plot_reward('./episode_rewards_100.npy')
    # plt.figure(figsize=(15, 15), dpi=100)
    # plt.subplot(2, 1, 2)
    # plt.title('DQN With Knowledge')
    # plt.plot(np.arange(len(reward_fuzzy) - 1), np.array(reward_fuzzy[1:] * 10), color='b')
    # plt.ylabel('Episode Reward', fontsize=20)
    # plt.xlabel('Episodes', fontsize=20)

    plt.savefig(figure_path)
    # plt.savefig('./results/figure/jpg/dqn_test_episode_reward.jpg')
    # plt.savefig('./figure/pdf/chinese_ddpg_episode_step.pdf')
    # plt.savefig('./figure/jpg/chinese_ddpg_episode_step.jpg')

    plt.show()


def plot_continuous_data(path):
    raw_data = np.load(path)
    print(raw_data)
    plt.figure(figsize=(20, 15))
    plt.title('Episode Reward')
    plt.tight_layout(pad=3, w_pad=0.5, h_pad=1.0)
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.98, top=0.9, wspace=0.23, hspace=0.22)
    # plt.subplots_adjust(left=0.065, bottom=0.1, right=0.995, top=0.9, wspace=0.2, hspace=0.2)
    data = np.zeros((len(raw_data), 12))
    for j in range(len(raw_data)):
        data[j] = raw_data[j, 0]
    for j in range(6):
        plt.subplot(2, 3, j + 1)
        plt.plot(data[:, j], linewidth=2.5, color='r')
        # plt.ylabel(YLABEL[j], fontsize=18)
        if j>2:
            plt.xlabel('steps', fontsize=30)
        plt.title(YLABEL[j],fontsize=30)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
    # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.2)
    # plt.savefig('raw_data.pdf')
    plt.show()


def chinese_plot_compare_raw_data(path1, path2):
    raw_data = np.load(path1)
    raw_data_1 = np.load(path2)
    plt.figure(figsize=(20, 12), dpi=1000)
    plt.title('Episode Reward')
    plt.tight_layout(pad=3, w_pad=0.5, h_pad=1.0)
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.95, wspace=0.33, hspace=0.15)
    data = np.zeros((len(raw_data), 12))
    for j in range(len(raw_data)):
        data[j] = raw_data[j, 0]

    data_1 = np.zeros((len(raw_data_1), 12))
    for j in range(len(raw_data_1)):
        data_1[j] = raw_data_1[j, 0]

    for j in range(6):
        plt.subplot(2, 3, j + 1)
        plt.plot(data[:100, j], linewidth=2.5, color='r', linestyle='--')
        plt.plot(data_1[:100, j], linewidth=2.5, color='b')
        # plt.ylabel(YLABEL[j], fontsize=18)

        if j > 2:
            plt.xlabel('搜索步数', fontsize=38)
        plt.title(YLABEL[j], fontsize=38)
        plt.xticks(fontsize=38)
        plt.yticks(fontsize=38)
    # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.2)
    plt.savefig('./figures/chinese_raw_data.pdf')
    # plt.show()


def plot_comparision_hist(fuzzy_path, none_fuzzy_path, figure_path):
    # fuzzy_data = np.load(fuzzy_path)
    # none_fuzzy_data = np.load(none_fuzzy_path)

    dfs = pd.read_excel(fuzzy_path)
    fuzzy_data = dfs.values.astype(np.float)[:, 0]
    dfs = pd.read_excel(none_fuzzy_path)
    none_fuzzy_data = dfs.values.astype(np.float)[:, 0]

    # print(len(fuzzy_data))
    # print(len(none_fuzzy_data))

    fuzzy_steps = fuzzy_data
    none_fuzzy_steps = none_fuzzy_data

    # for i in range(20):
    #     fuzzy_steps[i] = len(fuzzy_data[80+i])*0.5
    #
    # for j in range(20):
    #     none_fuzzy_steps[j] = len(none_fuzzy_data[80+j])*0.5

    plt.figure(figsize=(10, 8), dpi=300)
    plt.subplot(1, 1, 1)
    plt.tight_layout(pad=4.8, w_pad=1., h_pad=1.)
    plt.subplots_adjust(left=0.13, bottom=0.15, right=0.98, top=0.98, wspace=0.23, hspace=0.22)
    plt.hist(fuzzy_steps[:20], bins=20, histtype="stepfilled", label='Typical DDPG', color='b')
    plt.hist(none_fuzzy_steps[:20], bins=20, histtype="stepfilled", label='FLDVTSP-DDPG', color='r')

    plt.yticks(fontsize=34)
    plt.xticks(fontsize=34)
    plt.ylabel('Frequency', fontsize=34)
    plt.xlabel('Episode time(s)', fontsize=34)
    plt.grid(axis="y")
    plt.legend(fontsize=30, loc='best')

    plt.savefig(figure_path)
    # plt.savefig('./figure/jpg/dqn_test_episode_time.jpg')
    plt.show()


def plot_3d_point(state_path_1, state_path_2):
    # plot the 3d position
    from mpl_toolkits.mplot3d import Axes3D

    state_data_1 = np.load(state_path_1, allow_pickle=True)
    state_data_new = []
    for i in range(state_data_1.shape[0]):
        state_data_new += state_data_1[i]

    X_1 = np.array(state_data_new)
    print(X_1.shape)

    state_data_2 = np.load(state_path_2, allow_pickle=True)
    state_data_new = []
    for i in range(state_data_2.shape[0]):
        state_data_new += state_data_2[i]

    X_2 = np.array(state_data_new)
    print(X_2.shape)

    # plt.figure(figsize=(8, 8))
    plt.suptitle("Point position in 3D Space", fontsize=14)

    ax_1 = plt.subplot(111, projection='3d')
    ax_1.scatter(X_1[:, 0], X_1[:, 1], X_1[:, 2], c='r')
    ax_1.scatter(X_2[:, 0], X_2[:, 1], X_2[:, 2], c='b')

    # ax_2 = fig.add_subplot(212, projection='3d')
    # ax_2.scatter(X_2[:, 0], X_2[:, 1], X_2[:, 2], c='b')

    plt.show()


def plot_error_line(t, acc_mean_mat, acc_std_mat=None, legend_vec=None,
                    marker_vec=['+', '*', 'o', 'd', 'd', '*', '', '+', 'v', 'x'],
                    line_vec=['--', '-', ':', '-.', '-', '--', '-.', ':', '--', '-.'],
                    marker_size=5,
                    init_idx=0, idx_step=1):
    if acc_std_mat is None:
        acc_std_mat = 0 * acc_mean_mat
    # acc_mean_mat, acc_std_mat: rows: methods, cols: time
    color_vec = plt.cm.Dark2(np.arange(8))
    for r in range(acc_mean_mat.shape[0]):
        plt.plot(t, acc_mean_mat[r, :], linestyle=line_vec[idx_step * r + init_idx],
                 marker=marker_vec[idx_step * r + init_idx], markersize=marker_size, linewidth= 1,
                 color=color_vec[(idx_step * r + init_idx) % 8])
        plt.fill_between(t, acc_mean_mat[r, :] - acc_std_mat[r, :],
                         acc_mean_mat[r, :] + acc_std_mat[r, :], alpha=0.2,
                         color=color_vec[(idx_step * r + init_idx) % 8])
    if legend_vec is not None:
        # plt.legend(legend_vec)
        plt.legend(legend_vec, loc = 'upper left')


def read_csv_vec(file_name):
    data = np.loadtxt(open(file_name, "rb"), delimiter=",", skiprows=1)
    return data[:, -1]


def write_matrix_to_xlsx(data_mat, file_path='data/state_of_art_test_reward.xlsx', env_name = 'Ant',
                         index_label=['DDPG']):
    df = pd.DataFrame(data_mat)
    writer = pd.ExcelWriter(file_path, engine='openpyxl', mode='a')
    df.to_excel(writer, sheet_name=env_name, index_label=tuple(index_label), header=False)
    writer.save()
    writer.close()


def write_to_existing_table(data, file_name, sheet_name='label'):

    xl = pd.read_excel(file_name, sheet_name=None, header=0, index_col=0, dtype='object')
    xl[sheet_name].iloc[1:, :5] = data
    xl[sheet_name].iloc[1:, 5] = np.mean(data, axis=-1)
    xl[sheet_name].iloc[0, 5] = np.max(xl[sheet_name].iloc[1:, 5])
    xl[sheet_name].iloc[1:, 6] = np.std(data, axis=-1)
    xl[sheet_name].iloc[0, 6] = np.min(xl[sheet_name].iloc[1:, 6])
    print(xl[sheet_name])
    with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
        for ws_name, df_sheet in xl.items():
            df_sheet.to_excel(writer, sheet_name=ws_name)


def plot_acc_mat(acc_mat, legend_vec, env_name, plot_std=True, smooth_weight = 0.8, eval_freq = 0.05,
                 t=None, fig=None, fig_name = None, y_label='Test reward',
                 init_idx=0, idx_step=1, marker_size=2):
    # print(legend_vec)
    for r in range(acc_mat.shape[0]):
        for c in range(acc_mat.shape[1]):
            acc_mat[r, c, :] = smooth(acc_mat[r, c, :], weight=smooth_weight)
    mean_acc = np.mean(acc_mat, axis=1)
    std_acc = np.std(acc_mat, axis=1)
    # kernel = np.ones((1, 1), np.float32) / 1
    # mean_acc = cv2.filter2D(mean_acc, -1, kernel)
    # std_acc = cv2.filter2D(std_acc, -1, kernel)
    if t is None:
        time_step = acc_mat.shape[-1] - 1
        t = np.linspace(0, eval_freq * time_step, time_step+1)
    if fig is None:
        fig = plt.figure(figsize=(9, 6))
        # fig = plt.figure()
        plt.tight_layout()
        plt.rcParams.update({'font.size': 7, 'font.serif': 'Times New Roman'})
    if plot_std:
        plot_error_line(t, mean_acc, std_acc, legend_vec=legend_vec,
                        init_idx=init_idx, idx_step = idx_step, marker_size=marker_size)
    else:
        plot_error_line(t, mean_acc, legend_vec=legend_vec,
                        init_idx=init_idx, idx_step = idx_step, marker_size=marker_size)
    plt.xlabel('Time steps ' + r'($1 \times 10^{5}$)' + '\n{}'.format(env_name))
    plt.xlim((min(t), max(t)))
    plt.ylabel(y_label)
    if fig is None:
        plt.show()


def plot_reward_curves(policy_name_vec=None, result_path ='runs/ATD3_walker2d',
                       env_name='RoboschoolWalker2d', fig = None, fig_name='test_reward',
                       smooth_weight=0.8, eval_freq = 0.05):
    if policy_name_vec is None:
        policy_name_vec = ['TD3', 'TD3', 'ATD3', 'ATD3_RNN']
    reward_mat = None
    legend_vec = []
    last_reward = 0.0
    for r in range(len(policy_name_vec)):
        legend_vec.append(policy_name_vec[r])
        file_name_vec = glob.glob('{}/{}_{}*/test_accuracy.npy'.format(
            result_path, policy_name_vec[r], env_name))
        print(file_name_vec)
        for c in range(len(file_name_vec)):
            file_name = file_name_vec[c]
            # dfs = pd.read_excel(file_name)
            dfs = np.load(file_name)
            # acc_vec = dfs.values.astype(np.float)[:, 0]
            acc_vec = dfs
            if reward_mat is None:
                reward_mat = np.zeros((len(policy_name_vec), len(file_name_vec), len(acc_vec)))
            reward_mat[r, c, :] = acc_vec

        if reward_mat is not None:
            max_acc = np.max(reward_mat[r, :, :], axis=-1)
            # print(max_acc)
            print('Max acc for {}, mean: {}, std: {}, d_reward:{}'.format(
                policy_name_vec[r], np.mean(max_acc, axis=-1),
                np.std(max_acc, axis=-1), np.mean(max_acc, axis=-1)-last_reward))
            last_reward = np.mean(max_acc, axis=-1)

    if reward_mat is not None:
        # write_matrix_to_xlsx(np.max(reward_mat, axis = -1), env_name=env_name, index_label=policy_name_vec)
        # write_to_existing_table(np.max(reward_mat, axis = -1), file_name='data/state_of_art_test_reward_1e6.xlsx',
        #                         sheet_name=env_name)
        plot_acc_mat(reward_mat, None, env_name, fig=fig, fig_name=fig_name,
                     smooth_weight=smooth_weight, eval_freq=eval_freq, marker_size=0)
    return legend_vec


def plot_roboschool_test_reward():
    env_name_vec = [
        # 'RoboschoolHopper-v1',
        # 'RoboschoolWalker2d-v1',
        # 'RoboschoolHalfCheetah-v1',
        # 'RoboschoolAnt-v1',
        # 'Hopper-v2',
        # 'Walker2d-v2',
        # 'HalfCheetah-v2',
        # 'Ant-v2',
        'Peg-in-hole-single_assembly'
    ]
    fig = plt.figure(figsize=(6, 5))
    fig.tight_layout()
    plt.rcParams.update({'font.size': 8, 'font.serif': 'Times New Roman'})
    policy_name_vec = ['Average_TD3', 'TD3']

    for i in range(len(env_name_vec)):
        plt.subplot(2, 2, i+1)
        legend_vec = plot_reward_curves(result_path='results/runs/dual_assembly',
                                        env_name=env_name_vec[i],
                                        policy_name_vec=policy_name_vec, fig=fig)
        # plt.yticks([0, 1000, 2000, 3000])
        # plt.xticks([0, 5, 10])

    print(legend_vec)
    legend = fig.legend(policy_name_vec,
                        loc='lower center', ncol=len(policy_name_vec),
                        bbox_to_anchor=(0.50, 0.96), frameon=False)
    fig.tight_layout()
    plt.savefig('Assembly.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()


def smooth(scalars, weight=0.9):
    # Exponential moving average,
    # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return np.asarray(smoothed)


def plot_state_space(state_path_1, state_path_2, figure_path):
    state_data_1 = np.load(state_path_1, allow_pickle=True)
    state_data_new = []
    for i in range(state_data_1.shape[0]):
        for j in range(len(state_data_1[i])):
            state_data_new.append(state_data_1[i][j][1])
            # state_data_new.append(state_data_1[i])
    X_1 = np.array(state_data_new)

    state_data_2 = np.load(state_path_2, allow_pickle=True)
    state_data_new = []
    for i in range(state_data_2.shape[0]):
        for j in range(len(state_data_2[i])):
            state_data_new.append(state_data_2[i][j][1])
    X_2 = np.array(state_data_new)

    n_components = 2

    plt.figure(figsize=(10, 8), dpi=300)
    plt.subplot(1, 1, 1)
    plt.tight_layout(pad=4.8, w_pad=1., h_pad=1.)
    plt.subplots_adjust(left=0.165, bottom=0.15, right=0.98, top=0.98, wspace=0.23, hspace=0.22)
    plt.yticks(fontsize=34)
    plt.xticks(fontsize=34)
    # plt.ylabel('state', fontsize=34)
    # plt.xlabel('state', fontsize=34)

    """ T-SNE """
    tsne_2 = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
    Y_2 = tsne_2.fit_transform(X_2)
    plt.scatter(Y_2[:, 0], Y_2[:, 1], c='b', cmap=plt.cm.Spectral, label='Typical DDPG')

    tsne_1 = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
    Y_1 = tsne_1.fit_transform(X_1)
    plt.scatter(Y_1[:, 0], Y_1[:, 1], c='r', cmap=plt.cm.Spectral, label='FLDVTSP-DDPG')
    # plt.savefig('./results/figure/pdf/none_fuzzy_state_space.pdf')

    # plt.figure(figsize=(10, 8), dpi=300)
    # plt.subplot(1, 1, 1)
    # plt.tight_layout(pad=4.8, w_pad=1., h_pad=1.)
    # plt.yticks(fontsize=34)
    # plt.xticks(fontsize=34)
    # plt.ylabel('state', fontsize=34)
    # plt.xlabel('state', fontsize=34)
    # plt.subplots_adjust(left=0.165, bottom=0.15, right=0.98, top=0.98, wspace=0.23, hspace=0.22)
    plt.legend(fontsize=30, loc='upper right')
    plt.savefig(figure_path)

    # plt.scatter(Y_2[:, 0], Y_2[:, 1], c='b', cmap=plt.cm.Spectral)
    # ax.view_init(4, -72)

    # ax = fig.add_subplot(2, 1, 2)
    # plt.scatter(Y_2[:, 0], Y_2[:, 1], c='b', cmap=plt.cm.Spectral)

    # ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
    # ax.yaxis.set_major_formatter(NullFormatter())
    plt.show()


def plot_state_space_new(state_path_1, state_path_2, figure_path):

    state_data_1 = np.load(state_path_1, allow_pickle=True)
    state_data_new = state_data_1
    # state_data_new = []
    # for i in range(state_data_1.shape[0]):
    #     for j in range(len(state_data_1[i])):
    #         state_data_new.append(state_data_1[i][j])
    #         # state_data_new.append(state_data_1[i])
    X_1 = np.array(state_data_new)

    state_data_2 = np.load(state_path_2, allow_pickle=True)
    state_data_new = state_data_2
    # state_data_new = []
    # for i in range(state_data_2.shape[0]):
    #     for j in range(len(state_data_2[i])):
    #         state_data_new.append(state_data_2[i][j])
    X_2 = np.array(state_data_new)

    n_components = 2

    plt.figure(figsize=(10, 8), dpi=300)
    plt.subplot(1, 1, 1)
    plt.tight_layout(pad=4.8, w_pad=1., h_pad=1.)
    plt.subplots_adjust(left=0.165, bottom=0.15, right=0.98, top=0.98, wspace=0.23, hspace=0.22)
    plt.yticks(fontsize=34)
    plt.xticks(fontsize=34)

    # plt.ylabel('state', fontsize=34)
    # plt.xlabel('state', fontsize=34)
    # plt.legend(fontsize=30, loc='best')

    '''t-SNE'''
    # plt.savefig('./results/figure/pdf/dqn_state_space.pdf')

    # plt.figure(figsize=(10, 8), dpi=300)
    # plt.subplot(1, 1, 1)
    # plt.tight_layout(pad=4.8, w_pad=1., h_pad=1.)
    # plt.yticks(fontsize=34)
    # plt.xticks(fontsize=34)
    # plt.ylabel('state', fontsize=34)
    # plt.xlabel('state', fontsize=34)
    # plt.subplots_adjust(left=0.165, bottom=0.15, right=0.98, top=0.98, wspace=0.23, hspace=0.22)
    tsne_2 = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
    Y_2 = tsne_2.fit_transform(X_2)
    plt.scatter(Y_2[:, 0], Y_2[:, 1], c='b', cmap=plt.cm.Spectral, label='Typical DQN')

    tsne_1 = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
    Y_1 = tsne_1.fit_transform(X_1)
    plt.scatter(Y_1[:, 0], Y_1[:, 1], c='r', cmap=plt.cm.Spectral, label='FLDVTSP-DQN')

    plt.legend(fontsize=30, loc='upper right')
    plt.savefig(figure_path)

    # plt.scatter(Y_2[:, 0], Y_2[:, 1], c='b', cmap=plt.cm.Spectral)
    # ax.view_init(4, -72)

    # ax = fig.add_subplot(2, 1, 2)
    # plt.scatter(Y_2[:, 0], Y_2[:, 1], c='b', cmap=plt.cm.Spectral)

    # ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
    # ax.yaxis.set_major_formatter(NullFormatter())
    plt.show()


def plot_state_action_space(state_path, action_path):
    state_data = np.load(state_path, allow_pickle=True)
    action_data = np.load(action_path, allow_pickle=True)

    state_data_new = []
    for i in range(state_data.shape[0]):
        for j in range(len(state_data[i])):
            state_data_new.append(state_data[i][j])
    X_1 = np.array(state_data_new)

    state_data_new = []
    for i in range(action_data.shape[0]):
        for j in range(len(action_data[i])):
            state_data_new.append(action_data[i][j])
    X_2 = np.array(state_data_new)

    n_components = 2

    plt.figure(figsize=(10, 8), dpi=300)
    plt.subplot(1, 1, 1)
    plt.tight_layout(pad=4.8, w_pad=1., h_pad=1.)
    plt.subplots_adjust(left=0.165, bottom=0.15, right=0.98, top=0.98, wspace=0.23, hspace=0.22)
    plt.yticks(fontsize=34)
    plt.xticks(fontsize=34)
    plt.ylabel('state', fontsize=34)
    plt.xlabel('state', fontsize=34)
    # plt.legend(fontsize=30, loc='best')

    '''t-SNE'''
    tsne_1 = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
    Y_1 = tsne_1.fit_transform(X_1)
    plt.scatter(Y_1[:, 0], Y_1[:, 1], c='r', cmap=plt.cm.Spectral, label='VPB-DQN')
    # plt.savefig('./results/figure/pdf/dqn_state_space.pdf')

    # plt.figure(figsize=(10, 8), dpi=300)
    # plt.subplot(1, 1, 1)
    # plt.tight_layout(pad=4.8, w_pad=1., h_pad=1.)
    # plt.yticks(fontsize=34)
    # plt.xticks(fontsize=34)
    # plt.ylabel('state', fontsize=34)
    # plt.xlabel('state', fontsize=34)
    # plt.subplots_adjust(left=0.165, bottom=0.15, right=0.98, top=0.98, wspace=0.23, hspace=0.22)
    tsne_2 = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
    Y_2 = tsne_2.fit_transform(X_2)

    plt.scatter(Y_2[:, 0], Y_2[:, 1], c='b', cmap=plt.cm.Spectral, label='Typical DQN')
    plt.legend(fontsize=30, loc='upper right')

    plt.savefig('./results/figure/pdf/dqn_state_action_space.pdf')

    # plt.scatter(Y_2[:, 0], Y_2[:, 1], c='b', cmap=plt.cm.Spectral)
    # ax.view_init(4, -72)

    # ax = fig.add_subplot(2, 1, 2)
    # plt.scatter(Y_2[:, 0], Y_2[:, 1], c='b', cmap=plt.cm.Spectral)

    # ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
    # ax.yaxis.set_major_formatter(NullFormatter())
    plt.show()


def plot_3D_path(path_0, path_1, name_0):

    force_data_1 = np.load(path_0, allow_pickle=True)
    force_data_2 = np.load(path_1, allow_pickle=True)

    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax_1 = fig.add_subplot(111, projection='3d')

    v_forces_fx = []
    v_forces_fy = []
    v_forces_fz = []
    for i in range(len(force_data_1)):
        single_force_fx = []
        single_force_fy = []
        single_force_fz = []

        for j in range(len(force_data_1[i])):

            single_force_fx.append(force_data_1[i][j][6:7])
            single_force_fy.append(force_data_1[i][j][7:8])
            single_force_fz.append(force_data_1[i][j][8:9])

        v_forces_fx.append(np.squeeze(np.array(single_force_fx)))
        v_forces_fy.append(np.squeeze(np.array(single_force_fy)))
        v_forces_fz.append(np.squeeze(np.array(single_force_fz)))

    v_forces_mx = []
    v_forces_my = []
    v_forces_mz = []
    for i in range(len(force_data_2)):
        single_force_mx = []
        single_force_my = []
        single_force_mz = []

        for j in range(len(force_data_2[i])):
            single_force_mx.append(force_data_2[i][j][6:7])
            single_force_my.append(force_data_2[i][j][7:8])
            single_force_mz.append(force_data_2[i][j][8:9])

        v_forces_mx.append(np.squeeze(np.array(single_force_mx)))
        v_forces_my.append(np.squeeze(np.array(single_force_my)))
        v_forces_mz.append(np.squeeze(np.array(single_force_mz)))

    ax_1.plot3D(v_forces_fx[0], v_forces_fy[0], v_forces_fz[0], 'blue', label='Typical DQN')
    ax_1.plot3D(v_forces_mx[0], v_forces_my[0], v_forces_mz[0], 'red', label='FLDVTSP-DQN')
    for i in range(1, len(force_data_1)):
        ax_1.plot3D(v_forces_fx[i], v_forces_fy[i], v_forces_fz[i], 'red')
        ax_1.plot3D(v_forces_mx[i], v_forces_my[i], v_forces_mz[i], 'blue')

    plt.yticks([-40.5, -40.0, -39.5, -39.0, -38.5, -38.0], fontsize=FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE)
    plt.legend(fontsize=FONT_SIZE)
    ax_1.set_ylabel('$Y(mm)$', fontsize=FONT_SIZE, labelpad=FONT_SIZE)
    ax_1.set_xlabel('$X(mm)$', fontsize=FONT_SIZE, labelpad=FONT_SIZE)
    ax_1.set_zlabel('$Z(mm)$', fontsize=FONT_SIZE, labelpad=FONT_SIZE)

    plt.subplots_adjust(left=0.13, bottom=0.13, right=0.98, top=0.98)
    plt.tight_layout()

    plt.savefig(name_0)
    plt.show()


def plot_3D_path_new(path_0, path_1, name_0):

    force_data_1 = np.load(path_0, allow_pickle=True)
    force_data_2 = np.load(path_1, allow_pickle=True)

    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax_1 = fig.add_subplot(111, projection='3d')

    v_forces_fx = []
    v_forces_fy = []
    v_forces_fz = []
    for i in range(len(force_data_1)):
        single_force_fx = []
        single_force_fy = []
        single_force_fz = []
        for j in range(len(force_data_1[i])):
            single_force_fx.append(force_data_1[i][j][0][6:7])
            single_force_fy.append(force_data_1[i][j][0][7:8])
            single_force_fz.append(force_data_1[i][j][0][8:9])
        v_forces_fx.append(np.squeeze(np.array(single_force_fx)))
        v_forces_fy.append(np.squeeze(np.array(single_force_fy)))
        v_forces_fz.append(np.squeeze(np.array(single_force_fz)))

    v_forces_mx = []
    v_forces_my = []
    v_forces_mz = []
    for i in range(len(force_data_2)):
        single_force_mx = []
        single_force_my = []
        single_force_mz = []
        for j in range(len(force_data_2[i])):
            single_force_mx.append(force_data_2[i][j][0][6:7])
            single_force_my.append(force_data_2[i][j][0][7:8])
            single_force_mz.append(force_data_2[i][j][0][8:9])
        v_forces_mx.append(np.squeeze(np.array(single_force_mx)))
        v_forces_my.append(np.squeeze(np.array(single_force_my)))
        v_forces_mz.append(np.squeeze(np.array(single_force_mz)))

    ax_1.plot3D(v_forces_fx[i], v_forces_fy[i], v_forces_fz[i], 'blue', label='Typical DDPG')
    ax_1.plot3D(v_forces_mx[i], v_forces_my[i], v_forces_mz[i], 'red', label='FLDVTSP-DDPG')
    for i in range(1, len(force_data_1)):
        ax_1.plot3D(v_forces_fx[i], v_forces_fy[i], v_forces_fz[i], 'blue')
        ax_1.plot3D(v_forces_mx[i], v_forces_my[i], v_forces_mz[i], 'red')

    plt.yticks(fontsize=FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE)
    # plt.zticks(fontsize=FONT_SIZE)
    # plt.xticks(fontsize=FONT_SIZE)
    # plt.title('Path', fontsize=FONT_SIZE)
    plt.legend(fontsize=FONT_SIZE)
    ax_1.set_ylabel('$Y(mm)$', fontsize=FONT_SIZE, labelpad=FONT_SIZE)
    ax_1.set_xlabel('$X(mm)$', fontsize=FONT_SIZE, labelpad=FONT_SIZE)
    ax_1.set_zlabel('$Z(mm)$', fontsize=FONT_SIZE, labelpad=FONT_SIZE)

    plt.subplots_adjust(left=0.13, bottom=0.13, right=0.98, top=0.98)
    plt.tight_layout()
    plt.savefig(name_0)
    plt.show()


def heat_map(force_path_1):
    force_data_1 = np.load(force_path_1, allow_pickle=True)
    MAX_LENGTH = 96

    v_forces_fx = []
    v_forces_fy = []
    v_forces_fz = []
    v_forces_mx = []
    v_forces_my = []
    v_forces_mz = []
    for i in range(len(force_data_1)):
        single_force_fx = []
        single_force_fy = []
        single_force_fz = []
        single_force_mx = []
        single_force_my = []
        single_force_mz = []
        for j in range(MAX_LENGTH):
            if j > (len(force_data_1[i]) - 1):
                single_force_fx.append([0.0])
                single_force_fy.append([0.0])
                single_force_fz.append([0.0])
                single_force_mx.append([0.0])
                single_force_my.append([0.0])
                single_force_mz.append([0.0])
            else:
                single_force_fx.append(force_data_1[i][j][0][0:1])
                single_force_fy.append(force_data_1[i][j][0][1:2])
                single_force_fz.append(force_data_1[i][j][0][2:3])
                single_force_mx.append(force_data_1[i][j][0][3:4])
                single_force_my.append(force_data_1[i][j][0][4:5])
                single_force_mz.append(force_data_1[i][j][0][5:6])
                # single_force_fx.append(force_data_1[i][j][0:1])
                # single_force_fy.append(force_data_1[i][j][1:2])
                # single_force_fz.append(force_data_1[i][j][2:3])
                # single_force_mx.append(force_data_1[i][j][3:4])
                # single_force_my.append(force_data_1[i][j][4:5])
                # single_force_mz.append(force_data_1[i][j][5:6])
        v_forces_fx.append(np.squeeze(np.array(single_force_fx)))
        v_forces_fy.append(np.squeeze(np.array(single_force_fy)))
        v_forces_fz.append(np.squeeze(np.array(single_force_fz)))
        v_forces_mx.append(np.squeeze(np.array(single_force_mx)))
        v_forces_my.append(np.squeeze(np.array(single_force_my)))
        v_forces_mz.append(np.squeeze(np.array(single_force_mz)))


    plt.figure(figsize=(10, 8), dpi=300)

    plt.subplot(231)
    plt.tight_layout(pad=4.8, w_pad=1., h_pad=1.)
    plt.subplots_adjust(left=0.12, bottom=0.20, right=0.98, top=0.98, wspace=0.23, hspace=0.1)
    plt.imshow(v_forces_fx, aspect='auto', cmap=plt.cm.hot_r)
    cbar = plt.colorbar(orientation='horizontal')
    cbar.ax.set_ylabel('N', fontsize=FONT_SIZE)
    plt.yticks([0, 30], fontsize=FONT_SIZE)
    plt.xticks([0, MAX_LENGTH], fontsize=FONT_SIZE)
    plt.title('$F_x$', fontsize=FONT_SIZE)
    plt.ylabel('Test Number', fontsize=FONT_SIZE)
    # plt.xlabel('$F_x$', fontsize=FONT_SIZE)

    plt.subplot(232)
    plt.tight_layout(pad=4.8, w_pad=1., h_pad=1.)
    plt.subplots_adjust(left=0.165, bottom=0.20, right=0.98, top=0.98, wspace=0.1, hspace=0.1)
    plt.imshow(np.array(v_forces_fy), aspect='auto', cmap=plt.cm.hot_r)
    plt.yticks([0, 30], fontsize=FONT_SIZE)
    plt.xticks([0, MAX_LENGTH], fontsize=FONT_SIZE)
    plt.title('$F_y$', fontsize=FONT_SIZE)
    plt.colorbar(orientation='horizontal')
    # plt.ylabel('N', fontsize=FONT_SIZE)
    # plt.xlabel('$F_y$', fontsize=FONT_SIZE)

    plt.subplot(233)
    plt.tight_layout(pad=4.8, w_pad=1., h_pad=1.)
    plt.subplots_adjust(left=0.12, bottom=0.20, right=0.98, top=0.98, wspace=0.1, hspace=0.1)
    plt.imshow(np.array(v_forces_fz), aspect='auto', cmap=plt.cm.hot_r)
    # plt.colorbar(im)
    plt.title('$F_z$', fontsize=FONT_SIZE)
    plt.yticks([0, 30], fontsize=FONT_SIZE)
    plt.xticks([0, MAX_LENGTH], fontsize=FONT_SIZE)
    plt.colorbar(orientation='horizontal')
    # plt.ylabel('N', fontsize=FONT_SIZE)
    # plt.xlabel('$F_z$', fontsize=FONT_SIZE)

    plt.subplot(234)
    plt.tight_layout(pad=4.8, w_pad=1., h_pad=0.6)
    plt.subplots_adjust(left=0.165, bottom=0.20, right=0.98, top=0.98, wspace=0.23, hspace=0.22)
    plt.imshow(np.array(v_forces_mx), aspect='auto', cmap=plt.cm.hot_r)
    cbar = plt.colorbar(orientation='horizontal')
    cbar.ax.set_ylabel('Nm', fontsize=FONT_SIZE)
    plt.title('$M_x$', fontsize=FONT_SIZE)
    plt.yticks([0, 30], fontsize=FONT_SIZE)
    plt.xticks([0, MAX_LENGTH], fontsize=FONT_SIZE)
    plt.ylabel('Test Number', fontsize=FONT_SIZE)
    # plt.xlabel('$M_x$', fontsize=FONT_SIZE)

    plt.subplot(235)
    plt.tight_layout(pad=4.8, w_pad=1., h_pad=1.)
    plt.subplots_adjust(left=0.13, bottom=0.15, right=0.98, top=0.98, wspace=0.23, hspace=0.22)
    plt.imshow(np.array(v_forces_my), aspect='auto', cmap=plt.cm.hot_r)
    plt.title('$M_y$', fontsize=FONT_SIZE)
    plt.yticks([0, 30], fontsize=FONT_SIZE)
    plt.xticks([0, MAX_LENGTH], fontsize=FONT_SIZE)
    plt.colorbar(orientation='horizontal')
    # plt.ylabel('N', fontsize=FONT_SIZE)
    # plt.xlabel('$M_y$', fontsize=FONT_SIZE)

    plt.subplot(236)
    plt.tight_layout(pad=4.8, w_pad=1., h_pad=1.)
    plt.subplots_adjust(left=0.13, bottom=0.13, right=0.98, top=0.98, wspace=0.23, hspace=0.22)
    plt.imshow(np.array(v_forces_mz), aspect='auto', cmap=plt.cm.hot_r)
    plt.colorbar(orientation='horizontal')
    plt.title('$M_z$', fontsize=FONT_SIZE)
    plt.yticks([0, 30], fontsize=FONT_SIZE)
    plt.xticks([0, MAX_LENGTH], fontsize=FONT_SIZE)
    # plt.ylabel('N', fontsize=FONT_SIZE)
    # plt.xlabel('$M_z$', fontsize=FONT_SIZE)

    plt.subplots_adjust(left=0.13, bottom=0.13, right=0.98, top=0.98)
    plt.tight_layout()

    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # cb = fig.colorbar(ax_6, ax=[ax_4, ax_5, ax_6], cax=cbar_ax, orientation='horizontal')

    plt.savefig('./figure/pdf/ddpg_force_none_fuzzy.pdf')
    plt.show()


# plot_3d_point('./results/episode_state_fuzzy_noise_final.npy', './results/episode_state_none_fuzzy_noise_final.npy')

# plot_state_space('./transfer/dual_assembly_VPB_new/TD3_dual-peg-in-hole_seed_0/evaluation_states.npy',
#                  './transfer/dual_assembly/TD3_dual-peg-in-hole_seed_0/evaluation_states.npy',
#                  './figure/pdf/ddpg_state_action_space.pdf')

# plot_state_action_space('./results/transfer/dual_assembly_VPB_new/TD3_dual-peg-in-hole_seed_0/evaluation_states.npy',
                 # './results/transfer/dual_assembly/TD3_dual-peg-in-hole_seed_0/evaluation_states.npy')

# plot_state_space_new('./data_fuzzy_test_new/episode_actions.npy',
#                      './data_test/episode_actions.npy',
#                      './figure/pdf/dqn_state_action_space.pdf')

# plot_state_space_new('./data_fuzzy_test_new/episode_state.npy',
#                      './data_test/episode_state.npy',
#                      './figure/pdf/dqn_state_space.pdf')

# plot_state_action_space(state_path, action_path)

# data = np.load('./results/transfer/dual_assembly/TD3_dual-peg-in-hole_seed_0/buffer_data.npy', allow_pickle=True)
# print(data.shape)

# plot_compare('./transfer/dual_assembly/TD3_dual-peg-in-hole_seed_0/evaluation_reward.xls',
#              './transfer/dual_assembly_VPB_new/TD3_dual-peg-in-hole_seed_0/evaluation_reward.xls',
#              './figure/pdf/ddpg_test_episode_reward.pdf')

# plot_compare('./data_test/episode_rewards.npy',
#              './data_fuzzy_test_new/episode_rewards.npy',
#              './figure/pdf/dqn_test_episode_reward.pdf')

# plot_comparision_hist('./transfer/dual_assembly/TD3_dual-peg-in-hole_seed_0/evaluation_time.xls',
#                       './transfer/dual_assembly_VPB_new/TD3_dual-peg-in-hole_seed_0/evaluation_time.xls',
#                        './figure/pdf/ddpg_test_episode_time.pdf')

# plot_comparision_hist('./data_test/episode_time.npy',
#                       './data_fuzzy_test_new/episode_time.npy',
#                       './figure/pdf/dqn_test_episode_time.pdf')

# heat_map('./data_test/episode_state.npy')
# heat_map('./data_fuzzy_test_new/episode_state.npy')
# heat_map('./transfer/dual_assembly_VPB_new/TD3_dual-peg-in-hole_seed_0/evaluation_states.npy')
# heat_map('./transfer/dual_assembly/TD3_dual-peg-in-hole_seed_0/evaluation_states.npy')


plot_3D_path('./data_test/episode_state.npy',
             './data_fuzzy_test_new/episode_state.npy',
             './figure/pdf/dqn_path_fuzzy.pdf')

plot_3D_path_new('transfer/dual_assembly/TD3_dual-peg-in-hole_seed_0/evaluation_states_new.npy',
             'transfer/dual_assembly_VPB_new/TD3_dual-peg-in-hole_seed_0/evaluation_states_new_2.npy',
             './figure/pdf/ddpg_path_fuzzy.pdf')
