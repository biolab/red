"""
Copyright (C) 2013  Marinka Zitnik <marinka.zitnik@fri.uni-lj.si>

This file is part of Red.

Red is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Red is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Red.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import scipy.io as sio


def load_jonikas_data(path):
    data = sio.loadmat(path)
    genes = [data["qnames_out"][0, i][0] for i
             in xrange(data["qnames_out"].shape[1])]
    data["DM_Hill_merged"][np.isnan(data["DM_Hill_merged"])] = 0
    data["qindex_out"][np.isnan(data["qindex_out"])] = 0
    G = data["DM_array_merged"]
    S = np.reshape(data["qindex_out"][:, 1], (G.shape[0], 1))
    H = data["DM_Hill_merged"]
    return G, S, H, genes


def load_battle_KEGG_data(path_ordered, path_unordered):
    kegg = []
    for line in open(path_ordered):
        g1, g2 = line.strip().split()
        kegg.append((1, g1, g2))
    for line in open(path_unordered):
        g1, g2 = line.strip().split()
        kegg.append((0, g1, g2))
    return kegg


def load_n_linked_glycans_data(path_ordered, path_unordered):
    glycans = []
    for line in open(path_ordered):
        g1, g2 = line.strip().split()
        glycans.append((1, g1, g2))
    for line in open(path_unordered):
        g1, g2 = line.strip().split()
        glycans.append((0, g1, g2))
    return glycans


def get_alleviating_interactions(path, g_mean=180):
    G, S, H, genes = load_jonikas_data(path)
    pos_alleviating, neg_alleviating = [], []
    for i in xrange(G.shape[0]):
        for j in xrange(i+1, G.shape[1]):
            if np.isnan(G[i, j]) or S[i] == 0 or S[j] == 0:
                continue
            T = abs(G[i, j]-max(S[i], S[j]))
            a1 = np.sum(np.logical_not(np.isnan(G[i, :])))
            a2 = np.sum(np.logical_not(np.isnan(G[j, :])))
            if np.sqrt(a1 * a2) < g_mean:
                continue
            if G[i, j] < -T:
                pos_alleviating.append((1, genes[i], genes[j]))
            else:
                neg_alleviating.append((0, genes[i], genes[j]))
    np.random.shuffle(pos_alleviating)
    np.random.shuffle(neg_alleviating)
    n = int(0.05*len(pos_alleviating))
    return pos_alleviating[:n] + neg_alleviating[:n]