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

import os
import numpy as np
from sklearn.metrics import roc_auc_score
from data import loader
from red import Red


def run_red(G, S, H, genes):
    c = 100
    alpha = 0.1
    lambda_u = 1e-4
    lambda_v = lambda_u
    beta = alpha
    gene_red = Red(G, S, H, genes)
    gene_red.order(rank=c, lambda_u=lambda_u, lambda_v=lambda_v,
                               alpha=alpha, beta=beta,
                               verbose=False, callback=None)
    return gene_red


def predict_alleviating(G, S, H, genes, alleviating_set):
    g2i = {g: i for i, g in enumerate(genes)}
    for cls, g1, g2 in alleviating_set:
        G[g2i[g1], g2i[g2]] = G[g2i[g2], g2i[g1]] = np.nan
    gene_red = run_red(G, S, H, genes)
    pred = [gene_red.alleviating(u, v) for _, u, v  in alleviating_set]
    auc = roc_auc_score(zip(*alleviating_set)[0], pred)
    print "Alleviating AUC: %5.4f" % auc
    return gene_red, auc


def predict_kegg(G, S, H, genes, kegg):
    gene_red = run_red(G, S, H, genes)
    pred = [gene_red.epistatic_to(v, u) for _, u, v in kegg]
    auc = roc_auc_score(zip(*kegg)[0], pred)
    print "KEGG AUC: %5.4f" % auc
    return gene_red, auc


def predict_glycans(G, S, H, genes, glycans):
    gene_red = run_red(G, S, H, genes)
    pred = [gene_red.epistatic_to(v, u) for _, u, v in glycans]
    auc = roc_auc_score(zip(*glycans)[0], pred)
    print "N-linked glycosylation AUC: %5.4f" % auc
    return gene_red, auc


path = os.path.abspath("data/080930a_DM_data.mat")
G, S, H, genes = loader.load_jonikas_data(path)
np.random.seed(42)

# 1
path_ord_kegg = os.path.abspath("data/KEGG_ordered.txt")
path_unord_kegg = os.path.abspath("data/KEGG_nonordered.txt")
kegg = loader.load_battle_KEGG_data(path_ord_kegg, path_unord_kegg)
predict_kegg(G, S, H, genes, kegg)

# 2
path_neg_glycans = os.path.abspath("data/N-linked-glycans_negative.txt")
path_pos_glycans = os.path.abspath("data/N-linked-glycans_positive.txt")
glycans = loader.load_n_linked_glycans_data(path_pos_glycans, path_neg_glycans)
predict_glycans(G, S, H, genes, glycans)

# 3
alleviating_set = loader.get_alleviating_interactions(path)
predict_alleviating(G, S, H, genes, alleviating_set)

# 4
gene_red = run_red(G, S, H, genes)
glycan_genes = {'CWH41', 'DIE2', 'ALG8', 'ALG6', 'ALG5', 'ALG12',
                         'ALG9', 'ALG3', 'OST3', 'OST5'}
erad_genes = {'MNL1', 'YOS9', 'YLR104W', 'DER1', 'USA1', 'HRD3', 'HRD1',
                'UBC7', 'CUE1'}
tailanch_genes = {'SGT2', 'MDY2', 'YOR164C',
                                           'GET3', 'GET2', 'GET1'}
print '\n**N-linked glycosylation pathway'
gene_red.print_relationships(glycan_genes)
gene_red.construct_network(glycan_genes)
print '\n**ERAD pathway'
gene_red.print_relationships(erad_genes)
gene_red.construct_network(erad_genes)
print '\n**Tail-anchored protein insertion pathway'
gene_red.print_relationships(tailanch_genes)
gene_red.construct_network(tailanch_genes)