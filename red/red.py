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

from collections import defaultdict
from itertools import combinations
import numpy as np


class Red(object):
    def __init__(self, G, S, H, genes):
        self.genes = genes
        self._g2i = {g: i for i, g in enumerate(self.genes)}
        self.S = S
        self.H = H
        self.G = G
        self._n = self.G.shape[0]
        self._ga = np.ones((self._n, self._n))
        self._gb = np.ones((self._n, self._n))
        self._gc = np.ones((self._n, self._n))

        self._linear_u_downstream = np.zeros((self._n, self._n))
        self._linear_v_downstream = np.zeros((self._n, self._n))
        self._parallel = np.zeros((self._n, self._n))
        self._partially_interdependent = np.zeros((self._n, self._n))

    def u_downstream_proba(self, u, v):
        return self._linear_u_downstream[self._g2i[u], self._g2i[v]]

    def v_downstream_proba(self, u, v):
        return self._linear_v_downstream[self._g2i[u], self._g2i[v]]

    def parallel_proba(self, u, v):
        return self._parallel[self._g2i[u], self._g2i[v]]

    def partially_interdependent_proba(self, u, v):
        return self._partially_interdependent[self._g2i[u], self._g2i[v]]

    def epistatic_to(self, v, u):
        s = self.v_downstream_proba(u, v)+self.u_downstream_proba(u, v)
        p = self.v_downstream_proba(u, v)/s
        return p

    def alleviating(self, u, v):
        p = self.v_downstream_proba(u, v) + self.u_downstream_proba(u, v)
        return p

    def _initialize(self, c):
        self.U = np.random.normal(0, 1, (c, self._n))
        self.V = np.random.normal(0, 1, (c, self._n))

    def _predict_linear_u_downstream(self):
        """u and v in a linear pathway, u downstream"""
        S = np.tile(self.S, (1, self._n))
        d = np.abs(self._g(np.dot(self.U.T, self.V)) - S)
        d = (d + d.T) / 2.
        self._linear_u_downstream = 2. / (1 + np.exp(d))

    def _predict_linear_v_downstream(self):
        """u and v in a linear pathway, v downstream"""
        S = np.tile(self.S, (1, self._n))
        d = np.abs(self._g(np.dot(self.U.T, self.V)) - S.T)
        self._linear_v_downstream = 2. / (1 + np.exp(d))

    def _predict_parallelism(self):
        """u and v affect the reporter separately"""
        d = np.abs(self._g(np.dot(self.U.T, self.V)) - self.H)
        d = (d + d.T) / 2.
        self._parallel = 2. / (1 + np.exp(d))

    def _predict_partially_interdependent(self):
        """u and v are partially interdependent"""
        S = np.tile(self.S, (1, self._n))
        sign = 0.5 * (self.H + np.maximum(S, S.T))
        d = np.abs(self._g(np.dot(self.U.T, self.V)) - sign)
        d = (d + d.T) / 2.
        self._partially_interdependent = 2. / (1 + np.exp(d))

    def _compute_order_scores(self):
        self._predict_linear_u_downstream()
        self._predict_linear_v_downstream()
        self._predict_parallelism()
        self._predict_partially_interdependent()
        s = sum([self._linear_u_downstream, self._linear_v_downstream,
                 self._parallel, self._partially_interdependent])
        # transform relationship scores into a distribution
        self._linear_u_downstream /= s
        self._linear_v_downstream /= s
        self._parallel /= s
        self._partially_interdependent /= s

    def _g(self, x):
        return self._gc / (
            1. + np.multiply(self._ga, np.exp(np.multiply(-self._gb, x))))

    def _g_prime(self, x):
        tmp = np.exp(np.multiply(self._gb, x))
        tmp1 = np.multiply(self._ga, np.multiply(self._gb, self._gc))
        return np.multiply(tmp1, tmp) / (self._ga + tmp) ** 2

    def _U_prime(self, G, U, V, lambda_u):
        G_hat = np.dot(U.transpose(), V)
        G_hat_g_prime = self._g_prime(G_hat)
        G_hat_g = self._g(G_hat)
        G_p = G.copy()
        G_p[np.isnan(G)] = G_hat_g[np.isnan(G)]
        U_prime = np.dot(V, np.multiply(G_hat_g_prime, G_hat_g - G_p).T)
        return U_prime + lambda_u * U

    def _V_prime(self, G, U, V, lambda_v):
        G_hat = np.dot(U.transpose(), V)
        G_hat_g_prime = self._g_prime(G_hat)
        G_hat_g = self._g(G_hat)
        G_p = G.copy()
        G_p[np.isnan(G)] = G_hat_g[np.isnan(G)]
        V_prime = np.dot(U, np.multiply(G_hat_g_prime, G_hat_g - G_p))
        return V_prime + lambda_v * V

    def _a_prime(self, G, U, V, U_old, V_old):
        G_hat = np.dot(U.transpose(), V)
        G_hat_old = np.dot(U_old.transpose(), V_old)
        G_hat_g = self._g(G_hat)
        G_hat_g_old = self._g(G_hat_old)
        G_p = G.copy()
        G_p[np.isnan(G)] = G_hat_g_old[np.isnan(G)]
        tmp1 = np.exp(self._gb * G_hat)
        tmp = -self._gc * tmp1 / (tmp1 + self._ga) ** 2
        a_prime = np.multiply(tmp, G_hat_g - G_p)
        return a_prime

    def _b_prime(self, G, U, V, U_old, V_old):
        G_hat = np.dot(U.transpose(), V)
        G_hat_old = np.dot(U_old.transpose(), V_old)
        G_hat_g = self._g(G_hat)
        G_hat_g_old = self._g(G_hat_old)
        G_p = G.copy()
        G_p[np.isnan(G)] = G_hat_g_old[np.isnan(G)]
        tmp1 = np.exp(np.multiply(self._gb, G_hat))
        tmp = self._ga * self._gc * G_hat * tmp1 / (tmp1 + self._ga) ** 2
        b_prime = np.multiply(tmp, G_hat_g - G_p)
        return b_prime

    def _c_prime(self, G, U, V, U_old, V_old):
        G_hat = np.dot(U.transpose(), V)
        G_hat_old = np.dot(U_old.transpose(), V_old)
        G_hat_g = self._g(G_hat)
        G_hat_g_old = self._g(G_hat_old)
        G_p = G.copy()
        G_p[np.isnan(G)] = G_hat_g_old[np.isnan(G)]
        tmp = 1. / (
            1 + np.multiply(self._ga, np.exp(np.multiply(-self._gb, G_hat))))
        c_prime = np.multiply(tmp, G_hat_g - G_p)
        return c_prime

    def order(self, rank=100, lambda_u=1e-4, lambda_v=1e-4,
              alpha=0.1, beta=0.1, max_iter=200,
              verbose=False, callback=None):
        self._initialize(rank)

        fro_nrmse = []
        err = {-2: (1e10, self.U.copy(), self.V.copy()),
               -1: (1e8, self.U.copy(), self.V.copy())}

        if callback:
            self._compute_order_scores()
            callback(self)

        for iteration in xrange(max_iter):
            if not err[-1][0] < err[-2][0]:
                break
            err[-2] = err[-1][0], err[-1][1].copy(), err[-1][2].copy()

            if verbose:
                print "Iteration: %d" % iteration
            U_prime = self._U_prime(self.G, self.U, self.V, lambda_u)
            V_prime = self._V_prime(self.G, self.U, self.V, lambda_v)

            U_old = self.U.copy()
            V_old = self.V.copy()
            self.U = self.U - alpha * U_prime
            self.V = self.V - alpha * V_prime

            a_prime = self._a_prime(self.G, self.U, self.V, U_old, V_old)
            b_prime = self._b_prime(self.G, self.U, self.V, U_old, V_old)
            c_prime = self._c_prime(self.G, self.U, self.V, U_old, V_old)
            self._ga = self._ga - beta * a_prime
            self._gb = self._gb - beta * b_prime
            self._gc = self._gc - beta * c_prime
            for i in xrange(self._n):
                self._ga[i, :] = np.mean(self._ga[i, :])
                self._gb[i, :] = np.mean(self._gb[i, :])
                self._gc[i, :] = np.mean(self._gc[i, :])
            G_hat = self._g(np.dot(self.U.transpose(), self.V))

            error = np.sqrt(
                np.nansum(np.multiply(self.G - G_hat, self.G - G_hat)))
            if verbose:
                print "Fro. Error: %5.4f" % error
            Gma = np.ma.masked_array(self.G, np.isnan(self.G))
            nrmse = np.sqrt(
                np.mean(np.multiply(Gma - G_hat, Gma - G_hat)) / np.var(Gma))
            if verbose:
                print "NRMSE: %5.4f" % nrmse
            fro_nrmse.append((error, nrmse))

            err[-1] = error, self.U.copy(), self.V.copy()
            if callback:
                self._compute_order_scores()
                callback(self)
        self.Psi = [self._ga, self._gb, self._gc]
        if err[-1][0] < err[-2][0]:
            self.U = err[-1][1]
            self.V = err[-1][2]
        else:
            self.U = err[-2][1]
            self.V = err[-2][2]
        self._compute_order_scores()
        self.fro_error = fro_nrmse[-1][0]
        self.nrmse = fro_nrmse[-1][1]

    def construct_network(self, gene_set):
        """Retain non-violating and non-redundant edges of complete network."""
        N = np.zeros((self._n, self._n))
        for g1, g2 in combinations(gene_set, 2):
            if g1 not in self._g2i or g2 not in self._g2i: continue
            p0 = self._linear_u_downstream[self._g2i[g1], self._g2i[g2]]
            p1 = self._linear_v_downstream[self._g2i[g1], self._g2i[g2]]
            if p0 > p1:
                N[self._g2i[g2], self._g2i[g1]] = p0
            else:
                N[self._g2i[g1], self._g2i[g2]] = p1

        # two genes are considered adjacent if there is no evidence
        # for intervening genes
        genes = set([self._g2i[g] for g in gene_set])
        O = N.copy()
        n = len(genes)
        order = defaultdict(list)
        unordered = genes
        s_ind, e_ind = 0, n - 1
        for _ in xrange(n):
            for _ in xrange(n):

                # select vertex with no incoming edges, recurse on the graph
                # minus that vertex and prepend that vertex to the order
                while True:
                    rem = []
                    for i in unordered:
                        if O[:, i].sum() == 0:
                            order[s_ind].append(i)
                            rem.append(i)
                    if not rem:
                        break
                    else:
                        unordered.difference_update(rem)
                        s_ind += 1

                # look for vertices with no outgoing vertices and append them
                while True:
                    rem = []
                    for i in unordered:
                        if O[i, :].sum() == 0:
                            order[e_ind].append(i)
                            rem.append(i)
                    if not rem:
                        break
                    else:
                        unordered.difference_update(rem)
                        e_ind -= 1

            # if all vertices have incoming and outgoing arcs then select the
            # vertex with the highest differential between incoming and outgoing
            # degrees
            if unordered:
                l = [(O[:, i].sum() - O[i, :].sum(), i) for i in unordered]
                l = sorted(l, reverse=False)
                diff = np.diff(zip(*l)[0])
                if not len(diff):
                    i = l[0][1]
                    order[s_ind].append(i)
                    unordered.remove(i)
                    O[:, i] = O[i, :] = 0
                    s_ind += 1
                elif np.argmax(diff) > len(unordered) / 2. - 1:
                    for j in xrange(len(unordered) - 1, np.argmax(diff), -1):
                        i = l[-j][1]
                        order[e_ind].append(i)
                        unordered.remove(i)
                        O[:, i] = O[i, :] = 0
                    e_ind -= 1
                else:
                    for j in xrange(np.argmax(diff) + 1):
                        i = l[j][1]
                        order[s_ind].append(i)
                        unordered.remove(i)
                        O[:, i] = O[i, :] = 0
                    s_ind += 1

        ord2ind = {ordr: i for i, ordr in enumerate(sorted(set(order.keys())))}
        i2o = {g: ord2ind[ordr] for ordr, genes in order.iteritems() for g in
               genes}
        i2g = {i: g for g, i in self._g2i.iteritems()}

        # retain edges that go from upper to lower levels or are within levels
        # or represent direct connections
        for i in xrange(N.shape[0]):
            for j in xrange(N.shape[1]):
                t = N[i, :]
                if (N[i, j] and (i2o[i] + 1 == i2o[j] or i2o[i] == i2o[j])) or \
                        (N[i, j] and N[i, j] == np.min(t[t != 0])):
                    continue
                else:
                    N[i, j] = 0

        # remove triads
        for i in xrange(N.shape[0]):
            for j in xrange(N.shape[1]):
                if self._triad(N, i, j):
                    N[i, j] = 0
                    continue
                if N[i, j] != 0:
                    print '%s -> %s' % (i2g[i], i2g[j])

    def _triad(self, N, i, j):
        if N[i, j] == 0:
            return False
        for k in xrange(N.shape[1]):
            if N[k, j] != 0 and N[i, k] != 0:
                return True
        return False

    def print_relationships(self, gene_set):
        for g1, g2 in combinations(gene_set, 2):
            if g1 not in self._g2i or g2 not in self._g2i: continue
            p0 = self._linear_u_downstream[self._g2i[g1], self._g2i[g2]]
            p1 = self._linear_v_downstream[self._g2i[g1], self._g2i[g2]]
            p2 = self._parallel[self._g2i[g1], self._g2i[g2]]
            p3 = self._partially_interdependent[self._g2i[g1], self._g2i[g2]]
            rel = np.argmax([p0, p1, p2, p3])
            sym = ['<--', '-->', '||', '/_\\']
            print "%s %s %s" % (g1, sym[rel], g2)