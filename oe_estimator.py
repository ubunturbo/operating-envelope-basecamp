#!/usr/bin/env python3
"""Frozen estimator for the operating-envelope experiment.

Every definition below is copied VERBATIM from the frozen predecessor script
`run_b500_newton_recovered.py` (rescue package, SHA-256
b11ab78f3414ebef974c6adc51fe4a2f2eea3ad7704c1df5df9972e1cb327b1d) so that the
v0.3 confirmatory run tests the SAME estimator that was characterised in the
exploratory phase. Do not edit these definitions; any change requires a new
protocol version and a new external timestamp.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_k, '1')
import itertools, json, math
from collections import defaultdict
import numpy as np
from scipy.optimize import least_squares
from sklearn.model_selection import StratifiedKFold

ETA = .10  # parity label noise (frozen)


# ---------------------------------------------------------------- generator ---
def state_space(order):
    return np.array(list(itertools.product([0, 1], repeat=order + 2)), dtype=np.int8)


def joint_distribution(order, accuracy, alpha):
    states = state_space(order); probs = np.empty(len(states), float)
    for idx, state in enumerate(states):
        sources = state[:order]; y = int(state[-1]); pedestal = 0.
        for u in (0, 1):
            p = .25
            for x in sources: p *= accuracy if int(x) == u else 1. - accuracy
            p *= accuracy if y == u else 1. - accuracy; pedestal += p
        parity_target = 0
        for x in sources: parity_target ^= int(x)
        parity = (.5 ** order) * .5 * ((1. - ETA) if y == parity_target else ETA)
        probs[idx] = (1. - alpha) * pedestal + alpha * parity
    probs /= probs.sum(); return states, probs


def entropy(v):
    v = v[v > 0]; return float(-np.sum(v * np.log(v)))


def grouped(states, probs, cols):
    out = defaultdict(float)
    for s, p in zip(states, probs): out[tuple(int(s[c]) for c in cols)] += float(p)
    return out


def cmi(states, probs, xcols, ycols, ccols=()):
    joint = defaultdict(float); pc = defaultdict(float); pxc = defaultdict(float); pyc = defaultdict(float)
    for s, p in zip(states, probs):
        x = tuple(int(s[c]) for c in xcols); y = tuple(int(s[c]) for c in ycols); c = tuple(int(s[c]) for c in ccols); p = float(p)
        joint[(x, y, c)] += p; pc[c] += p; pxc[(x, c)] += p; pyc[(y, c)] += p
    return float(sum(p * math.log(p * pc[c] / (pxc[(x, c)] * pyc[(y, c)])) for (x, y, c), p in joint.items() if p > 0))


def metrics(order, accuracy, alpha):
    states, probs = joint_distribution(order, accuracy, alpha); src = tuple(range(order)); z = order - 1; cs = tuple(range(order - 1)); r = order; y = order + 1
    T = cmi(states, probs, (z,), (y,), (r,)); D = cmi(states, probs, (z,), (y,), cs + (r,)); M = T - D
    pis = [cmi(states, probs, (j,), (y,)) for j in src]; rows = []; mis = []; cmis = []; cois = []
    for i, j in itertools.combinations(src, 2):
        mi = cmi(states, probs, (i,), (j,)); cy = cmi(states, probs, (i,), (j,), (y,)); co = mi - cy
        rows.append({'source_i': i, 'source_j': j, 'mi': mi, 'cmi_given_y': cy, 'coinfo': co}); mis.append(mi); cmis.append(cy); cois.append(co)
    mh = [entropy(np.array(list(grouped(states, probs, (j,)).values()))) for j in src]
    tc = sum(mh) - entropy(np.array(list(grouped(states, probs, src).values())))
    return dict(true_M=M, T_Z=T, D_Z=D, pi_values=pis, pi_max=max(pis), pi_mean=float(np.mean(pis)), Pi_total=float(np.sum(pis)), gamma_max=max(mis) if mis else 0., gamma_mean=float(np.mean(mis)) if mis else 0., gamma_y_max=max(cmis) if cmis else 0., coi_mean=float(np.mean(cois)) if cois else 0., coi_min=min(cois) if cois else 0., coi_max=max(cois) if cois else 0., total_correlation=float(tc), pair_diagnostics_json=json.dumps(rows, sort_keys=True))


def solve(order, target_pi, target_depth):
    best = None
    for start in [(a, w) for a in (.55, .65, .75, .85, .95) for w in (.02, .10, .25, .45, .65, .85)]:
        def residual(x):
            m = metrics(order, float(x[0]), float(x[1])); return np.array([(m['true_M'] + target_depth) / .20, (m['pi_max'] - target_pi) / .03])
        fit = least_squares(residual, np.array(start, float), bounds=(np.array([.500001, 0.]), np.array([.999, 1.])), xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=5000)
        cand = (float(np.linalg.norm(residual(fit.x))), float(fit.x[0]), float(fit.x[1]))
        if best is None or cand < best: best = cand
    err, a, w = best
    if err > 1e-8: raise RuntimeError((order, target_pi, target_depth, err))
    return a, w, metrics(order, a, w)


# ---------------------------------------------------------------- estimators --
class Sat:
    def __init__(self, smoothing=.5): self.smoothing = smoothing
    def fit(self, x, y):
        counts = defaultdict(lambda: [0, 0])
        for row, o in zip(x, y): counts[tuple(int(v) for v in row)][int(o)] += 1
        self.mapping = {k: (n1 + self.smoothing) / (n0 + n1 + 2 * self.smoothing) for k, (n0, n1) in counts.items()}; self.global_p = (float(y.sum()) + self.smoothing) / (len(y) + 2 * self.smoothing); return self
    def predict_proba(self, x):
        p = np.array([self.mapping.get(tuple(int(v) for v in row), self.global_p) for row in x], float); return np.column_stack([1 - p, p])


class Node:
    __slots__ = ('feature', 'left', 'right', 'value')
    def __init__(self, feature=None, left=None, right=None, value=0.): self.feature = feature; self.left = left; self.right = right; self.value = float(value)


def fit_tree(x, g, h, w, max_depth=3, min_child=5, l2=.5):
    def build(idx, depth, used):
        G = float(g[idx].sum()); H = float(h[idx].sum()); leaf = Node(value=G / (H + l2))
        if depth >= max_depth: return leaf
        base = G * G / (H + l2); best = None
        for f in range(x.shape[1]):
            if f in used: continue
            li = idx[x[idx, f] == 0]; ri = idx[x[idx, f] == 1]
            if len(li) == 0 or len(ri) == 0 or w[li].sum() < min_child or w[ri].sum() < min_child: continue
            gl = float(g[li].sum()); hl = float(h[li].sum()); gr = float(g[ri].sum()); hr = float(h[ri].sum())
            gain = .5 * (gl * gl / (hl + l2) + gr * gr / (hr + l2) - base)
            if best is None or gain > best[0]: best = (gain, f, li, ri)
        if best is None or best[0] <= 1e-12: return leaf
        _, f, li, ri = best; u = used | {f}; return Node(f, build(li, depth + 1, u), build(ri, depth + 1, u))
    return build(np.arange(len(x), dtype=int), 0, set())


def apply_tree(tree, x):
    out = np.empty(len(x), float)
    for i, row in enumerate(x):
        n = tree
        while n.feature is not None: n = n.left if row[n.feature] == 0 else n.right
        out[i] = n.value
    return out


class GBM:
    def fit(self, x, y):
        x = np.asarray(x, dtype=np.int8); y = np.asarray(y, dtype=int); ux, inv = np.unique(x, axis=0, return_inverse=True); w = np.bincount(inv, minlength=len(ux)).astype(float); pos = np.bincount(inv, weights=y, minlength=len(ux)).astype(float)
        p0 = (float(y.sum()) + .5) / (len(y) + 1.); self.base = math.log(p0 / (1 - p0)); score = np.full(len(ux), self.base); self.trees = []
        for _ in range(30):
            p = 1 / (1 + np.exp(-np.clip(score, -30, 30))); g = pos - w * p; h = w * p * (1 - p); t = fit_tree(ux, g, h, w); score += .05 * apply_tree(t, ux); self.trees.append(t)
        return self
    def predict_proba(self, x):
        x = np.asarray(x, dtype=np.int8); score = np.full(len(x), self.base)
        for t in self.trees: score += .05 * apply_tree(t, x)
        p = 1 / (1 + np.exp(-np.clip(score, -30, 30))); return np.column_stack([1 - p, p])


def predsets(order):
    z = order - 1; cs = tuple(range(order - 1)); r = order; return [(r,), (r, z), cs + (r,), cs + (r, z)]


FOLDS = 5  # frozen
