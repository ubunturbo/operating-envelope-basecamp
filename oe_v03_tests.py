#!/usr/bin/env python3
"""Offset-free unit tests for operating-envelope v0.3-rc5.

`run_all(verbose, include_subprocess)` -> bool. Guards invoke it with
include_subprocess=False (so the subprocess meta-test cannot recurse). Standalone
execution runs everything. None of these generate a validation offset.
"""
from __future__ import annotations
import json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

import oe_estimator as est
import oe_v03_run as run


def _t_gates():
    assert [set(p) for p in est.predsets(2)] == [{2}, {1, 2}, {0, 2}, {0, 1, 2}]
    assert abs(run.T_ADJ - 2.402079) < 1e-4
    assert abs(run.DELTA_BIAS - 0.01) < 1e-12 and abs(run.DELTA_REP - 0.05) < 1e-12
    rng = np.random.default_rng(0)
    assert run.tier0a_equivalence(0.004 + 0.003 * rng.standard_normal(500))['passed']
    assert not run.tier0a_equivalence(0.020 + 0.003 * rng.standard_normal(500))['passed']
    prev = None
    for n in (100, 1000, 10000):
        r = run.tier0a_equivalence(0.006 + 0.02 * np.random.default_rng(1).standard_normal(n))
        w = r['ci_high'] - r['ci_low']; assert prev is None or w < prev; prev = w
    return True


def _t_coverage():
    def cov(k): return run.tier0b_coverage(np.where(np.arange(500) < k, 0.01, 0.09))
    assert not cov(485)['passed'] and cov(486)['passed']
    assert abs(cov(485)['cp_lower'] - 0.949253) < 1e-5 and abs(cov(486)['cp_lower'] - 0.951756) < 1e-5
    return True


def _t_four_score_identity():
    class MockR:
        def __init__(self, p): self.p = p
        def fit(self, x, y): return self
        def predict_proba(self, x): pp = np.full(len(x), self.p); return np.column_stack([1 - pp, pp])
    orig = est.GBM; est.GBM = lambda: MockR(0.6)
    try:
        data = np.array([[0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 1, 1], [1, 1, 1, 0]] * 10, dtype=np.int8)
        rec = run.oof_instrumented(data, 2, 12345)
        assert abs(rec['M'] - ((rec['S_RZ'] - rec['S_R']) - (rec['S_CRZ'] - rec['S_CR']))) < 1e-12
        assert len(rec['fold_details']) == run.FOLDS
        assert sum(fd['fold_size'] for fd in rec['fold_details']) == len(data)
        for fd in rec['fold_details']:
            assert abs(fd['M'] - (fd['G0'] - fd['G1'])) < 1e-12
    finally:
        est.GBM = orig
    return True


def _t_variance_decomposition():
    rng = np.random.default_rng(7); g0 = rng.standard_normal(400); g1 = 0.5 * g0 + rng.standard_normal(400); m = g0 - g1
    assert abs(np.var(m, ddof=1) - (np.var(g0, ddof=1) + np.var(g1, ddof=1) - 2 * np.cov(g0, g1, ddof=1)[0, 1])) < 1e-9
    return True


def _good_proof():
    return dict(commit_url='https://github.com/u/r/commit/' + 'a' * 40, commit_sha='a' * 40,
                package_sha256='b' * 64, timestamp_evidence_url='https://zenodo.org/records/12345',
                timestamp_evidence_type='zenodo', timestamp_utc='2026-07-11T12:00:00Z')


def _t_proof_schema_hardened():
    with tempfile.TemporaryDirectory() as d:
        cwd = os.getcwd(); os.chdir(d)
        try:
            assert not run.check_proof()[0]
            Path(run.PROOF).write_text(json.dumps({k: 'x' for k in run.PROOF_KEYS})); assert not run.check_proof()[0]
            Path(run.PROOF).write_text(json.dumps(_good_proof())); assert run.check_proof()[0]
            sha = 'a' * 40
            for bad in [{'commit_sha': 'zz'}, {'package_sha256': 'short'},
                        {'commit_url': 'http://github.com/u/r/commit/' + sha},
                        {'timestamp_utc': 'not-a-date'}, {'timestamp_evidence_type': 'osf'},
                        # substring-domain attack: evil host with zenodo.org in the PATH
                        {'timestamp_evidence_url': 'https://evil.example/zenodo.org/records/1'},
                        # SHA absent from canonical commit path
                        {'commit_url': 'https://github.com/u/r/commit/deadbeef'},
                        # query-only SHA attack
                        {'commit_url': 'https://github.com/u/r/commits?sha=' + sha},
                        # fragment-only SHA attack
                        {'commit_url': 'https://github.com/u/r/commit/deadbeef#' + sha},
                        # SHA elsewhere in path, not /commit/<sha>
                        {'commit_url': 'https://github.com/u/r/blob/' + sha + '/file'},
                        # extra path after SHA is noncanonical
                        {'commit_url': 'https://github.com/u/r/commit/' + sha + '/diff'},
                        # commit_url not on github
                        {'commit_url': 'https://gitlab.com/u/r/commit/' + sha}]:
                g = _good_proof(); g.update(bad); Path(run.PROOF).write_text(json.dumps(g))
                assert not run.check_proof()[0], bad
        finally:
            os.chdir(cwd)
    return True


def _t_package_hash():
    with tempfile.TemporaryDirectory() as d:
        cwd = os.getcwd(); os.chdir(d)
        try:
            Path(run.SHA_FILE).write_text('deadbeef  somefile\n')
            local = run._sha256(run.SHA_FILE)
            g = _good_proof(); g['package_sha256'] = local
            Path(run.PROOF).write_text(json.dumps(g)); assert run.check_package_hash()[0]
            g['package_sha256'] = 'c' * 64; Path(run.PROOF).write_text(json.dumps(g))
            assert not run.check_package_hash()[0]
        finally:
            os.chdir(cwd)
    return True


def _git_call(args, cwd):
    return subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True, text=True)


def _t_package_closure():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _git_call(['init', '-q'], root)
        _git_call(['config', 'user.email', 'test@example.invalid'], root)
        _git_call(['config', 'user.name', 'rc5-test'], root)
        (root / 'a.txt').write_text('a\n')
        (root / '.gitignore').write_text('__pycache__/\n')
        (root / run.SHA_FILE).write_text(
            f'{run._sha256(root / ".gitignore")}  .gitignore\n'
            f'{run._sha256(root / "a.txt")}  a.txt\n')
        _git_call(['add', '-A'], root); _git_call(['commit', '-qm', 'base'], root)
        cwd = os.getcwd(); os.chdir(root)
        try:
            assert run.check_git_package_closure({'.gitignore', 'a.txt'})[0]
            (root / 'sub').mkdir(); (root / 'sub' / 'extra.txt').write_text('extra\n')
            _git_call(['add', 'sub/extra.txt'], root); _git_call(['commit', '-qm', 'extra tracked'], root)
            assert not run.check_git_package_closure({'.gitignore', 'a.txt'})[0]
        finally:
            os.chdir(cwd)
    return True


def _t_env_lock():
    with tempfile.TemporaryDirectory() as d:
        cwd = os.getcwd(); os.chdir(d)
        try:
            assert not run.check_env_lock()[0]
            Path(run.ENV_LOCK).write_text(json.dumps({'_status': 'TEMPLATE'})); assert not run.check_env_lock()[0]
            locked = run.get_env(); locked['_status'] = 'LOCKED'
            Path(run.ENV_LOCK).write_text(json.dumps(locked)); assert run.check_env_lock()[0]
            locked['numpy'] = '0.0.0'; Path(run.ENV_LOCK).write_text(json.dumps(locked)); assert not run.check_env_lock()[0]
        finally:
            os.chdir(cwd)
    return True


def _t_stale_stage_rejected():
    with tempfile.TemporaryDirectory() as d:
        cwd = os.getcwd(); os.chdir(d)
        try:
            assert run.check_stale_stage_paths()[0]
            Path('nested').mkdir(); Path('nested/._oe_stage_old').mkdir()
            ok, errs = run.check_stale_stage_paths()
            assert not ok and any('nested/._oe_stage_old' in e for e in errs)
        finally:
            os.chdir(cwd)
    return True


def _synthetic_completion_frames(force_tier0_fail=False):
    import pandas as pd
    rep_rows, fold_rows, ndiag, decisions = [], [], [], []
    fold_sizes = [26, 26, 26, 25, 25]
    for ci in run.NULL_CELLS:
        vals = np.linspace(-0.004, 0.004, run.B) + ci * 0.0001
        if force_tier0_fail and ci == 0:
            vals = vals + 0.020
        for r, m in enumerate(vals):
            g0, g1 = 0.5 * m, -0.5 * m
            base = dict(cell_index=ci, kind='construction_null', order=run.ORDER, n=run.N,
                        pi_label=run.PI_LABELS[ci], target_pi=run.PI_LEVELS[ci], target_depth=0.0,
                        true_M=0.0, accuracy=0.8, alpha=0.1, replicate=r,
                        seed=run.bankseed(run.VAL_SEED, ci, r), M_gbm=m, M_tilde=m,
                        S_R=0.0, S_RZ=g0, S_CR=0.0, S_CRZ=g1, G0=g0, G1=g1,
                        min_support=1, zero_strata=0, mean_support=16.0)
            rep_rows.append(base)
            for fi, fs in enumerate(fold_sizes):
                fold_rows.append(dict(cell_index=ci, kind='construction_null', pi_label=run.PI_LABELS[ci],
                                      target_depth=0.0, replicate=r, seed=base['seed'], fold=fi,
                                      fold_size=fs, S_R=0.0, S_RZ=g0, S_CR=0.0, S_CRZ=g1,
                                      G0=g0, G1=g1, M=m))
        g = pd.DataFrame([x for x in rep_rows if x['cell_index'] == ci])
        a = run.tier0a_equivalence(g.M_tilde.to_numpy()); b = run.tier0b_coverage(g.M_tilde.to_numpy())
        sd = run.sign_diagnostics(g.M_tilde.to_numpy())
        decisions.append(dict(cell_index=ci, pi_label=run.PI_LABELS[ci],
                              **{f'a_{k}': v for k, v in a.items()},
                              **{f'b_{k}': v for k, v in b.items()},
                              **{f's_{k}': v for k, v in sd.items()}))
        v0 = float(np.var(g.G0, ddof=1)); v1 = float(np.var(g.G1, ddof=1))
        cov = float(np.cov(g.G0, g.G1, ddof=1)[0, 1]); vm = float(np.var(g.M_gbm, ddof=1))
        ndiag.append(dict(cell_index=ci, pi_label=run.PI_LABELS[ci], var_G0=v0, var_G1=v1,
                          cov_G0_G1=cov, var_M=vm,
                          decomposition_residual=vm - (v0 + v1 - 2 * cov)))
    return pd.DataFrame(rep_rows), pd.DataFrame(fold_rows), pd.DataFrame(ndiag), pd.DataFrame(decisions)


def _t_completion_integrity_hardened():
    rep, fold, ndf, decs = _synthetic_completion_frames()
    problems, metrics = run._integrity(rep, fold, ndf, decs, len(run.NULL_CELLS))
    assert not problems, problems
    assert metrics['identity_debias_max_abs'] == 0.0

    bad = rep.copy(); bad.loc[0, 'S_R'] = np.nan
    assert run._integrity(bad, fold, ndf, decs, len(run.NULL_CELLS))[0]
    bad = rep.copy(); bad.loc[0, 'M_tilde'] += 0.01
    assert any('M_tilde' in x for x in run._integrity(bad, fold, ndf, decs, len(run.NULL_CELLS))[0])
    bad = rep.copy(); bad.loc[0, 'G0'] += 0.01
    assert any('identity' in x for x in run._integrity(bad, fold, ndf, decs, len(run.NULL_CELLS))[0])
    assert any('rows' in x for x in run._integrity(rep, fold, ndf.iloc[:2], decs, len(run.NULL_CELLS))[0])
    bad_ndf = ndf.copy(); bad_ndf.loc[0, 'decomposition_residual'] = 1e-4
    assert any('variance decomposition' in x or 'raw replicate' in x
               for x in run._integrity(rep, fold, bad_ndf, decs, len(run.NULL_CELLS))[0])
    bad_decs = decs.copy(); bad_decs.loc[0, 'a_mean'] += 1e-3
    assert any('raw replicate recomputation' in x
               for x in run._integrity(rep, fold, ndf, bad_decs, len(run.NULL_CELLS))[0])
    return True


def _t_serialized_finalize_integrity():
    rep, fold, ndf, decs = _synthetic_completion_frames(force_tier0_fail=True)
    with tempfile.TemporaryDirectory() as d:
        cwd = os.getcwd(); os.chdir(d)
        try:
            stage = Path('._oe_stage_synthetic'); stage.mkdir()
            passed = run._finalize(rep, fold, ndf, decs, stage,
                                   tier1_generated=False, expected_cells=len(run.NULL_CELLS))
            assert not passed
            assert Path(run.COMPLETE).exists() and Path('TIER0_RESULT.json').exists()
            result = json.loads(Path('TIER0_RESULT.json').read_text())
            complete = json.loads(Path(run.COMPLETE).read_text())
            assert result['tier0a_pass'] is False and complete['tier1_generated'] is False
            assert not stage.exists()
            assert complete['completion_integrity']['identity_debias_max_abs'] <= run.ALGEBRA_TOL
        finally:
            os.chdir(cwd)
    return True


def _t_crash_burn_state_machine():
    with tempfile.TemporaryDirectory() as d:
        cwd = os.getcwd(); os.chdir(d)
        try:
            fd = os.open(run.START, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
            failed = False
            try:
                os.open(run.START, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                failed = True
            assert failed and Path(run.START).exists() and not Path(run.COMPLETE).exists()
        finally:
            os.chdir(cwd)
    return True


def _t_run_refuses_without_guards():
    """--run in a bare dir must exit nonzero and create NO START. The child's guards
    call run_all(include_subprocess=False), so no recursion occurs."""
    with tempfile.TemporaryDirectory() as d:
        for f in ('oe_estimator.py', 'oe_v03_run.py', 'oe_v03_tests.py', 'cell_table.csv', 'seed_manifest.csv'):
            if Path(f).exists(): Path(d, f).write_text(Path(f).read_text())
        r = subprocess.run([sys.executable, 'oe_v03_run.py', '--run'], capture_output=True, text=True, cwd=d, timeout=120)
        assert r.returncode != 0
        assert not Path(d, run.START).exists()
    return True


CORE = [_t_gates, _t_coverage, _t_four_score_identity, _t_variance_decomposition,
        _t_proof_schema_hardened, _t_package_hash, _t_package_closure, _t_env_lock,
        _t_stale_stage_rejected, _t_completion_integrity_hardened,
        _t_serialized_finalize_integrity, _t_crash_burn_state_machine]
SUBPROCESS = [_t_run_refuses_without_guards]


def run_all(verbose=True, include_subprocess=True):
    tests = CORE + (SUBPROCESS if include_subprocess else [])
    ok = True
    for t in tests:
        try:
            t()
            if verbose: print(f'  [PASS] {t.__name__}')
        except Exception as e:
            ok = False
            if verbose: print(f'  [FAIL] {t.__name__}: {e}')
    return ok


if __name__ == '__main__':
    sys.exit(0 if run_all(include_subprocess=True) else 1)
