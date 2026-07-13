#!/usr/bin/env python3
"""operating-envelope v0.3-rc5 -- k=2 base camp confirmatory runner.

Subcommands: --writeenv | --preflight | --run  (see PREREGISTRATION.md).

rc5 hardening (scientific core unchanged from rc4):
  1. Package closure: HEAD's tracked-file set must equal the SHA256SUMS target set
     plus SHA256SUMS.txt itself. Recursive un-hashed files and extra tracked files fail.
  2. Canonical commit URL: commit_sha must occupy the path
     /<owner>/<repo>/commit/<sha>; query/fragment-only inclusion fails.
  3. Completion integrity: serialized outputs are re-read and checked for finite
     replicate scores, identity debiasing, both M identities, exactly three null
     diagnostics, bounded variance-decomposition residuals, and raw-replicate
     recomputation of every saved Tier-0 decision.
  4. Any stale ._oe_stage_* path causes preflight refusal.

All guards are offset-free. No validation data are generated before the atomic START.
"""
from __future__ import annotations
import argparse, datetime, hashlib, itertools, json, math, os, platform, re, subprocess, sys, tempfile, shutil
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy.stats import beta, t as student_t

import oe_estimator as est

# ---- FROZEN CONSTANTS --------------------------------------------------------
VAL_SEED = 257841043574753
PRIOR_BASE = 20260711
B = 500
FOLDS = est.FOLDS
ORDER = 2
N = 128
PI_LABELS = ['low', 'medium', 'high']
PI_LEVELS = [.005, .015, .030]
DEPTHS = [.10, .15, .20]
M_MIN = 0.10

DELTA_BIAS = 0.10 * M_MIN
DELTA_REP = 0.50 * M_MIN
EPS = 0.05
ALPHA = 0.05
N_NULL_CELLS = 3
ALPHA_ADJ = ALPHA / N_NULL_CELLS
T_ADJ = float(student_t.ppf(1 - ALPHA_ADJ / 2, B - 1))

PROTOCOL_VERSION = 'v0.3-rc5'
PROOF = 'PREREGISTRATION_PUSHED.json'
START = 'VALIDATION_STARTED.json'
COMPLETE = 'VALIDATION_COMPLETE.json'
ENV_LOCK = 'env_lock.json'
CELL_TABLE = 'cell_table.csv'
SEED_MANIFEST = 'seed_manifest.csv'
SHA_FILE = 'SHA256SUMS.txt'
PROOF_KEYS = ('commit_url', 'commit_sha', 'package_sha256',
              'timestamp_evidence_url', 'timestamp_evidence_type', 'timestamp_utc')
EVIDENCE_DOMAINS = {'zenodo': 'zenodo.org', 'osf': 'osf.io', 'github_release': 'github.com'}
EXPECTED_CELLS = 12
NULL_CELLS = [0, 1, 2]
TIER1_CELLS = list(range(3, 12))
ALGEBRA_TOL = 1e-10
VAR_DECOMP_TOL = 1e-10
DECISION_FLOAT_TOL = 1e-10
ENV_PKGS = ('numpy', 'scipy', 'sklearn', 'pandas', 'joblib', 'threadpoolctl')
RUN_ARTEFACTS = (START, COMPLETE, 'validation_replicate_results.csv',
                 'validation_fold_results.csv', 'null_score_diagnostics.csv',
                 'tier0_decisions.csv', 'TIER0_RESULT.json')
HASHED_EXCLUDE = {SHA_FILE, PROOF, *RUN_ARTEFACTS}
ALLOWED_UNTRACKED = {PROOF}          # plus the __pycache__/ directory prefix


def cell_list():
    cells = [('construction_null', pi, 0.0) for pi in PI_LABELS]
    cells += [('tier1', pi, d) for pi in PI_LABELS for d in DEPTHS]
    return cells


def bankseed(base, cell_idx, rep):
    return int(np.random.SeedSequence([base, cell_idx, rep]).generate_state(1, dtype=np.uint32)[0])


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _excluded_dynamic_path(rel):
    rel = Path(rel).as_posix()
    if rel.startswith('./'):
        rel = rel[2:]
    parts = Path(rel).parts
    if not parts:
        return True
    if parts[0] == '.git' or any(part == '__pycache__' for part in parts):
        return True
    if any(part.startswith('._oe_stage_') for part in parts):
        return True
    return rel in HASHED_EXCLUDE


def _package_files_on_disk():
    """Return recursive non-dynamic regular files, excluding SHA256SUMS itself."""
    found, errs = set(), []
    for root, dirs, files in os.walk('.', topdown=True, followlinks=False):
        dirs[:] = [d for d in dirs if not _excluded_dynamic_path(Path(root, d))]
        for name in files:
            path = Path(root, name)
            rel = path.relative_to('.').as_posix()
            if _excluded_dynamic_path(rel):
                continue
            if path.is_symlink():
                errs.append(f'{rel}: symlink not allowed in package')
            elif not path.is_file():
                errs.append(f'{rel}: not a regular file')
            else:
                found.add(rel)
    return found, errs


# --------------------------------------------------- environment -------------
def get_env():
    import scipy, sklearn, pandas
    mods = {'numpy': np, 'scipy': scipy, 'sklearn': sklearn, 'pandas': pandas}
    for name in ('joblib', 'threadpoolctl'):
        try:
            mods[name] = __import__(name)
        except Exception:
            mods[name] = None
    env = {'python': sys.version.split()[0], 'platform': platform.platform(), 'machine': platform.machine()}
    for k in ENV_PKGS:
        env[k] = getattr(mods[k], '__version__', 'ABSENT') if mods[k] is not None else 'ABSENT'
    return env


def write_env():
    env = get_env(); env['_status'] = 'LOCKED'
    Path(ENV_LOCK).write_text(json.dumps(env, indent=2, sort_keys=True))
    files, errs = _package_files_on_disk()
    if errs:
        raise RuntimeError('package enumeration failed: ' + '; '.join(errs))
    with open(SHA_FILE, 'w') as out:
        for f in sorted(files):
            out.write(f'{_sha256(f)}  {f}\n')
    print(f'{ENV_LOCK} LOCKED and {SHA_FILE} regenerated.')
    print(f'package_sha256 for the proof = sha256({SHA_FILE}) = {_sha256(SHA_FILE)}')
    print('Commit exactly the SHA targets plus SHA256SUMS.txt, push, timestamp, then --preflight.')


# --------------------------------------------------- instrumented OOF M -------
def oof_instrumented(data, order, seed):
    y = data[:, -1].astype(int)
    folds = list(StratifiedKFold(FOLDS, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
    sets = est.predsets(order)
    scores = {p: np.empty(len(data)) for p in sets}
    fold_details = []
    for fi, (tr, te) in enumerate(folds):
        fs = {}
        for preds in sets:
            model = est.GBM(); model.fit(data[tr][:, preds], y[tr])
            p1 = model.predict_proba(data[te][:, preds])[:, 1]
            py = np.where(y[te] == 1, p1, 1 - p1)
            s = np.log(np.clip(py, 1e-12, 1))
            scores[preds][te] = s
            fs[preds] = float(np.mean(s))
        g0 = fs[sets[1]] - fs[sets[0]]; g1 = fs[sets[3]] - fs[sets[2]]
        fold_details.append(dict(fold=fi, fold_size=int(len(te)),
                                 S_R=fs[sets[0]], S_RZ=fs[sets[1]], S_CR=fs[sets[2]], S_CRZ=fs[sets[3]],
                                 G0=g0, G1=g1, M=g0 - g1))
    S0 = float(np.mean(scores[sets[0]])); S1 = float(np.mean(scores[sets[1]]))
    S2 = float(np.mean(scores[sets[2]])); S3 = float(np.mean(scores[sets[3]]))
    G0 = S1 - S0; G1 = S3 - S2; M = G0 - G1
    return dict(M=M, S_R=S0, S_RZ=S1, S_CR=S2, S_CRZ=S3, G0=G0, G1=G1, fold_details=fold_details)


def debias_identity(M):
    return M


# ------------------------------------------------------------- gates ----------
def tier0a_equivalence(m_tilde):
    x = np.asarray(m_tilde, float); n = len(x)
    mean = float(x.mean()); sd = float(x.std(ddof=1)); se = sd / math.sqrt(n)
    lo, hi = mean - T_ADJ * se, mean + T_ADJ * se
    return dict(mean=mean, sd=sd, ci_low=lo, ci_high=hi, passed=bool(lo >= -DELTA_BIAS and hi <= DELTA_BIAS))


def tier0b_coverage(m_tilde):
    x = np.asarray(m_tilde, float); n = len(x)
    X = int(np.sum(np.abs(x) <= DELTA_REP))
    lb = 0.0 if X == 0 else float(beta.ppf(ALPHA_ADJ, X, n - X + 1))
    return dict(successes=X, n=n, coverage_hat=X / n, cp_lower=lb, passed=bool(lb >= 1 - EPS))


def sign_diagnostics(m_tilde):
    x = np.asarray(m_tilde, float)
    return dict(mean=float(x.mean()), median=float(np.median(x)), pos_rate=float(np.mean(x > 0)),
                n_above=int(np.sum(x > DELTA_REP)), n_below=int(np.sum(x < -DELTA_REP)))


# ----------------------------------------------------- proof schema -----------
HEX40_64 = re.compile(r'^[0-9a-fA-F]{40}$|^[0-9a-fA-F]{64}$')
HEX64 = re.compile(r'^[0-9a-fA-F]{64}$')


def _host_ok(url, domain):
    h = (urlparse(url).hostname or '').lower()
    return h == domain or h.endswith('.' + domain)


def check_proof():
    p = Path(PROOF)
    if not p.exists():
        return False, [f'{PROOF} missing'], None
    try:
        d = json.loads(p.read_text())
    except Exception as e:
        return False, [f'{PROOF} not valid JSON: {e}'], None
    errs = [f'missing/empty key: {k}' for k in PROOF_KEYS if not d.get(k)]
    if not errs:
        if not HEX40_64.match(str(d['commit_sha'])): errs.append('commit_sha not 40/64 hex')
        if not HEX64.match(str(d['package_sha256'])): errs.append('package_sha256 not 64 hex')
        for u in ('commit_url', 'timestamp_evidence_url'):
            if urlparse(str(d[u])).scheme != 'https': errs.append(f'{u} not https')
        commit_url = urlparse(str(d['commit_url']))
        if not _host_ok(str(d['commit_url']), 'github.com'):
            errs.append('commit_url host != github.com')
        else:
            sha = str(d['commit_sha'])
            canonical = re.compile(r'^/[^/]+/[^/]+/commit/' + re.escape(sha) + r'/?$', re.IGNORECASE)
            if not canonical.fullmatch(commit_url.path):
                errs.append('commit_url path must be /owner/repo/commit/<commit_sha>')
        et = d.get('timestamp_evidence_type')
        if et not in EVIDENCE_DOMAINS:
            errs.append(f'timestamp_evidence_type must be one of {tuple(EVIDENCE_DOMAINS)}')
        elif not _host_ok(str(d['timestamp_evidence_url']), EVIDENCE_DOMAINS[et]):
            errs.append(f'evidence url host != {EVIDENCE_DOMAINS[et]}')
        try:
            ts = datetime.datetime.fromisoformat(str(d['timestamp_utc']).replace('Z', '+00:00'))
            if ts.utcoffset() is None or ts.utcoffset().total_seconds() != 0:
                errs.append('timestamp_utc not UTC')
        except Exception:
            errs.append('timestamp_utc not ISO-8601')
    return (not errs), errs, (d if not errs else None)


# ----------------------------------------------------- integrity helpers ------
def verify_sha256sums():
    p = Path(SHA_FILE)
    if not p.exists():
        return False, [f'{SHA_FILE} missing'], set()
    listed, bad = set(), []
    for lineno, line in enumerate(p.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            h, name = line.split(None, 1)
        except ValueError:
            bad.append(f'{SHA_FILE}:{lineno}: malformed line')
            continue
        name = name.strip().replace('\\', '/')
        if not HEX64.fullmatch(h):
            bad.append(f'{SHA_FILE}:{lineno}: hash not 64 hex')
        pp = Path(name)
        if pp.is_absolute() or '..' in pp.parts or name in ('', '.', SHA_FILE):
            bad.append(f'{SHA_FILE}:{lineno}: invalid target {name!r}')
            continue
        if name in listed:
            bad.append(f'{SHA_FILE}:{lineno}: duplicate target {name}')
            continue
        listed.add(name)
        if not Path(name).is_file() or Path(name).is_symlink():
            bad.append(f'{name}: missing, non-regular, or symlink')
            continue
        if HEX64.fullmatch(h) and _sha256(name) != h.lower():
            bad.append(f'{name}: hash mismatch')
    present, enum_errs = _package_files_on_disk()
    bad.extend(enum_errs)
    for f in sorted(present - listed):
        bad.append(f'{f}: present but not in SHA256SUMS')
    for f in sorted(listed - present):
        bad.append(f'{f}: listed but not present package file')
    return (not bad), bad, listed


def check_package_hash():
    ok, _, _ = check_proof()
    if not ok:
        return False, ['proof not schema-valid']
    d = json.loads(Path(PROOF).read_text())
    if not Path(SHA_FILE).exists():
        return False, [f'{SHA_FILE} missing']
    local = _sha256(SHA_FILE)
    if local != d['package_sha256']:
        return False, [f'package_sha256 != sha256({SHA_FILE}) (proof={d["package_sha256"][:12]} local={local[:12]})']
    return True, []


def _git(args):
    return subprocess.check_output(['git', *args], text=True, stderr=subprocess.DEVNULL)


def check_git_package_closure(listed):
    """HEAD must contain exactly SHA targets + SHA256SUMS.txt, all byte-identical."""
    try:
        tracked = set(_git(['ls-tree', '-r', 'HEAD', '--name-only']).splitlines())
    except Exception as e:
        return False, [f'git unavailable: {e}']
    expected = set(listed) | {SHA_FILE}
    errs = []
    for f in sorted(expected - tracked):
        errs.append(f'{f}: expected package file not tracked in HEAD')
    for f in sorted(tracked - expected):
        errs.append(f'{f}: tracked in HEAD but outside SHA package closure')
    for f in sorted(expected & tracked):
        try:
            if _git(['diff', '--name-only', 'HEAD', '--', f]).strip():
                errs.append(f'{f}: working tree differs from HEAD')
        except Exception as e:
            errs.append(f'{f}: git diff failed ({e})')
    return (not errs), errs


def check_untracked_allowlist():
    try:
        out = _git(['status', '--porcelain'])
    except Exception as e:
        return False, [f'git status failed: {e}']
    errs = []
    for line in out.splitlines():
        if not line.strip():
            continue
        code, path = line[:2], line[3:].strip()
        if code == '??':
            if path in ALLOWED_UNTRACKED or path.rstrip('/') == '__pycache__' or path.startswith('__pycache__/'):
                continue
            errs.append(f'untracked not allowed: {path}')
        else:
            errs.append(f'tracked change: {line.strip()}')
    return (not errs), errs


def check_stale_stage_paths():
    stale = []
    for root, dirs, files in os.walk('.', topdown=True, followlinks=False):
        dirs[:] = [d for d in dirs if d != '.git']
        for name in [*dirs, *files]:
            if name.startswith('._oe_stage_'):
                stale.append(Path(root, name).relative_to('.').as_posix())
    stale = sorted(set(stale))
    return (not stale), ([f'stale stage path: {x}' for x in stale] if stale else [])


def check_env_lock():
    p = Path(ENV_LOCK)
    if not p.exists():
        return False, [f'{ENV_LOCK} missing']
    try:
        locked = json.loads(p.read_text())
    except Exception as e:
        return False, [f'{ENV_LOCK} invalid JSON: {e}']
    if locked.get('_status') != 'LOCKED':
        return False, [f'{ENV_LOCK} is a TEMPLATE; run --writeenv in the confirmatory environment']
    actual = get_env(); diffs = []
    for k in ['python', 'platform', 'machine'] + list(ENV_PKGS):
        if str(locked.get(k)) != str(actual.get(k)):
            diffs.append(f'{k}: locked={locked.get(k)} actual={actual.get(k)}')
    return (not diffs), diffs


def check_cell_table():
    import pandas as pd
    if not Path(CELL_TABLE).exists():
        return False, [f'{CELL_TABLE} missing'], None
    t = pd.read_csv(CELL_TABLE)
    errs = []
    if list(t.cell_index) != list(range(EXPECTED_CELLS)):
        return False, ['cell_index not 0..11'], None
    for i, (kind, pi, depth) in enumerate(cell_list()):
        r = t[t.cell_index == i].iloc[0]
        if r.kind != kind: errs.append(f'cell {i} kind={r.kind}!={kind}')
        if r.pi_label != pi: errs.append(f'cell {i} pi_label')
        if abs(float(r.target_depth) - depth) > 1e-12: errs.append(f'cell {i} target_depth')
        if abs(float(r.target_pi) - PI_LEVELS[PI_LABELS.index(pi)]) > 1e-12: errs.append(f'cell {i} target_pi')
        if abs(float(r.target_pi) - float(r.pi_max)) > 1e-9: errs.append(f'cell {i} target_pi!=pi_max')
        if kind == 'construction_null':
            if abs(float(r.true_M)) > 1e-6: errs.append(f'cell {i} null true_M!=0')
        else:
            if abs(float(r.true_M) + depth) > 1e-9: errs.append(f'cell {i} true_M!=-depth')
        mm = est.metrics(ORDER, float(r.accuracy), float(r.alpha))
        if abs(mm['true_M'] - float(r.true_M)) > 1e-9 or abs(mm['pi_max'] - float(r.pi_max)) > 1e-9:
            errs.append(f'cell {i} metrics mismatch')
    return (not errs), errs, t


def check_seed_manifest():
    import pandas as pd
    if not Path(SEED_MANIFEST).exists():
        return False, [f'{SEED_MANIFEST} missing']
    mf = pd.read_csv(SEED_MANIFEST); errs = []
    if len(mf) != EXPECTED_CELLS * B:
        errs.append(f'manifest rows {len(mf)} != {EXPECTED_CELLS*B}')
    exp = cell_list()
    for _, r in mf.iterrows():
        ci = int(r.cell_index)
        if bankseed(VAL_SEED, ci, int(r.replicate)) != int(r.seed):
            errs.append('seed mismatch'); break
        kind, pi, depth = exp[ci]
        if r.kind != kind or r.pi_label != pi or abs(float(r.target_depth) - depth) > 1e-12 \
                or int(r.order) != ORDER or int(r.n) != N:
            errs.append(f'metadata mismatch at cell {ci}'); break
    seeds = list(mf.seed)
    if len(set(seeds)) != len(seeds):
        errs.append('duplicate seeds')
    prior = set(int(np.random.SeedSequence([PRIOR_BASE, PI_LABELS.index(l) + 1, 99 if d == 0 else DEPTHS.index(d) + 1, r]).generate_state(1, dtype=np.uint32)[0])
                for l in PI_LABELS for d in [0.] + DEPTHS for r in range(B))
    if set(seeds) & prior:
        errs.append('overlap with prior grid')
    return (not errs), errs


# ------------------------------------------------- offset-free guard set ------
def run_offset_free_guards(verbose=True):
    results = []
    def add(name, ok, detail=''):
        results.append((name, ok, detail))
    h_ok, h_bad, listed = verify_sha256sums(); add('SHA256SUMS complete & matching', h_ok, '; '.join(h_bad))
    p_ok, p_err, proof = check_proof(); add('proof JSON schema', p_ok, '; '.join(p_err))
    if p_ok: add(*(('package_sha256 == sha256(SHA256SUMS.txt)',) + check_package_hash()))
    try:
        head = _git(['rev-parse', 'HEAD']).strip()
        add('git HEAD == commit_sha', bool(proof) and head == proof['commit_sha'],
            '' if (proof and head == proof['commit_sha']) else 'HEAD/commit_sha mismatch or no proof')
        add(*(('HEAD package closure == SHA targets + SHA256SUMS',) + check_git_package_closure(listed)))
        add(*(('untracked allowlist',) + check_untracked_allowlist()))
        add(*(('no stale ._oe_stage_* paths',) + check_stale_stage_paths()))
    except Exception as e:
        add('git repo present', False, str(e))
    add(*(('environment matches env_lock',) + check_env_lock()))
    c_ok, c_bad, _ = check_cell_table(); add('cell_table semantics & consistency', c_ok, '; '.join(c_bad))
    add(*(('seed manifest == code & disjoint',) + check_seed_manifest()))
    stray = [f for f in RUN_ARTEFACTS if Path(f).exists()]
    add('output clean (no run artefacts)', not stray, ', '.join(stray))
    try:
        import oe_v03_tests
        add('offset-free core unit tests', oe_v03_tests.run_all(verbose=False, include_subprocess=False))
    except Exception as e:
        add('offset-free core unit tests', False, str(e))
    ok = all(r[1] for r in results)
    if verbose:
        for name, passed, detail in results:
            print(f'  [{"PASS" if passed else "FAIL"}] {name}' + (f' -- {detail}' if detail else ''))
    return ok, results


# ------------------------------------------------- confirmatory run -----------
def _claim_bank_or_die():
    if Path(COMPLETE).exists():
        sys.stderr.write('\nREFUSED: VALIDATION_COMPLETE.json exists -- this bank already completed.\n'); sys.exit(3)
    try:
        fd = os.open(START, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        sys.stderr.write(
            '\nREFUSED: VALIDATION_STARTED.json exists without VALIDATION_COMPLETE.json.\n'
            'Consumed bank; confirmatory standing lost. No resume/lock-deletion/same-seed re-run.\n'
            'A new attempt needs a new unexposed bank, manifest, package hash, timestamp, and an\n'
            'explicit aborted record (e.g. v0.3.1).\n\n')
        sys.exit(4)
    proof = json.loads(Path(PROOF).read_text())
    marker = dict(protocol_version=PROTOCOL_VERSION, preregistration_commit_sha=_git(['rev-parse', 'HEAD']).strip(),
                  proof_sha256=_sha256(PROOF), package_sha256=proof.get('package_sha256'),
                  seed_manifest_sha256=_sha256(SEED_MANIFEST), cell_table_sha256=_sha256(CELL_TABLE),
                  sha256sums_sha256=_sha256(SHA_FILE), env=get_env(),
                  start_utc=datetime.datetime.utcnow().isoformat() + 'Z', argv=' '.join(sys.argv))
    with os.fdopen(fd, 'w') as f:
        json.dump(marker, f, indent=2)


def _generate_cells(table, indices):
    import pandas as pd
    rep_rows, fold_rows = [], []
    predcols = tuple(range(ORDER + 1)); strata = list(itertools.product([0, 1], repeat=ORDER + 1))
    for ci in indices:
        r = table[table.cell_index == ci].iloc[0]; a, w = float(r.accuracy), float(r.alpha)
        states, probs = est.joint_distribution(ORDER, a, w)
        for rep in range(B):
            seed = bankseed(VAL_SEED, ci, rep); rng = np.random.default_rng(seed)
            data = states[rng.choice(len(states), size=N, p=probs)].copy()
            gi = oof_instrumented(data, ORDER, seed)
            counts = defaultdict(int)
            for row in data: counts[tuple(int(row[j]) for j in predcols)] += 1
            sc = np.array([counts[s] for s in strata])
            rep_rows.append(dict(cell_index=ci, kind=r.kind, order=ORDER, n=N, pi_label=r.pi_label,
                                 target_pi=float(r.target_pi), target_depth=float(r.target_depth),
                                 true_M=float(r.true_M), accuracy=a, alpha=w, replicate=rep, seed=seed,
                                 M_gbm=gi['M'], M_tilde=debias_identity(gi['M']),
                                 S_R=gi['S_R'], S_RZ=gi['S_RZ'], S_CR=gi['S_CR'], S_CRZ=gi['S_CRZ'],
                                 G0=gi['G0'], G1=gi['G1'], min_support=int(sc.min()),
                                 zero_strata=int(np.sum(sc == 0)), mean_support=float(sc.mean())))
            for fdt in gi['fold_details']:
                fold_rows.append(dict(cell_index=ci, kind=r.kind, pi_label=r.pi_label,
                                      target_depth=float(r.target_depth), replicate=rep, seed=seed, **fdt))
    return pd.DataFrame(rep_rows), pd.DataFrame(fold_rows)


def _coerce_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {'true', 'false'}:
        return value.strip().lower() == 'true'
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    raise ValueError(f'not a boolean value: {value!r}')


def _finite_columns(df, columns, label, problems):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        problems.append(f'{label}: missing columns {missing}')
        return False
    try:
        arr = df[columns].to_numpy(dtype=float)
    except Exception:
        arr = df[columns].apply(lambda x: __import__('pandas').to_numeric(x, errors='coerce')).to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        problems.append(f'{label}: missing or non-finite values')
        return False
    return True


def _integrity(rep, fold, ndf, decs, expected_cells):
    problems = []
    metrics = dict(replicate_four_score_max_abs=0.0, fold_four_score_max_abs=0.0,
                   identity_debias_max_abs=0.0,
                   variance_diagnostic_recalculation_max_abs=0.0,
                   variance_decomposition_residual_max_abs=0.0,
                   tier0_recalculation_max_abs=0.0)
    if expected_cells not in (len(NULL_CELLS), EXPECTED_CELLS):
        problems.append(f'unsupported expected_cells={expected_cells}')
    expected_cell_set = set(NULL_CELLS if expected_cells == len(NULL_CELLS) else range(EXPECTED_CELLS))

    rep_required = ['cell_index', 'replicate', 'seed', 'S_R', 'S_RZ', 'S_CR', 'S_CRZ',
                    'G0', 'G1', 'M_gbm', 'M_tilde']
    fold_required = ['cell_index', 'replicate', 'fold', 'fold_size', 'S_R', 'S_RZ',
                     'S_CR', 'S_CRZ', 'G0', 'G1', 'M']
    rep_finite = _finite_columns(rep, rep_required, 'replicate output', problems)
    fold_finite = _finite_columns(fold, fold_required, 'fold output', problems)

    if rep.shape[0] != expected_cells * B:
        problems.append(f'replicate rows {rep.shape[0]} != {expected_cells*B}')
    if fold.shape[0] != expected_cells * B * FOLDS:
        problems.append(f'fold rows {fold.shape[0]} != {expected_cells*B*FOLDS}')
    if 'cell_index' in rep and set(rep.cell_index.astype(int).unique()) != expected_cell_set:
        problems.append('replicate cell set mismatch')
    if 'cell_index' in fold and set(fold.cell_index.astype(int).unique()) != expected_cell_set:
        problems.append('fold cell set mismatch')

    if {'cell_index', 'replicate'} <= set(rep.columns):
        sizes = rep.groupby('cell_index').size()
        if len(sizes) != expected_cells or not (sizes == B).all():
            problems.append('not all cells have exactly B replicates')
        if rep.duplicated(['cell_index', 'replicate']).any():
            problems.append('duplicate replicate keys')
    if {'cell_index', 'replicate', 'fold'} <= set(fold.columns):
        if fold.duplicated(['cell_index', 'replicate', 'fold']).any():
            problems.append('duplicate fold keys')
        fc = fold.groupby(['cell_index', 'replicate']).fold.agg(['count', 'nunique'])
        if not ((fc['count'] == FOLDS).all() and (fc['nunique'] == FOLDS).all()):
            problems.append('not exactly 5 distinct folds/replicate')
        if set(fold.fold.astype(int).unique()) != set(range(FOLDS)):
            problems.append('fold indices != {0..4}')
        if 'fold_size' in fold and not (fold.groupby(['cell_index', 'replicate']).fold_size.sum() == N).all():
            problems.append('fold sizes != 128')

    if rep_finite:
        for _, rr in rep.iterrows():
            if int(rr.seed) != bankseed(VAL_SEED, int(rr.cell_index), int(rr.replicate)):
                problems.append('replicate seed != manifest')
                break
        e_debias = float(np.max(np.abs(rep.M_tilde.to_numpy(float) - rep.M_gbm.to_numpy(float))))
        e_rg = float(np.max(np.abs(rep.M_gbm.to_numpy(float) -
                                    (rep.G0.to_numpy(float) - rep.G1.to_numpy(float)))))
        e_rs = float(np.max(np.abs(rep.M_gbm.to_numpy(float) -
                                    ((rep.S_RZ.to_numpy(float) - rep.S_R.to_numpy(float)) -
                                     (rep.S_CRZ.to_numpy(float) - rep.S_CR.to_numpy(float))))))
        metrics['identity_debias_max_abs'] = e_debias
        metrics['replicate_four_score_max_abs'] = max(e_rg, e_rs)
        if e_debias > ALGEBRA_TOL:
            problems.append(f'M_tilde != M_gbm ({e_debias:.2e})')
        if max(e_rg, e_rs) > ALGEBRA_TOL:
            problems.append(f'replicate M identity violated ({max(e_rg,e_rs):.2e})')

    if fold_finite:
        e_fg = float(np.max(np.abs(fold.M.to_numpy(float) -
                                    (fold.G0.to_numpy(float) - fold.G1.to_numpy(float)))))
        e_fs = float(np.max(np.abs(fold.M.to_numpy(float) -
                                    ((fold.S_RZ.to_numpy(float) - fold.S_R.to_numpy(float)) -
                                     (fold.S_CRZ.to_numpy(float) - fold.S_CR.to_numpy(float))))))
        metrics['fold_four_score_max_abs'] = max(e_fg, e_fs)
        if max(e_fg, e_fs) > ALGEBRA_TOL:
            problems.append(f'fold M identity violated ({max(e_fg,e_fs):.2e})')

    ndiag_cols = ['cell_index', 'var_G0', 'var_G1', 'cov_G0_G1', 'var_M', 'decomposition_residual']
    ndf_finite = _finite_columns(ndf, ndiag_cols, 'null_score_diagnostics.csv', problems)
    if len(ndf) != N_NULL_CELLS:
        problems.append(f'null_score_diagnostics.csv rows {len(ndf)} != {N_NULL_CELLS}')
    if 'cell_index' in ndf and (ndf.duplicated(['cell_index']).any() or
                                set(ndf.cell_index.astype(int).unique()) != set(NULL_CELLS)):
        problems.append('null_score_diagnostics cell set/uniqueness mismatch')
    if ndf_finite and rep_finite and len(ndf) == N_NULL_CELLS:
        vmax, residual_max = 0.0, 0.0
        for ci in NULL_CELLS:
            rows = ndf[ndf.cell_index.astype(int) == ci]
            if len(rows) != 1:
                continue
            row = rows.iloc[0]
            g = rep[rep.cell_index.astype(int) == ci]
            exp = dict(var_G0=float(np.var(g.G0, ddof=1)), var_G1=float(np.var(g.G1, ddof=1)),
                       cov_G0_G1=float(np.cov(g.G0, g.G1, ddof=1)[0, 1]),
                       var_M=float(np.var(g.M_gbm, ddof=1)))
            exp['decomposition_residual'] = exp['var_M'] - (exp['var_G0'] + exp['var_G1'] - 2 * exp['cov_G0_G1'])
            for key, val in exp.items():
                diff = abs(float(row[key]) - val); vmax = max(vmax, diff)
                if diff > DECISION_FLOAT_TOL:
                    problems.append(f'null diagnostics cell {ci} {key} != raw replicate recomputation')
            residual_max = max(residual_max, abs(float(row.decomposition_residual)),
                               abs(exp['decomposition_residual']))
            if abs(float(row.decomposition_residual)) > VAR_DECOMP_TOL or abs(exp['decomposition_residual']) > VAR_DECOMP_TOL:
                problems.append(f'variance decomposition residual exceeds tolerance at cell {ci}')
            if 'pi_label' not in ndf.columns or str(row.get('pi_label', '')) != PI_LABELS[ci]:
                problems.append(f'null diagnostics cell {ci} pi_label mismatch')
        metrics['variance_diagnostic_recalculation_max_abs'] = vmax
        metrics['variance_decomposition_residual_max_abs'] = residual_max

    decision_float = ['a_mean', 'a_sd', 'a_ci_low', 'a_ci_high', 'b_coverage_hat', 'b_cp_lower',
                      's_mean', 's_median', 's_pos_rate']
    decision_int = ['b_successes', 'b_n', 's_n_above', 's_n_below']
    decision_bool = ['a_passed', 'b_passed']
    dec_required = ['cell_index', 'pi_label', *decision_float, *decision_int, *decision_bool]
    missing_dec = [c for c in dec_required if c not in decs.columns]
    if missing_dec:
        problems.append(f'tier0_decisions.csv: missing columns {missing_dec}')
    if len(decs) != N_NULL_CELLS:
        problems.append(f'tier0_decisions.csv rows {len(decs)} != {N_NULL_CELLS}')
    if 'cell_index' in decs and (decs.duplicated(['cell_index']).any() or
                                 set(decs.cell_index.astype(int).unique()) != set(NULL_CELLS)):
        problems.append('tier0 decision cell set/uniqueness mismatch')
    if not missing_dec and len(decs) == N_NULL_CELLS and rep_finite:
        dmax = 0.0
        for ci in NULL_CELLS:
            rows = decs[decs.cell_index.astype(int) == ci]
            if len(rows) != 1:
                continue
            row = rows.iloc[0]
            g = rep[rep.cell_index.astype(int) == ci]
            mt = g.M_tilde.to_numpy(float)
            a = tier0a_equivalence(mt); b = tier0b_coverage(mt); sd = sign_diagnostics(mt)
            expected = {**{f'a_{k}': v for k, v in a.items()},
                        **{f'b_{k}': v for k, v in b.items()},
                        **{f's_{k}': v for k, v in sd.items()}}
            expected_pi = PI_LABELS[ci]
            if str(row.pi_label) != expected_pi:
                problems.append(f'tier0 decision cell {ci} pi_label mismatch')
            for key in decision_float:
                try:
                    saved = float(row[key]); target = float(expected[key])
                except Exception:
                    problems.append(f'tier0 decision cell {ci} {key} non-numeric')
                    continue
                if not (math.isfinite(saved) and math.isfinite(target)):
                    problems.append(f'tier0 decision cell {ci} {key} non-finite')
                    continue
                diff = abs(saved - target); dmax = max(dmax, diff)
                if diff > DECISION_FLOAT_TOL:
                    problems.append(f'tier0 decision cell {ci} {key} != raw replicate recomputation')
            for key in decision_int:
                try:
                    if int(row[key]) != int(expected[key]):
                        problems.append(f'tier0 decision cell {ci} {key} != raw replicate recomputation')
                except Exception:
                    problems.append(f'tier0 decision cell {ci} {key} invalid integer')
            for key in decision_bool:
                try:
                    if _coerce_bool(row[key]) != bool(expected[key]):
                        problems.append(f'tier0 decision cell {ci} {key} != raw replicate recomputation')
                except Exception:
                    problems.append(f'tier0 decision cell {ci} {key} invalid boolean')
        metrics['tier0_recalculation_max_abs'] = dmax

    return problems, metrics


def _finalize(rep, fold, ndf, decs, stage, tier1_generated, expected_cells):
    import pandas as pd
    rep.to_csv(stage / 'validation_replicate_results.csv', index=False)
    fold.to_csv(stage / 'validation_fold_results.csv', index=False)
    ndf.to_csv(stage / 'null_score_diagnostics.csv', index=False)
    decs.to_csv(stage / 'tier0_decisions.csv', index=False)
    try:
        rep_saved = pd.read_csv(stage / 'validation_replicate_results.csv')
        fold_saved = pd.read_csv(stage / 'validation_fold_results.csv')
        ndf_saved = pd.read_csv(stage / 'null_score_diagnostics.csv')
        decs_saved = pd.read_csv(stage / 'tier0_decisions.csv')
        problems, integ = _integrity(rep_saved, fold_saved, ndf_saved, decs_saved, expected_cells)
    except Exception as e:
        problems, integ = [f'serialized output read/integrity exception: {e}'], {}
    if problems:
        sys.stderr.write('\nINTEGRITY FAILURE (bank stays burned, no COMPLETE):\n  ' + '\n  '.join(problems) + '\n')
        sys.exit(5)

    tier0a = all(_coerce_bool(v) for v in decs_saved.a_passed)
    tier0b = all(_coerce_bool(v) for v in decs_saved.b_passed)
    if tier1_generated != tier0a:
        sys.stderr.write('\nINTEGRITY FAILURE (bank stays burned, no COMPLETE):\n'
                         '  tier1_generated must equal the verified Tier-0a decision\n')
        sys.exit(5)
    result = dict(protocol_version=PROTOCOL_VERSION, tier0a_pass=tier0a, tier0b_pass=tier0b,
                  tier1_generated=tier1_generated, delta_bias=DELTA_BIAS, delta_rep=DELTA_REP,
                  eps=EPS, t_adj=T_ADJ,
                  decision=('base camp secured; k=2 Tier-1 may be read' if tier0a
                            else 'v0.3 confirmatory FAIL; no Tier-1 exposed; repair belongs to v0.4'))
    (stage / 'TIER0_RESULT.json').write_text(json.dumps(result, indent=2))
    saved_result = json.loads((stage / 'TIER0_RESULT.json').read_text())
    if saved_result != result:
        sys.stderr.write('\nINTEGRITY FAILURE (bank stays burned, no COMPLETE):\n'
                         '  TIER0_RESULT.json round-trip mismatch\n')
        sys.exit(5)

    outs = ['validation_replicate_results.csv', 'validation_fold_results.csv',
            'null_score_diagnostics.csv', 'tier0_decisions.csv', 'TIER0_RESULT.json']
    for f in outs:
        os.replace(stage / f, f)
    shutil.rmtree(stage, ignore_errors=True)
    complete = dict(protocol_version=PROTOCOL_VERSION, tier1_generated=tier1_generated,
                    total_replicate_rows=int(rep_saved.shape[0]), total_fold_rows=int(fold_saved.shape[0]),
                    four_score_identity_max_abs=max(integ['replicate_four_score_max_abs'],
                                                    integ['fold_four_score_max_abs']),
                    completion_integrity=integ,
                    output_sha256={f: _sha256(f) for f in outs},
                    complete_utc=datetime.datetime.utcnow().isoformat() + 'Z')
    tmp = COMPLETE + '.tmp'; Path(tmp).write_text(json.dumps(complete, indent=2)); os.replace(tmp, COMPLETE)
    return tier0a


def confirmatory_run():
    import pandas as pd
    ok, _ = run_offset_free_guards(verbose=True)
    if not ok:
        sys.stderr.write('\nFAIL-CLOSED: offset-free guards did not all pass; bank not claimed.\n'); sys.exit(2)
    _, _, table = check_cell_table()
    _claim_bank_or_die()
    stage = Path(tempfile.mkdtemp(prefix='._oe_stage_', dir='.'))
    rep0, fold0 = _generate_cells(table, NULL_CELLS)
    dec, ndiag = [], []
    for ci in NULL_CELLS:
        g = rep0[rep0.cell_index == ci]; mt = g.M_tilde.to_numpy()
        a_ = tier0a_equivalence(mt); b_ = tier0b_coverage(mt); s_ = sign_diagnostics(mt)
        dec.append(dict(cell_index=ci, pi_label=g.pi_label.iloc[0], **{f'a_{k}': v for k, v in a_.items()},
                        **{f'b_{k}': v for k, v in b_.items()}, **{f's_{k}': v for k, v in s_.items()}))
        vG0 = float(np.var(g.G0, ddof=1)); vG1 = float(np.var(g.G1, ddof=1))
        cov = float(np.cov(g.G0, g.G1, ddof=1)[0, 1]); vM = float(np.var(g.M_gbm, ddof=1))
        ndiag.append(dict(cell_index=ci, pi_label=g.pi_label.iloc[0], var_G0=vG0, var_G1=vG1,
                          cov_G0_G1=cov, var_M=vM, decomposition_residual=vM - (vG0 + vG1 - 2 * cov)))
    decs = pd.DataFrame(dec); ndf = pd.DataFrame(ndiag)
    if not bool(decs.a_passed.all()):
        t0 = _finalize(rep0, fold0, ndf, decs, stage, tier1_generated=False, expected_cells=len(NULL_CELLS))
        print('CONFIRMATORY RUN COMPLETE. Tier-0a PASS: False (Tier-1 not generated).'); return
    rep1, fold1 = _generate_cells(table, TIER1_CELLS)
    rep = pd.concat([rep0, rep1], ignore_index=True); fold = pd.concat([fold0, fold1], ignore_index=True)
    _finalize(rep, fold, ndf, decs, stage, tier1_generated=True, expected_cells=EXPECTED_CELLS)
    print('CONFIRMATORY RUN COMPLETE. Tier-0a PASS: True (Tier-1 generated).')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--writeenv', action='store_true')
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--run', action='store_true')
    a = ap.parse_args()
    if sum([a.writeenv, a.preflight, a.run]) != 1:
        sys.stderr.write('choose exactly one of --writeenv / --preflight / --run\n'); sys.exit(1)
    if a.writeenv: write_env(); return
    if a.preflight:
        print('PREFLIGHT (offset-free):'); ok, _ = run_offset_free_guards(verbose=True)
        print('PREFLIGHT', 'OK' if ok else 'FAILED'); sys.exit(0 if ok else 1)
    if a.run: confirmatory_run()


if __name__ == '__main__':
    main()
