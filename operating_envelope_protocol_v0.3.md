# Operating-envelope protocol v0.3-rc5 (k=2 base camp)

## Status and epistemic scope

Preregistered confirmatory protocol. Confirmatory **only** once this package is
externally timestamped (third-party time record) **before** any validation dataset is
generated. Until then: exploratory draft, no verification standing.

Scope: **k=2 within-order base camp only**. Cross-order remains frozen. Frozen
operational status (unchanged): JES manuscript fixed; reviewer reserve fixed; ASSB
application paper frozen; cross-order quarantined; k=2 base camp only.

## Object and contrast

Frozen deterministic binary Newton GBM (`oe_estimator.py`, depth 3, 30 trees, lr 0.05,
l2 0.5, min_child 5), identical to the exploratory phase. `M = I(Z;Y|R) - I(Z;Y|C,R)`,
estimated OOF as `M_hat = G0 - G1`, `G0 = S(R,Z)-S(R)`, `G1 = S(C,R,Z)-S(C,R)`.

## Primary question and Tier-0 gate

Does the estimator pass the construction-null bias-amplitude gate at B=500, k=2, n=128,
across pi_max in {low, medium, high}?

**Tier-0a (PRIMARY; licenses Tier-1).** Bonferroni simultaneous equivalence CI on the
bias: PASS iff every construction-null cell's two-sided `1 - ALPHA/N_null` Student-t CI
(`df=499`, `t=2.402079`) for the mean debiased estimate lies wholly within
`[-DELTA_BIAS, +DELTA_BIAS]`; all three cells must pass. B-robust.
`DELTA_BIAS = 0.10*m_min = 0.01`. Principle: the null bias may shift fractional
compression by at most 0.10 at the smallest-|M*| cell. Provenance: exploratory offsets
were known before preregistration; the margin was a prospective 10%-of-m_min tolerance,
not fitted to validation data, fixed before any validation output existed.

**Tier-0b (SECONDARY).** One-sided Clopper-Pearson lower bound (confidence
`1-ALPHA/N_null`) on `P(|M_tilde|<=DELTA_REP)` `>= 1-EPS`. `DELTA_REP=0.05`, `EPS=0.05`.
Minimum passing count 486/500.

## Debiasing and two-stage execution

Identity only; no fitted debiasing, no fallback, no calibration bank. Under a single
START marker: generate the three construction-null cells first, evaluate Tier-0a, and
generate the nine Tier-1 cells **only if Tier-0a passes**. A Tier-0a FAIL exposes no
Tier-1 replicate; v0.3 ends as a confirmatory FAIL and repair moves to a separately
timestamped v0.4.

## Instrumentation and cell table

Per replicate and per fold: `S(R), S(R,Z), S(C,R), S(C,R,Z)`, `G0`, `G1`, `M`, fold
index, fold size. Per null cell: `Var(G0), Var(G1), Cov(G0,G1), Var(M)` and the
decomposition residual. Generator cells are LOADED from the hashed `cell_table.csv` (no
solver after START). Guards verify the table's semantics: cells 0-2 are
`construction_null` with `true_M≈0`; cells 3-11 are `tier1` on the expected
pi_max x |M*| grid with `true_M == -target_depth`; `target_pi == pi_max`; and
`metrics(accuracy, alpha)` reproduces `true_M` and `pi_max` to 1e-9. The seed manifest's
metadata (kind, pi_label, depth, order, n) is checked against the table and constants.

## Environment lock

`env_lock.json` ships as a TEMPLATE. `--writeenv`, run in the confirmatory environment,
records interpreter/numpy/scipy/scikit-learn/pandas/joblib/threadpoolctl/OS-arch, sets
`_status=LOCKED`, and regenerates `SHA256SUMS.txt`. `requirements.txt` pins all seven
libraries. `--preflight`/`--run` refuse unless the actual environment equals the lock.

## Offset-free guards enforced by --run

`--preflight` runs the guard set and reports (no data). `--run` re-runs the SAME guard
set and reaches `_claim_bank_or_die()` only if every guard passes. Guards (all
offset-free):
- recursive SHA256SUMS completeness and matching; duplicate, malformed, missing,
  symlink, and un-hashed recursive targets fail;
- proof JSON schema (see below);
- `package_sha256 == sha256(SHA256SUMS.txt)`;
- `git HEAD == commit_sha`;
- **exact package closure**: the HEAD tracked-file set equals the SHA target set plus
  `SHA256SUMS.txt`, and every member is byte-identical to HEAD. Thus an empty commit,
  un-hashed subdirectory, or extra tracked file fails;
- **untracked allowlist**: `git status --porcelain` shows nothing except the proof file
  and `__pycache__/` (a shipped `.gitignore` keeps those from being committed);
- no stale `._oe_stage_*` file or directory;
- environment == env_lock; cell_table semantics; seed manifest == code and disjoint;
- output directory clean; offset-free core unit tests (recursion-free: the subprocess
  meta-test is excluded from the guard-invoked suite).

## Crash-burn and completion integrity

START is created atomically (`O_EXCL`) before any data. If START exists without COMPLETE,
the bank is consumed and has lost confirmatory standing (no resume/lock-deletion/same-seed
re-run; a new attempt needs a fresh unexposed bank, manifest, hash, timestamp, and an
aborted record e.g. v0.3.1). A stale `._oe_stage_*` path is a preflight failure. Before
COMPLETE, all staged CSVs are written, re-read, and integrity-checked: replicate rows =
cells*500; fold rows = cells*500*5; 500 replicates/cell; no duplicate replicate/fold keys;
exactly 5 distinct folds/replicate with indices {0..4} and fold sizes summing to 128;
saved seeds == manifest; replicate-level scores (`S_R`, `S_RZ`, `S_CR`, `S_CRZ`,
`G0`, `G1`, `M_gbm`, `M_tilde`) all present and finite; `M_tilde=M_gbm`; `M_gbm=G0-G1`; and
`M_gbm=(S_RZ-S_R)-(S_CRZ-S_CR)`. Fold-level identities remain checked to 1e-10.
`null_score_diagnostics.csv` must have exactly three unique null-cell rows; every saved
variance/covariance field is recomputed from raw replicates and each decomposition
residual must be within 1e-10. `tier0_decisions.csv` must have exactly three unique
null-cell rows, and every saved float, count, and PASS boolean is recomputed from raw
replicates. Only then are outputs promoted and COMPLETE written atomically.

## External timestamp requirement (with a manual gate)

`PREREGISTRATION_PUSHED.json` is format-checked: `commit_sha` 40/64 hex; `package_sha256`
64 hex and equal to `sha256(SHA256SUMS.txt)`; URLs https with `urlparse` hostname on the
right domain (`commit_url` on github.com with canonical path
`/<owner>/<repo>/commit/<commit_sha>`; query/fragment-only SHA inclusion is rejected;
evidence url on
zenodo.org / osf.io / github.com per `timestamp_evidence_type`); `timestamp_utc` ISO-8601
UTC. **Manual external gate:** the runner checks format only; whether the external record
certifies THIS exact commit/package must be verified by the user before placing the
proof. `package_sha256 = sha256(SHA256SUMS.txt)` is a content commitment to the whole
package, computed after `--writeenv` and enforced against the local file.

## Epistemic grade

Unsigned, third-party-timestamped preregistration. The record attests this content
existed before validation; it does not cryptographically attest the author's private
state.

## Primary decision

Tier-0a PASS on all three null cells => base camp secured; k=2 Tier-1 envelope may be
read. Tier-0a FAIL on any cell => v0.3 confirmatory FAIL (no Tier-1 exposed); proceed to
the four-score offset diagnosis in a separately timestamped v0.4.
