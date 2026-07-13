# Validation seed non-exposure declaration (v0.3-rc5)

## Claim

The validation bank seeds (base `VAL_SEED = 257841043574753`) have **never** been fed
to the estimator in any prior phase -- exploratory design, deterministic rescue
reconstruction, or trial run. No offset, coverage, score, gain, or pass/fail statistic
has been evaluated on any of these seeds prior to the preregistered one-shot validation
run.

There is **no calibration bank** in v0.3-rc5 (the fitted-debiasing fallback was removed;
identity is the only estimator, and any repair moves to a separately timestamped v0.4).

## Machine-checked portion

All prior exploratory and rescue runs used base seed `20260711`. The full enumerable
base-20260711 grid (3 pi levels x {construction_null + 3 depths} x 500 = 6000 distinct
seeds) was regenerated and intersected with the 6000 validation seeds derived here:

- validation ∩ prior(20260711) : 0
- validation seeds unique : 6000 / 6000

Reproducible from `seed_manifest.csv` and `oe_v03_run.py::bankseed`
(`SeedSequence([base, cell_index, replicate])`), and re-run by `--preflight`.

## User-attested portion

The machine check covers the dominant, most-recent exposure (base 20260711). Other
historical debug runs used their own bases and cannot all be enumerated from this
package alone. The undersigned attests that none used base `257841043574753`, and that
this integer was freshly minted for this preregistration and not previously exposed.

Signed (research identity): Takayuki Takagi -- ORCID 0009-0003-5188-2314.
Soli Deo Gloria.
