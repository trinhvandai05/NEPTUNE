# NEPTUNE — amendment log (frozen with preregistration v1.2.3)

**Status.** Frozen before any result on MovieLens-25M was seen. No version up to and including v1.2.3
has been executed on real data (only synthetic smoke data). The SHA-256 of this file is recorded in `configs/preregistered_v1_2_3.yaml` and verified by the
config loader, so this log cannot be edited without re-freezing under a new version.

**This repository is NOT a 1:1 implementation of the frozen v1 spec** (`neptune-v1-spec.md`). Every
departure is listed below. Results must be reported as "NEPTUNE H1-v1.2", not "NEPTUNE v1".

## Changes to the estimand or the model

**A1 — H1 moderator: Shannon diversity → Simpson collision concentration (ESTIMAND CHANGE).**
v1 defined H1's moderator as semantic entropy H_sem = −Σ p_c log p_c. v1.2 uses
λ = Σ p_c², estimated without bias by λ̂ = Σ n_c(n_c−1) / (n(n−1)). These are *different
functionals* (Hill numbers of order 1 and 2); they can order users differently. This is not "a
better estimator of the same quantity". H1-v1.2 reads: *the gain of M>1 over M=1 grows as collision
concentration falls* (∂Δ/∂λ̂ < 0). Reason: the plug-in H_sem is capped at log min(n, K) and so partly
measures history length; λ̂ is unbiased at every n ≥ 2. v1 results on H_sem are not comparable.

**M1 — Potential V_θ: 3-layer MLP (spec §5.1) → kernel-attractor sum (MODEL CHANGE).**
V(z; c) = −Σ_j w_j(c) κ_{h_V}(z, u_j(c)). Closed-form ∇V (no double backward inside every RK4
stage), and the G1 assimilation channel becomes inspectable.

**M2 — No title text embedding.** The 1128-tag genome carries the information more densely.

**M3 — Prior particles from encoder images of random items, not semantic centroids.** Prevents an
information path from the H1 stratification variable into the model.

**M4 — H1 slice omits** log Z, OU intensity, ρ(c), time likelihood, compensator, RFF, HNSW, budget
residual, drag estimator, power iteration. The interaction coefficient γ·sg[Z] is γ (Z ≡ 1).

**M5 — Fixed-step RK4 only** (J = 1 under ordinal time); Dormand–Prince returns only as D8.

## Changes to measurement and analysis

**A1.1** τ² of λ̂ by session-block bootstrap; reliability R_j and DEFF_j per length stratum.
**A1.2** Split-half IV over alternating *sessions*; robustness only; first-stage F reported.
**A1.3** Tail-fraction confound from train-only popularity (tail_test and pre-treatment tail_train).
**A2** Per-arm h selected on validation; σ shared unless sharing costs any F01 arm > 1% relative
validation NDCG (neutral shared σ, symmetric cost).
**A3** Condition C5 direction: the H1-true world *with* measurement error yields β_j = β·R_j, so C5
passes when β_j/R_j is negative and stable (CV ≤ 0.75) and the stratum-FE slope is < 0.
**A4** Evaluation cohort: ≥ 20 train positives and ≥ 3 test positives; attrition reported.

**A5 — Seeds and inference (external review of v1.1).**
Seed 0 is the tuning seed: it selects h*(M), σ*(M) and is reported only as descriptive row D0 —
never in C1–C5 (winner's curse). Confirmatory seeds: {1..5} for the primary arms M ∈ {1, 16},
{1, 2} for the other arms (D1 shape only). Every gating interval is two-level:
SE² = SE²_user + Var_s(θ^(s))/S with critical value t_{S−1}; C1 additionally requires every
confirmatory seed's mean Δ > 0. Rationale: a CI over ~10⁴ users can be tiny while the effect flips
sign between trained models. Seed s of the treatment arm is paired with seed s of the control arm as a
**common-random-numbers design**: the same seed gives both arms the same encoder / potential /
satiation initialization, batch order and negative stream, so Δ^(s) is a paired difference. D1 compares
every arm with the control averaged over the *same* seeds as that arm.

**A6 — D17 gate, one threshold.** Final mean pairwise ⟨e_i, e_j⟩ must be ≤ 0.60 for every OPEN-1
pilot run (else ADOPT v1.3) AND for every analysed claim run (else C6 fails → INVALID_PROTOCOL). The
spec names the diagnostic but no number; 0.60 is our choice. v1.2 used 0.60 for an 8-epoch pilot but
0.80 for 20-epoch claim runs — a level that forced v1.3 in the pilot was tolerated in the claim runs.
**What it measures (v1.2.2):** mean cosine detects collapse *toward one direction* (a cone — the OPEN-1
mechanism). It is blind to other collapses: a ±v antipodal manifold has mean cosine ≈ 0 while being
one-dimensional. The check is therefore named `D17_mean_cos_ok`, not "no collapse". Effective rank,
mean |cos| and pairwise-cosine quantiles are logged per epoch and reported by the pilot and the
analysis, but are **not gated**: turning any of them into a gate is an amendment that must be frozen
before the real pilot, never after seeing it.

**A7 — Cold items** are items whose first rating of *any* value by *any* user (full corpus,
before thresholding and cohort sampling) is ≥ T_cut. Negatives are sampled from non-cold items.

**A8 — Robustness time arm** uses Δ = log1p(clip(days, 0, 365)), a log-compressed elapsed time. It
is named LOGDT and must never be reported as raw Δt or as a continuous-time (H2) result.

**A9 — No truncation of the train history.** v1.1–v1.2 trained on the last 1024 train events of a
user, while λ̂ is computed from the whole train history and evaluation rolls the whole history — a
train/eval mismatch concentrated exactly in the longest (most measurable) histories H1 is about. BPTT
(64 events) already bounds graph memory; the cap only saved compute. `train.max_len = 0`. The pilot
report prints the history-length distribution so the compute cost is visible.

**A10 — Code is locked (v1.2.2; extended in v1.2.3).** The preregistration freezes
`implementation.sha256`, a fingerprint of src/neptune/**/*.py, scripts/*.py and pyproject.toml. Claim
runs, the OPEN-1 pilot, the OPEN-1 decision, the σ-policy and h\*(M) freezes, and the analysis refuse
other code — the code that *computes the verdict* is locked, not only the code that trains; every run records the fingerprint and data-manifest hash it ran with
(`provenance.json`); reuse of a run made by other code or on another data artifact is refused; the
analysis refuses to mix runs from two code versions or two data artifacts. Before, the prereg hash
locked the design but not the implementation.

**A11 — Cohort shortfall is fatal (v1.2.2).** If fewer evaluable users exist than `data.n_users`,
`prepare` writes `cohort_shortfall.json` and no manifest (the artifact is unusable). A smaller cohort
must be adopted by an explicit amendment of `data.n_users`, never silently.

**A12 — Artifact integrity (v1.2.3).** Data and covariate manifests record the SHA-256 of every
payload file; the covariate manifest is linked to the exact data manifest it was built from. The
σ policy and h\*(M) are write-once, stamped with prereg + code, and cite the hashes of every validation
file they used; h\*(M) also anchors the data manifest and the covariate manifest, so the H1 moderator
must exist before selection and cannot change afterwards. Before confirmatory runs and before any test
file is unsealed, `verify_frozen_selection` re-checks all of this and **re-derives** the policy and
h\*(M) from the validation runs with the frozen rules. The OPEN-1 verdict is likewise re-derived from
its hashed evidence on every use; the decision file is never trusted on its own.

## Restorations of the spec (bugs in v1.1, not amendments)

- Pruning of decayed atoms now happens at step 1 (before evolve and the heads), as spec §5.5 requires;
  v1.1 pruned only inside assimilate.
- ESS guard is accumulated over every active event; v1.1 applied it only to users active at the last
  column of a BPTT window.
- Runs, selections and reports are namespaced by preregistration hash; reuse requires an identical
  stored config.
- (v1.2.1) Dataset and covariate artifacts carry a code schema and the prereg hash; claim runs and the
  analysis refuse artifacts built by another schema or preregistration (v1.1 and v1.2 share `data_cfg`
  but define cold items differently). Default data directories are namespaced like runs.
- SASRec control: v1.1 trained on non-overlapping chunks, and the position read at evaluation (W−1)
  never received gradient. v1.2 uses half-overlapping windows; every position read at eval is trained.

## OPEN items (must be resolved before any claim-eligible run)

**OPEN-1 — Angular scale vs. bandwidth h are not separately identified.**
With a fixed-h kernel on the sphere and a free encoder, the encoder can shrink the angular spread of
the catalog, which raises the *effective* bandwidth and partly undoes the h grid. Evidence (synthetic
data, real widths d=64, M=4, 6 epochs): mean pairwise cos rose 0.37 → 0.54 at h=0.30 and 0.36 → 0.45
at h=0.50, i.e. faster at smaller h, as the hypothesis predicts. At d=8 it reached 0.95. Synthetic
data says nothing about ML-25M. Consequences if it holds on ML-25M: the h grid is partly vacuous, the
D17 gate may invalidate the claim runs after the compute is spent, and M=1 vs M=16 may compress
differently (a geometry confound in Δ).
Decision rule (revised in v1.2.1, before any pilot was run), **enforced by code**:
the claim-INELIGIBLE pilot runs M ∈ {1, 16} × h ∈ {0.20, 0.50} at σ = 0.30, seed 0, 5 000 users and
the **claim epoch count (20)**. Since v1.2.2 this pilot is a frozen **contract** (`gates.open1` in the
prereg): each pilot config must equal the claim config for its sweep except `data.n_users` = 5000 and
`train.warmup_steps` = 500, checked key by key before the pilot may start and again when deciding.
The pilot is protocol-critical although claim-ineligible: exact artifact match (schema, prereg,
actual cohort size, implementation), frozen code, guarded reuse. `scripts/pilot_report.py` freezes `gates/<prereg sha12>/open1_decision.json`
(write-once, with the SHA-256 of every pilot summary). Verdict ADOPT_V1_3 if any pilot run's final
D17 > 0.60 (the A6 threshold), else KEEP_CURRENT_PREREG. The runner refuses every claim-eligible run
unless that file exists, belongs to this preregistration and frozen implementation, says KEEP, and
all its evidence — each pilot run's summary, config, provenance and data manifest — is byte-identical
to what the decision recorded. The v1.2 "rising over the last 3 epochs" trigger is dropped: it existed only to extrapolate
a pilot shorter than the claim runs. The D17 difference between M = 1 and M = 16 is reported
descriptively (a differential-compression confound would show there).

## Changelog
- 1.2.0 — first external review of v1.1 (namespacing, seed rules, two-level inference, pruning,
  ESS guard, cold items, SASRec windows, LOGDT naming, D17).
- 1.2.1 — second external review of v1.2: artifact provenance, OPEN-1 hard gate, single D17
  threshold with pilot at the claim epoch count, no history truncation, D1 matched seeds, CRN wording.
  "v1.3" remains reserved for the OPEN-1 architectural amendment.
- 1.2.3 — fourth external review of v1.2.2: OPEN-1 verdict re-derived, decision/selection/analysis
  code locked, data + covariate payload hashes, selection anchored and re-derived.
- 1.2.2 — third external review of v1.2.1: canonical OPEN-1 pilot contract, pilot treated as
  protocol-critical, implementation fingerprint lock with provenance per run, fatal cohort
  shortfall, D17 renamed to what it measures plus descriptive spread diagnostics, doc fixes.
