# NarrativeCast — ND4 Task 1 Forecast Protocol

## Status and purpose

Status: PROTOCOL_SPECIFIED; FORECASTING_NOT_AUTHORIZED.

Task 1 specifies the data scope, temporal boundaries,
candidate-set constraints, prominence definition, and
evaluation partitions. It does not train a forecast model.

## Data identity and scope

- Corpus: 11,597,712 POST+COMMENT records, 2014–2020.
- Usable POST registry: 751,607 unique POSTs.
- ND3 v2 baseline: 7,244 POSTs, 24 retrospective clusters.
- Experimental expanded scope: 7,331 POSTs.
- Added POSTs: 44 lexical + 43 semantic; AI-assisted labels,
  not independently human verified.
- Directly linked COMMENT: 117,792.
- Usable directly linked COMMENT: 108,492.
- Usable COMMENT no more than seven days after parent: 108,046.

Corpus-wide relevance is not established. The expanded scope
is a retrospective, provisionally reviewed subset rather
than an exhaustive Net Neutrality corpus.

## Unit of time and forecast intervals

UTC is the sole time zone. Time units are seven-day
Monday–Sunday calendar weeks.

The forecast origin is the end of the completed UTC week.
Conceptual historical interval: (t-7 days, t].
Conceptual future interval: (t, t+7 days].

Implementation with exact timestamps uses equivalent
half-open intervals at Monday 00:00 UTC:
history [origin-7 days, origin);
future [origin, origin+7 days).

This avoids double-counting boundary events. Implementations
must use one consistent boundary convention.

Baseline observation = seven days.
Baseline forecast horizon = seven days.
Observation sensitivity = 14 and 28 days.
Three-day horizon and 14-day horizon are extension scenarios
only and not part of the baseline acceptance decision.

## Data availability and comment policy

At each origin, only observations timestamped before the
future window are eligible for input.

POST provides candidate anchors and topic volume.
COMMENT provides contextual engagement signals, linked to
scope POST through verified direct parent_id matches.

Use publication time of each COMMENT, not its parent POST time.

Do not equate COMMENT count with POST count or count all
COMMENT as narrative-specific evidence.

Potential features, not fitted in Task 1:
- POST volume
- distinct active parent POST count
- sum log1p(thread comment counts)
- distinct user count
- raw COMMENT count as comparator
- per-parent-per-week cap of 20 as sensitivity

COMMENT older than seven days is excluded from the proposed
main feature view, but available for sensitivity analysis.

Cap = 20 is a predeclared exploratory choice, not a fitted
or performance-optimized threshold.

## Candidate-set at time t

C_t must be constructed using material observable strictly
by forecast origin t.

Permitted inputs: historical POST text and other available
historical signals. Any semantic preprocessing or learned
candidate representation must respect the fitting boundary.

New candidates may enter C_t when discovered using past-only
evidence. Candidate identity must be frozen at origin for
the purpose of a single forecast.

The 24 ND3 full-period clusters are retrospective diagnostic
groups only. They are NOT historically available candidate IDs.

Any cluster discovery, naming, alignment, embedding
transformation, threshold selection, or candidate ranking
learned from future observations is prohibited.

At each origin, candidate generation must operate using
past-only fitting or a legitimately pre-fitted artifact
available at that origin. Candidate matching rules must be
specified before examining the corresponding future target.

The actual algorithm and past-only candidate stability
remain to be validated in subsequent implementation work.
Task 1 does not claim this gate has passed.

## Target: future narrative prominence

For each candidate c in C_t:

Let N_future denote the number of ALL valid, in-scope
POST documents published in the future horizon, including
noise, other narratives, and unassigned POSTs.

Let n_c_future denote the number of those future POSTs
assigned to candidate c by a documented, causally valid
assignment rule fixed at origin t.

Prominence(c,t) = n_c_future / N_future, when N_future > 0.

If N_future = 0, the prominence ratio is undefined:
record missing/undefined, do not impute zero and do not
silently discard the forecast origin.

Noise and unassigned documents stay in the denominator.
They are not removed to inflate candidate prominence.

COMMENT is not included in the denominator of this
POST-based primary target. A COMMENT-based target would
require a separate, explicitly declared protocol.

No future-document assignments are created in Task 1.

## Temporal partitions

TRAIN origins: 2014–2017
VALIDATION origins: 2018
TEST origins: 2019–2020

The split is chronological and must not be shuffled.
A forecast origin is assigned to the split containing
its origin timestamp.

Training and tuning must respect temporal availability.
Validation may choose hyperparameters but not test data.
TEST must remain untouched until the method is frozen.

For rolling-origin evaluation, each candidate discovery
or representation fit must use data available at that
origin, not the full TRAIN period for earlier TRAIN origins.

The scope itself was selected retrospectively; this creates
an additional selection-bias/leakage risk that cannot be
removed merely by timestamp-filtering model features.

Before claiming leakage-free forecasting, a past-only scope
and candidate protocol, or an explicit retrospective
conditional-scope estimand with its limitations, is required.

## Gating and interpretation

Technical audit: PASS.
Scope decision: EXPAND_CONDITIONALLY.
Candidate-generation leakage gate: NOT VERIFIED.
Forecasting execution: NO-GO until candidate protocol
is implemented and validated.

Sparse TEST coverage is an explicit risk, especially in 2020.
Do not remove difficult years to improve metrics.
Do not interpret technical PASS as forecasting performance.
