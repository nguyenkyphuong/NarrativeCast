# NarrativeCast — ND4 Task 1 Scope Decision

## Decision

**EXPAND_CONDITIONALLY** for the data scope.
**NO-GO** for forecast training/execution until
time-respecting candidate generation is validated.

This is a Task 1 scoping decision, not a model result.

## Evidence

### Corpus and registry

- Corpus: 11,597,712 records.
- POST rows: 768,483; COMMENT rows: 10,829,229.
- Registry: 751,607 distinct usable POSTs.
- Difference: 16,876 corpus-only POSTs, all marked
  is_usable_text=False.
- No registry-only POSTs, duplicate POST IDs, or
  timestamp mismatches were found in the audits.
- Expanded scope: 7,331 distinct POSTs in the corpus.

### POST retrieval expansion

ND3 baseline: 7,244 POSTs.

AI-assisted lexical review selected 44 additional POSTs.
AI-assisted semantic review selected 43 additional POSTs.

Combined scenario: 7,331 POSTs.
Active weeks rose from 339 to 341 out of 364.
Weeks with at least five POSTs rose from 239 to 240.
Weeks with at least ten POSTs rose from 181 to 182.

The expansion has minimal temporal coverage gain.
The added 87 POSTs remain provisionally reviewed.
The original 7,244 scope remains the benchmark.

### COMMENT evidence

- 117,792 directly linked COMMENTs.
- 108,492 usable directly linked COMMENTs.
- 108,046 usable COMMENTs within seven days of parent POST.
- 4,573 scope POSTs have usable direct COMMENTs.
- Top 5% of active parent POSTs account for approximately
  60.6% of usable COMMENT volume.
- Capping 20 COMMENTs per parent/week reduces the
  observed top-5% share to about 17.3% in the
  seven-day subset.
- COMMENT activity adds context, not narrative ground truth.

### Temporal split feasibility

Seven-day observation; seven-day horizon:

TRAIN 2014–2017:
207 forecast origins; 165 with >=5 historical POSTs.

VALIDATION 2018:
53 origins; 51 with >=5 historical POSTs.

TEST 2019–2020:
103 origins; 24 with >=5 historical POSTs.

2020 alone has only one of 51 seven-day origins
with >=5 historical POSTs.

The TEST period is substantially sparser than
TRAIN and VALIDATION; it must not be silently excluded.

### Retrospective ND3 narrative clusters

24 ND3 clusters cover all 7,244 baseline POSTs.

Median active clusters per week:
TRAIN = 7; VALIDATION = 9; TEST = 2.

Weeks with at least two clusters each having >=5 POSTs:
TRAIN = 44/208; VALIDATION = 15/53; TEST = 2/103.

These clusters were constructed using full-period data.
They cannot be used as historical narrative identities.

## Final Task 1 scope choice

1. Preserve ND3 v2 7,244 POSTs unchanged.
2. Carry expanded_v1 7,331 POSTs as a provisional
   experimental scope with review provenance.
3. Include direct usable COMMENTs as contextual signals;
   no COMMENT-to-POST count conflation.
4. Preserve all VALID, in-scope POSTs including noise
   and unassigned documents in the target denominator.
5. Use seven-day observation and seven-day horizon;
   14/28-day observation are sensitivity settings.
6. Preserve chronological TRAIN/VALIDATION/TEST
   partitions and report sparse TEST coverage.
7. Exclude the 24 retrospective ND3 clusters from
   historical candidate identity and target generation.

## ND3 implications

- Do NOT overwrite ND3 base scope or cluster outputs.
- A full-period reclustering of 7,331 POSTs is NOT
  accepted as a leakage-free forecasting solution.
- If implementation needs expanded-scope representations,
  rerun only necessary mapping and past-only candidate
  procedures, with timestamp-safe fit boundaries.
- Track the 87 AI-assisted added POSTs separately.
- Human review would strengthen scope validity.

## Risks and unresolved gates

- Retrospective scope selection bias.
- Candidate-set identity and stability at each origin.
- Potential future leakage from semantic prototypes,
  clustering, naming, or feature normalization.
- Sparse TEST data and near-zero candidate competition.
- COMMENT concentration and topic drift.
- AI-assisted review is not independently validated.

## Acceptance statement

Task 1 delivers data audit evidence, versioned scope,
temporal coverage, and a specified forecast protocol.

This does not authorize model training or establish
the feasibility of leakage-free narrative forecasting.

The next implementation stage must satisfy the
candidate-generation leakage gate before forecasting GO.
