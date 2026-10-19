# E5 human judging

Run commands from the repository root, with the dependencies in `environment.yml`
installed. The judging workflow uses `openpyxl==3.1.5`, Pillow and matplotlib;
scoring does not need a GPU.

## Existing Chinese packages

Continue using the packages already distributed. Keep their original `_key/key.csv`:
image IDs must match the exact package used by the judges. The scorer accepts the
original `初判` / `复核` tabs and `通过` / `失败` / `无法判断` verdicts, including
Chinese verdicts in the discussion CSV. New packages use English names and verdicts.
Do not regenerate packages during an ongoing judging round.

## Create a new round

```bash
python -m src.e5_judging pack --images outputs/e5/parti
```

The default destination is `outputs/e5_judging/`. It must not already exist;
choose a new `--out` directory for a new round. The script never removes an
existing destination. Send each judge only their `E5_judge_A/` (or B/C/D) folder,
including its workbook and `images/` directory. Keep `_key/` private.

The workbook contains Guide, Primary and Recheck tabs. Judges fill the yellow
Verdict and Notes cells. Valid verdicts are `pass`, `fail`, and `unsure`. Image
IDs link to the 512 x 512 images. The key and assignment seed retain the original
sampling scheme; English labels do not change the random image assignments.

## Score returned workbooks

Collect the returned `.xlsx` files directly inside one directory (not in nested
subdirectories). Keep one copy of each workbook; Excel lock files are ignored,
but duplicate judgments and unknown verdicts are errors.

```bash
python -m src.e5_judging score \
  --key outputs/e5_judging/_key/key.csv \
  --sheets outputs/e5_returned \
  --images outputs/e5/parti \
  --packages outputs/e5_judging
```

For previously distributed packages, replace `--key` and `--packages` with their
existing paths. `--images` and `--packages` are optional. Automatic scores are read
only from the configuration recorded in the key and must cover every expected
prompt at seed 0, with both CLIP Score and ImageReward. `--prompts` can supply a
custom frozen prompt file for either command; use the same file for both.

Results default to `outputs/e5_results/`:

- `e5_failure.csv`: failure counts, rates, Wilson 95% intervals, unsure and missing
  primary counts, optional automatic scores, and final/provisional status.
- `e5_summary.txt`: missing primary/recheck judgments, agreement, Cohen's kappa,
  and unresolved disagreements. Kappa is undefined when expected agreement is 1.
- `disagreements.csv`: completed primary/recheck pairs that disagree, with a
  `final` column for the discussion decision. Missing judgments are counted in
  the summary and are not treated as disagreements.
- `e5_failure_rate.png/pdf`: failure rates and Wilson intervals, with status in
  the title.
- `e5_failure_cases.png/pdf`: available with `--packages` when at least one prompt
  has both pass and fail outcomes across methods. Up to two prompts per challenge
  are sampled with a fixed seed, and the figure identifies them as selected cases.
  Obsolete case figures are removed when a subsequent run has no such examples
  or omits `--packages`.

Failure rate is `fail / (fail + pass)`; unsure and unfilled judgments do not enter
that denominator. Results remain `PROVISIONAL` until every primary/recheck verdict
is present and every disagreement has a final decision. Agreement and kappa always
use the independent judgments before discussion.

## Apply discussion decisions

Fill `final` in `disagreements.csv` with `pass`, `fail`, or `unsure` (the original
Chinese equivalents also work), then explicitly supply the file when rerunning:

```bash
python -m src.e5_judging score \
  --key outputs/e5_judging/_key/key.csv \
  --sheets outputs/e5_returned \
  --resolved outputs/e5_results/disagreements.csv \
  --images outputs/e5/parti \
  --packages outputs/e5_judging
```

The scorer reads the discussion file before rewriting it. It refuses to discard
filled final decisions when `--resolved` is omitted, or to apply decisions whose
primary/recheck verdicts have changed since the discussion file was generated.
Only completed disagreements can receive a final override.

## Regression checks

```bash
python -m unittest discover -s tests -v
```

Checks cover English/Chinese compatibility, duplicate and invalid input, protected
output directories, discussion decisions, missing reviews, undefined kappa,
configuration-specific automatic metrics, and a single-row case figure.
