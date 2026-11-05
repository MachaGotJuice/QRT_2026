# QRT 2026 Data Challenge: predicting the sign of asset allocation returns

A personal project on the [QRT 2026 challenge](https://challengedata.ens.fr/participants/challenges/167/)
of the ENS Challenge Data platform.

## Results

Accuracy on the public leaderboard:

| Approach | Score |
|---|---|
| Official benchmark | 0.5079 |
| Always predict "up" | ~0.52 |
| LightGBM on the 20-day history | ~0.52 (cross-validation) |
| Reconstruction v2 (one set of exposures) | **0.6117** |
| Reconstruction v3 (local exposures + direct reading) | **0.6724** |

> **Important.** The reconstruction is **not** a forecasting model. It uses the real stock returns of the
> day of the target, taken from Yahoo Finance (external data). The score shows that the dataset could be
> reverse engineered. It says nothing about predicting the market.

## The task

There are 278 anonymized asset allocations (portfolios). For each allocation and each day we get:

- the last 20 daily returns `RET_1 ... RET_20` (`RET_1` is the most recent),
- the last 20 signed volumes,
- a `GROUP` (1 to 4) and an anonymized date `DATE_xxxx`.

The goal is to predict the **sign of the next-day return**. The metric is the accuracy.

## Step 1: a classic machine learning model (about 52%)

Features: means of past returns, volatility, momentum, volumes, rank of the allocation in its group.

| Model | Loss | Accuracy |
|---|---|---|
| Ridge | squared error + L2 penalty | ~52% |
| LightGBM, tuned with Optuna | binary log-loss | 52.0% |
| MLP with embeddings for `ALLOCATION` and `GROUP` | binary cross-entropy | 51.9% |

"Always predict up" already gives 52.2%. The best single feature (`RET_1`) has an AUC of 0.526.
**Conclusion: the 20-day history contains almost no signal about tomorrow.** This is expected: daily
returns are very hard to predict.

## Step 2: look at the data, not at the model

Four observations changed the approach:

1. **The rows are sliding windows.** For the same allocation, `RET_2..RET_20` of one day are almost equal to
   `RET_1..RET_19` of the previous day (median correlation 0.992). So we can find which row comes after
   which row, and rebuild the order of the days.
2. **The date labels are shuffled.** `DATE_0001`, `DATE_0002`... are not in time order. A "time" split on
   these labels is in fact a random split.
3. **There is a leak in a naive validation.** The sign of the target equals the sign of `RET_1` of the next
   day in about 90% of the rows. If neighbor days end up in train and validation, the model sees the answer.
4. **Each `GROUP` is a stock exchange.** A group has no rows on the days its exchange is closed (holidays).

## Step 3: put the dates back on the real calendar

`calendar_dating.ipynb`, `utils/calendar_dating.py`

- Link each row to the next one (correlation of the overlapping returns), then chain the days.
- The missing days of each group form a pattern of holidays. We slide this pattern over real exchange
  calendars (with an FFT) and find the exchanges: **Tokyo, Hong Kong, Korea, Shanghai**.
- Align the order of the days on the calendar (dynamic programming).

Result: 2,342 of the 2,522 train dates placed on the calendar, from 2006 to 2015.

## Step 4: the allocations are baskets of stocks

`ceiling_check.ipynb`

Once a row has a date, we can compare its target with the real stock returns of the next session.
Each allocation behaves like a **weighted mix of stocks of its exchange**:

```
target(day d)  ≈  β · stock returns(next session after d)
```

The weights `β` are the **exposures** of the allocation. They change slowly over time, so they must be
re-estimated close to the date.

## Step 5: reconstruct the test set

`pipeline.ipynb`, `utils/reconstruction.py`

1. **Exposures.** One **ridge regression** per allocation (130 to 450 stocks per exchange).
   Loss: squared error + L2 penalty, `‖y − Aβ‖² + λ‖β‖²`. The penalty `λ` is chosen on a validation year.
   Ridge is a good fit here: the relation is linear, there are many correlated stocks and the data is noisy.
2. **Find the date of each test day.** We only know its 20-day history. For each possible calendar day, we
   compute how well the predicted returns explain this history (a Gaussian log-likelihood, computed for all
   days at once with convolutions). All allocations and groups of the same day vote together.
3. **Refresh the exposures.** The test is years after the train, so we re-estimate the exposures on the dated
   test rows, pulled toward the train exposures (ridge with a prior). We split the test in two halves, so a
   row never confirms its own date (cross-fitting).
4. **Predict.** `sign(β · stock returns of the next session)`. In v3, the exposures are re-estimated for each
   date with the dates less than 180 days away (`predict_local`).
5. **Direct reading (v3).** When two test days are less than 20 sessions apart, the history of the later one
   contains the target of the earlier one: we read it instead of predicting it.

## How we validated

We cannot check on the test set (no labels), so we **simulated the test on the train set**: hide about 30
dates of 2013-2015, spaced like the test dates, learn the exposures before 2013, then run the full pipeline.

| | true dates | estimated dates |
|---|---|---|
| train exposures | 60.0% | 58.5% |
| refreshed exposures | 64.6% | 62.1% |
| local exposures (180 days) | 64.8% | 62.8% |

The simulation matched the leaderboard well (v2: 61.9% simulated, 61.2% public).

**Tried without gain:** a second model to correct the predictions, smooth time weights, PCA factors instead
of all the stocks, grouping similar allocations.

## Limits

- **External data.** Prices come from Yahoo Finance, only for the current members of the indexes. Stocks
  that left the index since 2007 are missing (survivorship bias).
- **Wrong dates.** About 1 test date in 7 is dated wrong, and then the prediction is a coin flip.
- **Not a forecast.** See the note at the top.

## What I learned

- Look at the structure of the data before tuning models.
- A validation that does not copy the real test conditions can lie (shuffled dates, overlapping windows).
- A simple model that matches the problem (a linear mix of stocks, so a ridge) beats a complex one.

## Repository

```
main.ipynb                  baseline: features, LightGBM + Optuna
MLP/MLP_embeddings.ipynb    baseline: MLP with embeddings
calendar_dating.ipynb       put the train dates on the calendar
ceiling_check.ipynb         how well stock returns explain the target when the date is known
stocks_check.ipynb          checks on the stock data
japan_stocks_check.ipynb
pipeline.ipynb              simulation, then reconstruction of the test set (v2)
utils/
  calendar_dating.py        links, chains, holidays, exchange search, alignment
  reconstruction.py         ridge exposures, dating, refresh, predict_local
  data_handling.py, k_folds.py
calendar/                   dates found for the train and the test, direct links
```

The challenge files (`data/`) and the Yahoo Finance prices (`market_data/`) are not in the repo.

## Credits

- This project was built with the help of **Claude** :)
