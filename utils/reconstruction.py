# Reconstruction pipeline for the QRT 2026 challenge.
#
# Each GROUP trades on one exchange. An allocation is a portfolio of stocks of that exchange,
# so its return on a day is (roughly) a linear mix of the stock returns of that day.
#   1. exposures: learn this mix (one ridge per allocation) on the dated train rows
#   2. dating: an isolated row has a 20-day history -> find the calendar day where the
#      real stock returns explain this history best (all the groups of the TS vote together)
#   3. refresh: the test is years after the train, so the exposures are re-estimated from
#      the histories of the dated test rows (shrunk toward the train exposures), then we re-date
#   4. prediction: target sign = sign(exposures . stock returns of the next session)
#
# Not a forecast: it uses the real returns of the target day.

import os
import numpy as np
import pandas as pd
import exchange_calendars as xcals

HIST_COLS = [f"RET_{i}" for i in range(20, 0, -1)]   # oldest -> newest

MARKETS = {
    1: {"exchange": "XTKS", "index": "NIKKEI225", "stocks": "market_data/japan_returns.csv"},
    2: {"exchange": "XHKG", "index": "HSCEI", "stocks": "market_data/hongkong_returns.csv"},
    3: {"exchange": "XKRX", "index": "KOSPI", "stocks": "market_data/korea_returns.csv"},
    4: {"exchange": "XSHG", "index": "SHANGHAI", "stocks": "market_data/china_returns.csv"},
}


# ---------------------------------------------------------------- market data

class Market:
    # stock returns of one exchange on its own calendar, standardized

    def __init__(self, group, index_returns, first_day, std_start, std_end, min_coverage=0.8):
        cfg = MARKETS[group]
        self.group = group
        self.sessions = pd.DatetimeIndex(xcals.get_calendar(cfg["exchange"], start="2004-01-01").sessions)

        if os.path.exists(cfg["stocks"]):
            stocks = pd.read_csv(cfg["stocks"], index_col="Date", parse_dates=True)
        else:
            stocks = pd.DataFrame(index=self.sessions)
            print(f"group {group}: no stock file ({cfg['stocks']}), index only")

        # keep stocks that exist on most of the useful period (train start -> today)
        useful = stocks.loc[stocks.index >= first_day]
        coverage = useful.notna().mean() if len(useful) else pd.Series(dtype=float)
        keep = list(coverage[coverage >= min_coverage].index)

        R = stocks[keep].clip(-0.3, 0.3).reindex(self.sessions).fillna(0)
        R["INDEX_" + cfg["index"]] = index_returns[cfg["index"]].reindex(self.sessions).fillna(0)
        std = R.loc[(R.index >= std_start) & (R.index < std_end)].std().replace(0, 1)
        self.R = R / std
        self.M = self.R.values
        self.columns = list(self.R.columns)
        self.pos = {d: i for i, d in enumerate(self.sessions)}
        print(f"group {group} ({cfg['exchange']}): {len(keep)} stocks + index")

    def next_session(self, dates):
        return self.sessions[self.sessions.searchsorted(pd.DatetimeIndex(dates), side="right")]

    def rows_of(self, dates):
        return self.M[[self.pos[d] for d in dates]]

    def window(self, date):
        # the 20 sessions ending at `date` (same order as HIST_COLS)
        i = self.pos[date]
        return self.M[i - 19:i + 1]


# ---------------------------------------------------------------- 1. exposures

def ridge(A, b, lam, prior=None):
    k = A.shape[1]
    if prior is None:
        return np.linalg.solve(A.T @ A + lam * np.eye(k), A.T @ b)
    return np.linalg.solve(A.T @ A + lam * np.eye(k), A.T @ b + lam * prior)


def fit_exposures(rows, market, lam, min_rows=100):
    # rows: dated rows of one group with "next_date" and "target"
    # returns {allocation: beta} and {allocation: residual std}
    betas, noise = {}, {}
    for alloc, sub in rows.groupby("ALLOCATION"):
        if len(sub) < min_rows:
            continue
        A = market.rows_of(sub["next_date"])
        b = sub["target"].values
        beta = ridge(A, b, lam)
        betas[alloc] = beta
        noise[alloc] = max(np.std(b - A @ beta), 1e-5)
    return betas, noise


def choose_lambda(rows, market, val_start, val_end, grid=(100, 1000, 10000, 100000)):
    # pick the ridge penalty on a validation year, before it
    fit = rows[rows["date"] < val_start]
    val = rows[(rows["date"] >= val_start) & (rows["date"] < val_end)]
    best, best_err = grid[0], np.inf
    for lam in grid:
        betas, _ = fit_exposures(fit, market, lam)
        err, n = 0.0, 0
        for alloc, sub in val.groupby("ALLOCATION"):
            if alloc in betas:
                pred = market.rows_of(sub["next_date"]) @ betas[alloc]
                err += ((sub["target"].values - pred) ** 2).sum()
                n += len(sub)
        if n and err / n < best_err:
            best, best_err = lam, err / n
    return best


# ---------------------------------------------------------------- 2. dating

class Dater:
    # scores every possible end date of the 20-day histories, for all groups at once

    def __init__(self, markets, betas, noise):
        self.markets = markets
        self.betas = betas      # {group: {alloc: beta}}
        self.noise = noise      # {group: {alloc: std}}
        # predicted return of each allocation on every session of its exchange
        self.pred = {g: {a: markets[g].M @ b for a, b in betas[g].items()} for g in betas}

    def group_scores(self, rows, g):
        # log-likelihood gain of every end date, for the rows of one group
        sessions = self.markets[g].sessions
        total = np.zeros(len(sessions) - 19)
        used = 0
        for alloc, hist in zip(rows["ALLOCATION"], rows[HIST_COLS].values):
            if alloc not in self.pred[g]:
                continue
            p = self.pred[g][alloc]
            h = np.nan_to_num(hist.astype(float))
            gain = 2 * np.convolve(p, h[::-1], mode="valid") - np.convolve(p ** 2, np.ones(20), mode="valid")
            total += gain / (2 * self.noise[g][alloc] ** 2)
            used += 1
        if used == 0:
            return None
        return pd.Series(total, index=sessions[19:])

    def scores(self, rows, allowed=None):
        # all groups of one TS: the date must be a session of every group present
        total = None
        for g, sub in rows.groupby("GROUP"):
            s = self.group_scores(sub, g)
            if s is None:
                continue
            total = s if total is None else (total + s).dropna()
        if total is None:
            return None
        if allowed is not None:
            total = total[total.index.isin(allowed)]
        return total


def score_all(X_rows, dater, allowed, only_ts=None):
    # score series (best first) of every TS
    found = {}
    for ts, rows in X_rows.groupby("TS"):
        if only_ts is not None and ts not in only_ts:
            continue
        s = dater.scores(rows, allowed)
        if s is not None and len(s) > 1:
            found[ts] = s.sort_values(ascending=False)
    return found


def assign_dates(found, min_gap_days=25):
    # most confident TS first; two TS can't share overlapping 20-day windows
    # (seen in the data: test rows never overlap each other)
    order = sorted(found, key=lambda ts: found[ts].iloc[0] - found[ts].iloc[1], reverse=True)
    taken = []
    out = []
    for ts in order:
        s = found[ts]
        best = s.index[0]
        if min_gap_days > 0:
            for d in s.index[:300]:
                if all(abs((d - t).days) > min_gap_days for t in taken):
                    best = d
                    break
        taken.append(best)
        out.append({"TS": ts, "date": best, "score": s[best], "margin": s[best] - s.drop(best).iloc[0]})
    return pd.DataFrame(out)


def date_all(X_rows, dater, allowed, min_gap_days=25):
    return assign_dates(score_all(X_rows, dater, allowed), min_gap_days)


def date_with_refresh(X_rows, markets, betas0, noise, lams, allowed, rounds=2, min_gap_days=25, prior_strength=1.0,
                      final_prior_strength=0.5, final_trust=0.8, verbose=True):
    # round 1 with the train exposures, then (optional) rounds where the exposures are refreshed
    # from the histories of the dated rows. Cross-fitting: the TS of one half are dated with
    # exposures refreshed on the other half only, so a row never confirms its own date.
    # Simulation: no gap between train and test -> round 2 = round 1 (85% of dates right);
    # 2.5 years of gap -> 71% at round 1, 83% at round 2. The test is far from the train -> rounds=2.
    # The final refresh (used for the predictions) is what helps the accuracy.
    dated = date_all(X_rows, Dater(markets, betas0, noise), allowed, min_gap_days)
    history = [dated]
    if verbose:
        print(f"round 1: median margin {dated['margin'].median():.1f}")

    all_ts = sorted(X_rows["TS"].unique())
    halves = [set(all_ts[0::2]), set(all_ts[1::2])]
    for r in range(2, rounds + 1):
        found = {}
        for k in [0, 1]:
            other = dated[~dated["TS"].isin(halves[k])]
            betas_k = refresh_exposures(X_rows, other, markets, betas0, lams, prior_strength=prior_strength)
            found.update(score_all(X_rows, Dater(markets, betas_k, noise), allowed, only_ts=halves[k]))
        dated = assign_dates(found, min_gap_days)
        history.append(dated)
        if verbose:
            print(f"round {r}: median margin {dated['margin'].median():.1f}")

    # final exposures for the prediction: refreshed on all the dated rows
    betas = refresh_exposures(X_rows, dated, markets, betas0, lams,
                              prior_strength=final_prior_strength, trust=final_trust)
    return dated, betas, history


# ---------------------------------------------------------------- 3. refresh of the exposures

def refresh_exposures(X_rows, dated, markets, prior_betas, lam, min_dates=5, trust=0.5, prior_strength=1.0):
    # re-estimate each allocation's exposures from the 20-day histories of the dated rows,
    # shrunk toward the train exposures. Only the most confident dates are used.
    good = dated[dated["margin"] >= dated["margin"].quantile(1 - trust)]
    date_of = dict(zip(good["TS"], good["date"]))
    rows = X_rows[X_rows["TS"].isin(date_of)]

    new = {}
    for g in prior_betas:
        new[g] = dict(prior_betas[g])
        sub_g = rows[rows["GROUP"] == g]
        m = markets[g]
        lam_g = (lam[g] if isinstance(lam, dict) else lam) * prior_strength
        for alloc, sub in sub_g.groupby("ALLOCATION"):
            if alloc not in prior_betas[g]:
                continue
            A, b = [], []
            for ts, hist in zip(sub["TS"], sub[HIST_COLS].values):
                d = date_of[ts]
                if d in m.pos:
                    A.append(m.window(d))
                    b.append(np.nan_to_num(hist.astype(float)))
            if len(A) < min_dates:
                continue
            new[g][alloc] = ridge(np.vstack(A), np.concatenate(b), lam_g, prior=prior_betas[g][alloc])
    return new


# ---------------------------------------------------------------- 4. prediction

def predict(X_rows, dated, markets, betas):
    # predicted target of every row = exposures . stock returns of the next session
    date_of = dict(zip(dated["TS"], dated["date"]))
    pred = pd.Series(np.nan, index=X_rows.index)
    for g, sub_g in X_rows.groupby("GROUP"):
        m = markets[g]
        for idx, alloc, ts in zip(sub_g.index, sub_g["ALLOCATION"], sub_g["TS"]):
            if ts not in date_of or alloc not in betas[g]:
                continue
            d = date_of[ts]
            nxt = m.next_session([d])[0]
            if nxt in m.pos:
                pred[idx] = m.M[m.pos[nxt]] @ betas[g][alloc]
    return pred


def allowed_days(markets, excluded_dates, start="2004-03-01", end=None, gap_days=25):
    # candidate dates: sessions of at least one exchange, far from the dates we must avoid
    days = sorted(set().union(*[set(m.sessions) for m in markets.values()]))
    days = pd.DatetimeIndex(days)
    days = days[days >= start]
    if end is not None:
        days = days[days <= end]
    if len(excluded_dates):
        ex = pd.DatetimeIndex(sorted(excluded_dates)).values
        pos = np.searchsorted(ex, days.values)
        near = np.zeros(len(days), dtype=bool)
        for shift in [-1, 0]:
            j = np.clip(pos + shift, 0, len(ex) - 1)
            near |= np.abs((days.values - ex[j]).astype("timedelta64[D]").astype(int)) <= gap_days
        days = days[~near]
    return days


# ---------------------------------------------------------------- 5. local exposures (time-varying)

def predict_local(X_rows, dated, markets, betas, lams, window_days=180, trust=0.8, prior_strength=0.5):
    # For each dated TS, re-estimate the exposures with the histories of the dated TS that are
    # less than `window_days` away (shrunk toward `betas`), then predict the next session.
    # Simulation 2011-2015: +2 points vs one exposure for the whole period (best around 90-180 days).
    date_of = dict(zip(dated["TS"], dated["date"]))
    good = set(dated[dated["margin"] >= dated["margin"].quantile(1 - trust)]["TS"])
    pred = pd.Series(np.nan, index=X_rows.index)

    for g, m in markets.items():
        rows_g = X_rows[X_rows["GROUP"] == g]
        # the gram matrix of a window is the same for every allocation of the group
        G, cross = {}, {}
        for ts, sub in rows_g.groupby("TS"):
            if ts not in good or date_of[ts] not in m.pos:
                continue
            W = m.window(date_of[ts])
            G[ts] = W.T @ W
            H = np.nan_to_num(sub[HIST_COLS].values.astype(float))
            for alloc, c in zip(sub["ALLOCATION"], H @ W):
                cross[(alloc, ts)] = c

        lam = (lams[g] if isinstance(lams, dict) else lams) * prior_strength
        k = m.M.shape[1]
        for ts, sub in rows_g.groupby("TS"):
            d = date_of.get(ts)
            if d is None:
                continue
            nxt = m.next_session([d])[0]
            if nxt not in m.pos:
                continue
            near = [s for s in G if abs((date_of[s] - d).days) <= window_days]
            A = lam * np.eye(k)
            for s in near:
                A += G[s]
            B, idx = [], []
            for row_id, alloc in zip(sub.index, sub["ALLOCATION"]):
                if alloc not in betas[g]:
                    continue
                b = lam * betas[g][alloc]
                for s in near:
                    if (alloc, s) in cross:
                        b = b + cross[(alloc, s)]
                B.append(b)
                idx.append(row_id)
            if B:
                local_betas = np.linalg.solve(A, np.array(B).T)
                pred[idx] = m.M[m.pos[nxt]] @ local_betas
    return pred
