# Helpers to put the anonymized dates (DATE_xxxx) of the QRT 2026 challenge back on the real calendar.
#
# Idea in 4 steps:
#   1. rows of the same allocation that follow each other share 19 returns
#      (RET_2..RET_20 of day t+1 = RET_1..RET_19 of day t)  -> links between dates
#   2. each GROUP follows its own exchange calendar, so we chain the dates group by group
#      and merge the 4 chains into one global order
#   3. a group is missing on the days its exchange is closed -> holiday pattern
#   4. we slide this holiday pattern over real exchange calendars to find the dates
#
# pip install exchange_calendars scipy

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
import exchange_calendars as xcals

RET_COLS = [f"RET_{i}" for i in range(1, 21)]
OLD_TO_NEW = [f"RET_{i}" for i in range(20, 0, -1)]   # oldest return first


# ---------------------------------------------------------------- step 1 : links

def normalize_rows(M):
    M = M - M.mean(axis=1, keepdims=True)
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def find_next_row(X, shift=1):
    # for each row, find the row of the same allocation that comes `shift` days later
    results = []
    for alloc in X["ALLOCATION"].unique():
        sub = X[X["ALLOCATION"] == alloc]
        R = sub[RET_COLS].fillna(0).values

        later = normalize_rows(R[:, shift:])
        earlier = normalize_rows(R[:, :20 - shift])

        corr = later @ earlier.T          # corr[s, p] : how well row s follows row p
        np.fill_diagonal(corr, -1)
        best = corr.argmax(axis=0)

        df = pd.DataFrame()
        df["TS"] = sub["TS"].values
        df["ALLOCATION"] = alloc
        df["GROUP"] = sub["GROUP"].values
        df["TS_next"] = sub["TS"].values[best]
        df["corr"] = corr[best, np.arange(len(sub))]
        results.append(df)
    return pd.concat(results, ignore_index=True)


def vote_next_date(links, X, group, min_support=0.5, min_corr=0.95):
    # for one group: the next date of each date = the one most allocations agree on
    lg = links[links["GROUP"] == group]
    alloc_sets = X[X["GROUP"] == group].groupby("TS")["ALLOCATION"].agg(set)

    votes = lg.groupby(["TS", "TS_next"]).agg(count=("corr", "size"), corr=("corr", "median")).reset_index()
    votes = votes.sort_values("count", ascending=False).drop_duplicates("TS")

    common = []
    for a, b in zip(votes["TS"], votes["TS_next"]):
        common.append(len(alloc_sets[a] & alloc_sets.get(b, set())))
    votes["support"] = votes["count"] / np.maximum(common, 1)

    good = votes[(votes["support"] >= min_support) & (votes["corr"] >= min_corr)]
    good = good.drop_duplicates("TS_next")     # a date can only have one previous date
    return dict(zip(good["TS"], good["TS_next"]))


def build_chains(next_of, all_dates):
    has_prev = set(next_of.values())
    chains = []
    for d in all_dates:
        if d in has_prev:
            continue
        chain = [d]
        while chain[-1] in next_of and next_of[chain[-1]] not in chain:
            chain.append(next_of[chain[-1]])
        chains.append(chain)
    chains.sort(key=len, reverse=True)
    return chains


# ---------------------------------------------------------------- step 2 : global order

def global_order(group_chains, all_dates):
    # merge the chains of the 4 groups: every consecutive pair in a group chain is an edge
    succ = {d: set() for d in all_dates}
    pred = {d: set() for d in all_dates}
    for g in group_chains:
        for chain in group_chains[g]:
            for a, b in zip(chain[:-1], chain[1:]):
                succ[a].add(b)
                pred[b].add(a)

    # connected pieces
    piece = {}
    pieces = []
    for d in all_dates:
        if d in piece:
            continue
        todo = [d]
        piece[d] = len(pieces)
        members = []
        while todo:
            x = todo.pop()
            members.append(x)
            for y in succ[x] | pred[x]:
                if y not in piece:
                    piece[y] = len(pieces)
                    todo.append(y)
        pieces.append(members)

    # topological sort inside each piece
    orders = []
    for members in pieces:
        members = set(members)
        indeg = {d: len(pred[d] & members) for d in members}
        ready = sorted([d for d in members if indeg[d] == 0])
        order = []
        while ready:
            d = ready.pop()
            order.append(d)
            for e in succ[d]:
                indeg[e] -= 1
                if indeg[e] == 0:
                    ready.append(e)
        orders.append(order)
    orders.sort(key=len, reverse=True)
    return orders


# ---------------------------------------------------------------- step 3 : holiday pattern

def presence_matrix(X, order, groups=(1, 2, 3, 4), max_hole=7):
    # P[i, g] = 1 if group g has rows on date i
    # M[i, g] = 0 on long absences (group not in the data yet), we only trust short holes = holidays
    counts = X.groupby(["TS", "GROUP"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=list(groups), fill_value=0)
    P = (counts.loc[order].values > 0).astype(float)
    M = np.ones_like(P)
    n = len(P)
    for g in range(P.shape[1]):
        i = 0
        while i < n:
            if P[i, g] == 0:
                j = i
                while j < n and P[j, g] == 0:
                    j += 1
                if j - i > max_hole:
                    M[i:j, g] = 0
                i = j
            else:
                i += 1
    return P, M


def load_sessions(codes, start="2004-01-01", end=None):
    if end is None:
        end = pd.Timestamp.today().normalize()
    sessions = {}
    for code in codes:
        try:
            sessions[code] = set(xcals.get_calendar(code, start=start, end=end).sessions)
        except Exception as e:
            print("could not load", code, ":", e)
    return sessions


def union_calendar(sessions, codes):
    # all days where at least one of the exchanges is open, and who is open that day
    days = sorted(set().union(*[sessions[c] for c in codes]))
    Q = np.array([[d in sessions[c] for c in codes] for d in days], dtype=float)
    return pd.DatetimeIndex(days), Q


def mismatch_by_offset(P, M, Q):
    # number of disagreements between the data pattern and the calendar, for every start position
    n = len(P)
    if len(Q) < n:
        return None
    total = np.zeros(len(Q) - n + 1)
    for g in range(P.shape[1]):
        m, p, q = M[:, g], P[:, g], Q[:, g]
        # sum_i m_i (p_i - q_{o+i})^2  =  sum m p + sum m q - 2 sum m p q   (p, q are 0/1)
        total += (m * p).sum()
        total += fftconvolve(q, m[::-1], mode="valid")
        total -= 2 * fftconvolve(q, (m * p)[::-1], mode="valid")
    return np.round(total)


def longest_run(mask):
    # start and end of the longest stretch of True values
    best = (0, 0)
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            if j - i > best[1] - best[0]:
                best = (i, j)
            i = j
        else:
            i += 1
    return best


def find_exchanges(P, M, sessions, verbose=True):
    # Which exchange calendar does each group follow?
    # We look at the groups two by two: on the stretch where both groups are in the data,
    # the days where at least one of them trades must be exactly the union of their two
    # exchange calendars. We score every pair of exchanges, then pick the best combination.
    codes = list(sessions)
    n_groups = P.shape[1]
    pair_table = {}

    for g in range(n_groups):
        for h in range(g + 1, n_groups):
            a, b = longest_run((M[:, g] == 1) & (M[:, h] == 1))
            sub_P = P[a:b][:, [g, h]]
            keep = sub_P.sum(axis=1) > 0           # days where at least one of the two trades
            sub_P = sub_P[keep]
            sub_M = np.ones_like(sub_P)
            table = np.full((len(codes), len(codes)), np.inf)
            if len(sub_P) < 50:
                pair_table[(g, h)] = np.zeros_like(table)
                continue
            for i, ca in enumerate(codes):
                for j, cb in enumerate(codes):
                    U, Q = union_calendar(sessions, [ca, cb])
                    mm = mismatch_by_offset(sub_P, sub_M, Q)
                    if mm is not None:
                        table[i, j] = mm.min() / len(sub_P)   # share of wrong cells
            pair_table[(g, h)] = table
            if verbose:
                i, j = np.unravel_index(np.argmin(table), table.shape)
                print(f"groups {g + 1}&{h + 1} ({len(sub_P)} days): best pair {codes[i]} / {codes[j]}")

    # total score of every combination (one exchange per group)
    total = np.zeros([len(codes)] * n_groups)
    for (g, h), table in pair_table.items():
        shape = [1] * n_groups
        shape[g] = len(codes)
        shape[h] = len(codes)
        if g < h:
            total = total + table.reshape(shape)
    best = np.unravel_index(np.argmin(total), total.shape)
    best = [codes[i] for i in best]

    U, Q = union_calendar(sessions, best)
    best_score = mismatch_by_offset(P, M, Q).min()
    if verbose:
        print("chosen:", best, "->", int(best_score), "mismatches on the full pattern")
    return best, best_score


# ---------------------------------------------------------------- step 4 : exact alignment

def dp_align(P, M, Q, k, band=40, skip_cost=2.0, extra_cost=3.0):
    # dynamic programming: match row i to calendar day j, j always increasing.
    # skipping calendar days (days missing in the data) costs skip_cost each,
    # leaving a row without date (day not in the calendar) costs extra_cost.
    n = len(P)
    width = 2 * band + 1
    big = 1e18
    D = np.full((n, width), big)
    back = np.zeros((n, width), dtype=int)

    def cal_index(i):
        return np.arange(k + i - band, k + i + band + 1)

    def match_cost(i):
        j = cal_index(i)
        c = np.full(width, big)
        ok = (j >= 0) & (j < len(Q))
        c[ok] = (np.abs(P[i] - Q[j[ok]]) * M[i]).sum(axis=1)
        return c

    D[0] = match_cost(0)
    for i in range(1, n):
        c = match_cost(i)
        j_prev = cal_index(i - 1)
        j_cur = cal_index(i)
        for t in range(width):
            # previous rows matched to a calendar day before j_cur[t]
            ok = j_prev < j_cur[t]
            cand = D[i - 1][ok] + skip_cost * (j_cur[t] - j_prev[ok] - 1)
            b = np.argmin(cand)
            D[i, t] = cand[b] + c[t]
            back[i, t] = np.where(ok)[0][b]
            # or: row i gets no date, the calendar pointer stays where row i-1 was
            if t + 1 < width and D[i - 1, t + 1] + extra_cost < D[i, t]:
                D[i, t] = D[i - 1, t + 1] + extra_cost
                back[i, t] = -(t + 1) - 1

    t = int(np.argmin(D[-1]))
    cal = [None] * n
    for i in range(n - 1, -1, -1):
        b = back[i, t]
        if b < 0:
            cal[i] = None
            t = -b - 1
        else:
            cal[i] = k + i - band + t
            t = b
    return cal, D[-1].min()


def date_piece(X, order, exchanges, sessions, used_days=None, band=40):
    # finds the dates of one ordered piece, given the exchange of each group
    P, M = presence_matrix(X, order)
    U, Q = union_calendar(sessions, exchanges)
    mm = mismatch_by_offset(P, M, Q)
    if mm is None:
        return None
    if used_days is not None and len(used_days) > 0:
        # do not put this piece on days already taken by another piece
        taken = np.array([d in used_days for d in U], dtype=float)
        overlap = fftconvolve(taken, np.ones(len(P)), mode="valid")
        mm = mm + 10 * np.round(overlap)
    k = int(np.argmin(mm))
    second = np.min(np.concatenate([mm[:max(k - 5, 0)], mm[k + 5:]])) if len(mm) > 10 else np.inf

    cal, cost = dp_align(P, M, Q, k, band=band)
    out = pd.DataFrame({"TS": order})
    out["date"] = [U[j] if j is not None else pd.NaT for j in cal]
    out["piece_size"] = len(order)
    out["holes_used"] = int(((1 - P) * M).sum())
    out["best_mismatch"] = mm[k]
    out["second_mismatch"] = second
    return out


# ---------------------------------------------------------------- checks

def check_with_index(X, dated, index_returns, group_to_index, y=None):
    # if the dates are right, the group average RET_1 on day d moves with the index on day d
    # (and the target with the index on day d+1), and not at other lags
    d = dict(zip(dated["TS"], dated["date"]))
    X = X.copy()
    X["date"] = X["TS"].map(d)
    if y is not None:
        X["target"] = y
    rows = []
    for g, name in group_to_index.items():
        avg = X[X["GROUP"] == g].groupby("date").mean(numeric_only=True)
        r = index_returns[name].dropna()
        for col in ["RET_1", "target"]:
            if col not in avg:
                continue
            for lag in [-1, 0, 1]:
                rr = r.shift(-lag).reindex(avg.index)
                ok = rr.notna() & avg[col].notna()
                rows.append({"group": g, "index": name, "column": col, "lag": lag,
                             "corr": np.corrcoef(avg[col][ok], rr[ok])[0, 1]})
    return pd.DataFrame(rows).pivot_table(index=["group", "index", "column"], columns="lag", values="corr")


# ---------------------------------------------------------------- step 5 : dating with returns
# For pieces without holidays (often only one group in the data) and for isolated dates
# (the test set), the holiday pattern says nothing. Instead we use the returns:
# each allocation is (a bit) exposed to the market indexes. We learn these exposures on the
# dates we already know, then look for the calendar day where the 20-day history of the
# allocations is best explained by the real index returns.

def index_features(index_returns, days):
    # same-day returns of every index + returns of the previous session (US closes after Asia)
    F = index_returns.reindex(days).fillna(0).values
    F_prev = np.vstack([np.zeros((1, F.shape[1])), F[:-1]])
    return np.hstack([F, F_prev])


def fit_betas(X, dated, index_returns, sessions, group_exchange, ridge=1e-6):
    # one small regression per allocation: RET_1 on day d ~ index returns around day d
    date_of = dict(zip(dated["TS"], dated["date"]))
    X = X.copy()
    X["date"] = X["TS"].map(date_of)
    X = X.dropna(subset=["date"])

    feats_by_group = {}
    for g, code in group_exchange.items():
        cal = pd.DatetimeIndex(sorted(sessions[code]))
        feats_by_group[g] = pd.DataFrame(index_features(index_returns, cal), index=cal)

    betas = {}
    for alloc, sub in X.groupby("ALLOCATION"):
        g = sub["GROUP"].iloc[0]
        F = feats_by_group[g].reindex(sub["date"]).values
        y = sub["RET_1"].values
        ok = ~np.isnan(F).any(axis=1) & ~np.isnan(y)
        F, y = F[ok], y[ok]
        if len(y) < 50:
            continue
        betas[alloc] = np.linalg.solve(F.T @ F + ridge * len(y) * np.eye(F.shape[1]), F.T @ y)
    return betas


class ReturnMatcher:
    # precomputes, for each group, all the 20-day windows of index features on its exchange calendar

    def __init__(self, betas, index_returns, sessions, group_exchange):
        self.betas = betas
        self.windows = {}
        for g, code in group_exchange.items():
            cal = pd.DatetimeIndex(sorted(sessions[code]))
            F = index_features(index_returns, cal)
            W = np.lib.stride_tricks.sliding_window_view(F, 20, axis=0)   # (n_days-19, n_feats, 20)
            G = np.einsum("okt,olt->okl", W, W)                              # gram matrix of each window
            self.windows[g] = (cal[19:], W, G)

    def score_date(self, rows):
        # rows = all the rows of ONE TS. Returns a score for every possible calendar day:
        # how much the index returns reduce the squared error of the allocations' 20-day history
        total = None
        for g, sub in rows.groupby("GROUP"):
            if g not in self.windows:
                continue
            Y, B = [], []
            for alloc, hist in zip(sub["ALLOCATION"], sub[OLD_TO_NEW].values):
                if alloc in self.betas:
                    Y.append(np.nan_to_num(hist.astype(float)))
                    B.append(self.betas[alloc])
            if not Y:
                continue
            Y, B = np.array(Y), np.array(B)
            days, W, G = self.windows[g]
            lin = np.einsum("okt,kt->o", W, B.T @ Y)
            quad = np.einsum("okl,kl->o", G, B.T @ B)
            s = pd.Series(2 * lin - quad, index=days)
            total = s if total is None else (total + s).dropna()
        return total

    def date_piece(self, X, order, calendar_days, used_days=None):
        # a piece = consecutive sessions. row i is on calendar_days[o + i], we look for o.
        by_ts = X.groupby("TS")
        n = len(order)
        total = np.zeros(len(calendar_days) - n + 1)
        for i, ts in enumerate(order):
            s = self.score_date(by_ts.get_group(ts))
            if s is None:
                continue
            s = s.reindex(calendar_days).fillna(-1e6).values
            total += s[i:i + len(total)]
        if used_days:
            # forbid positions that overlap days already given to another piece
            taken = np.array([d in used_days for d in calendar_days], dtype=float)
            overlap = np.round(fftconvolve(taken, np.ones(n), mode="valid"))
            total[overlap > 0] = -np.inf
        best = int(np.argmax(total))
        rest = np.concatenate([total[:max(best - 5, 0)], total[best + 5:]])
        second = rest.max() if len(rest) else -np.inf
        out = pd.DataFrame({"TS": order, "date": calendar_days[best:best + n]})
        out["piece_size"] = n
        out["score"] = total[best]
        out["second_score"] = second
        return out


def date_isolated(X, matcher, excluded_days=None):
    # one date at a time (no links): best calendar day + how far ahead of the 2nd best it is
    rows = []
    for ts, sub in X.groupby("TS"):
        s = matcher.score_date(sub)
        if s is None or len(s) == 0:
            rows.append({"TS": ts, "date": pd.NaT})
            continue
        if excluded_days:
            s = s[~s.index.isin(list(excluded_days))]
        s = s.sort_values(ascending=False)
        rows.append({"TS": ts, "date": s.index[0], "score": s.iloc[0],
                     "margin": s.iloc[0] - s.iloc[1], "second_date": s.index[1]})
    return pd.DataFrame(rows)
