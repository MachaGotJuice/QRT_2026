"""
Étape 1 — Reconstituer l'ordre chronologique des dates du train.

Principe : pour une même allocation, si la ligne B est k jours après la ligne A,
alors RET_{1+k}..RET_20 de B ≈ RET_1..RET_{20-k} de A (fenêtre glissante,
valeurs légèrement bruitées). On cherche ces correspondances allocation par
allocation, on agrège les votes au niveau des dates, puis on chaîne les dates.
Les décalages k>1 servent à enjamber des jours absents du train.

Sortie : calendar/train_order.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RET = [f"RET_{i}" for i in range(1, 21)]
MAX_SHIFT = 5        # décalage max (en séances) pour enjamber des jours manquants
TOP = 3              # candidats retenus par ligne et par décalage
MIN_DIST = 0.15      # distance relative max pour accepter un recouvrement ligne→ligne
MIN_SUPPORT = 0.5    # part des allocations communes qui doivent confirmer le lien


def row_links(X):
    """Recouvrements ligne→ligne (même allocation) pour chaque décalage k."""
    out = []
    for alloc, a in X.groupby("ALLOCATION", sort=False):
        R = a[RET].fillna(0).to_numpy(np.float64)
        ts = a.TS.values
        for k in range(1, MAX_SHIFT + 1):
            S, P = R[:, k:20], R[:, 0:20 - k]          # S : successeur, P : prédécesseur
            d2 = (S**2).sum(1)[:, None] + (P**2).sum(1)[None, :] - 2 * S @ P.T
            D = np.sqrt(np.clip(d2, 0, None)) / (np.linalg.norm(P, axis=1)[None, :] + 1e-12)
            np.fill_diagonal(D, np.inf)
            idx = np.argsort(D, axis=0)[:TOP]           # meilleurs successeurs pour chaque prédécesseur
            for r in range(TOP):
                d = D[idx[r], np.arange(len(ts))]
                m = d < MIN_DIST
                out.append(pd.DataFrame({"TS": ts[m], "TS_next": ts[idx[r]][m],
                                         "k": k, "dist": d[m], "ALLOCATION": alloc}))
    return pd.concat(out, ignore_index=True)


def date_links(links, X):
    alloc_sets = X.groupby("TS").ALLOCATION.agg(frozenset)
    v = links.groupby(["TS", "TS_next", "k"]).size().rename("votes").reset_index()
    v["common"] = [len(alloc_sets[a] & alloc_sets[b]) for a, b in zip(v.TS, v.TS_next)]
    v["support"] = v.votes / v.common.clip(lower=1)
    v = v[(v.support >= MIN_SUPPORT) & (v.votes >= 5)]
    # pour chaque date : le plus petit décalage bien soutenu
    v = v.sort_values(["TS", "k", "support"], ascending=[True, True, False]).drop_duplicates("TS")
    v = v.sort_values(["k", "support"], ascending=[True, False]).drop_duplicates("TS_next")
    return v


def build_chains(dates, nxt, gap):
    prev = {b: a for a, b in nxt.items()}
    rows, seen = [], set()
    for seg, h in enumerate([d for d in dates if d not in prev]):
        d, day = h, 0
        while d is not None and d not in seen:
            seen.add(d)
            rows.append((d, seg, day))
            n = nxt.get(d)
            day += gap.get(d, 0)
            d = n
    rows += [(d, -1, 0) for d in dates if d not in seen]
    return pd.DataFrame(rows, columns=["TS", "segment", "session_in_segment"])


if __name__ == "__main__":
    X = pd.read_csv(ROOT / "data/X_train.csv", usecols=["ROW_ID", "TS", "ALLOCATION"] + RET)
    links = row_links(X)
    dl = date_links(links, X)
    nxt, gap = dict(zip(dl.TS, dl.TS_next)), dict(zip(dl.TS, dl.k))
    dates = sorted(X.TS.unique())
    cal = build_chains(dates, nxt, gap)
    seg = cal.groupby("segment").session_in_segment.agg(["size", "max"])
    cal = cal.merge(dl[["TS", "TS_next", "k", "support"]], on="TS", how="left")
    cal.to_csv(ROOT / "calendar/train_order.csv", index=False)

    print(f"dates: {len(dates)} | liens: {len(dl)} | segments: {len(seg)}")
    print("liens par décalage k:", dl.k.value_counts().sort_index().to_dict())
    print("plus longs segments (nb dates):", seg["size"].sort_values(ascending=False).head(12).tolist())
    print("segments d'une seule date:", int((seg["size"] == 1).sum()))
    print("support (quantiles):", dl.support.quantile([.05, .25, .5]).round(3).to_dict())
