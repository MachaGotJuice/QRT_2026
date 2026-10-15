

def get_features(include_categorical=False):
    features = []
    #RET_features = [f'RET_{i}' for i in range(1, 21)]
    #SIGNED_VOLUME_features = [f'SIGNED_VOLUME_{i}' for i in range(1, 21)]
    #TURNOVER_features = ['MEDIAN_DAILY_TURNOVER']

    #features += RET_features + SIGNED_VOLUME_features + TURNOVER_features

    # Moyennes & std
    features += [f'AVERAGE_PERF_{i}' for i in [3, 5, 10, 15, 20]]
    features += [f'ALLOCATIONS_AVERAGE_PERF_{i}' for i in [3, 5, 10, 15, 20]]
    features += [f'STD_PERF_{i}' for i in [5, 20]]
    features += [f'ALLOCATIONS_STD_PERF_{i}' for i in [5, 20]]

    # Momentum / mean-reversion
    features += [f'Momentum_{i}' for i in [3, 5, 10, 20]]
    features += ['Reversion_5_20', 'Reversion_3_10']

    # Qualité du signal
    features += ['Sharpe_20', 'Sharpe_5']
    features += ['Pct_Positive_20', 'Pct_Positive_5']
    features += ['Ret1_Zscore', 'Vol_Ratio_5_20']

    # Volume
    features += ['Vol_Momentum_5', 'Vol_Momentum_20']
    features += ['Vol_Ret_Confirm_1', 'Vol_Ret_Confirm_5']
    features += ['Signed_Vol_Ratio_5_20']

    # Cross-allocation (groupe)
    features += [f'Ret_vs_Group_{i}' for i in [5, 20]]
    features += [f'Rank_in_Group_{i}' for i in [5, 20]]
    features += ['Vol_vs_Group']

    if include_categorical:
        features += ['GROUP', 'ALLOCATION']

    return features

def add_features(X):
    RET_features = [f'RET_{i}' for i in range(1, 21)]
    VOL_features = [f'SIGNED_VOLUME_{i}' for i in range(1, 21)]

    # ── Features benchmark ──────────────────────────────────────────────────
    for i in [3, 5, 10, 15, 20]:
        X[f'AVERAGE_PERF_{i}'] = X[RET_features[:i]].mean(1)
        X[f'ALLOCATIONS_AVERAGE_PERF_{i}'] = X.groupby('TS')[f'AVERAGE_PERF_{i}'].transform('mean')

    for i in [5, 20]:
        X[f'STD_PERF_{i}'] = X[RET_features[:i]].std(1)
        X[f'ALLOCATIONS_STD_PERF_{i}'] = X.groupby('TS')[f'STD_PERF_{i}'].transform('mean')

    # ── Momentum / Mean-reversion ────────────────────────────────────────────
    for i in [3, 5, 10, 20]:
        X[f'Momentum_{i}'] = X[RET_features[:i]].sum(1)

    # Signal de retournement : momentum court terme vs long terme
    X['Reversion_5_20'] = X['Momentum_5'] / (X['Momentum_20'].abs() + 1e-8)
    X['Reversion_3_10'] = X['Momentum_3'] / (X['Momentum_10'].abs() + 1e-8)

    # ── Qualité / régularité du signal ───────────────────────────────────────
    # Sharpe approximé
    X['Sharpe_20'] = X['Momentum_20'] / (X['STD_PERF_20'] + 1e-8)
    X['Sharpe_5']  = X['Momentum_5']  / (X['STD_PERF_5']  + 1e-8)

    # Pourcentage de jours positifs sur 20j
    X['Pct_Positive_20'] = X[RET_features].gt(0).mean(axis=1)
    X['Pct_Positive_5']  = X[RET_features[:5]].gt(0).mean(axis=1)

    # Z-score du dernier rendement (RET_1 = le plus récent)
    X['Ret1_Zscore'] = X['RET_1'] / (X['STD_PERF_20'] + 1e-8)

    # Ratio volatilité récente / longue (régime calme vs agité)
    X['Vol_Ratio_5_20'] = X['STD_PERF_5'] / (X['STD_PERF_20'] + 1e-8)

    # ── Features SIGNED_VOLUME ───────────────────────────────────────────────
    # Momentum volume
    X['Vol_Momentum_5']  = X[VOL_features[:5]].mean(1)
    X['Vol_Momentum_20'] = X[VOL_features].mean(1)

    # Confirmation volume/rendement récent (même signe = confirmation)
    X['Vol_Ret_Confirm_1'] = X['SIGNED_VOLUME_1'] * X['RET_1']
    X['Vol_Ret_Confirm_5'] = X['Vol_Momentum_5']  * X['AVERAGE_PERF_5']

    # Ratio volume récent / volume long (accélération des flux)
    X['Signed_Vol_Ratio_5_20'] = X['Vol_Momentum_5'] / (X['Vol_Momentum_20'].abs() + 1e-8)

    # ── Features cross-allocation (par GROUP) ────────────────────────────────
    for i in [5, 20]:
        # Performance relative à la moyenne du groupe ce jour-là
        X[f'Ret_vs_Group_{i}'] = (
            X[f'AVERAGE_PERF_{i}']
            - X.groupby(['TS', 'GROUP'])[f'AVERAGE_PERF_{i}'].transform('mean')
        )
        # Rang percentile dans le groupe
        X[f'Rank_in_Group_{i}'] = (
            X.groupby(['TS', 'GROUP'])[f'AVERAGE_PERF_{i}'].rank(pct=True)
        )

    # Volatilité relative au groupe
    X['Vol_vs_Group'] = (
        X['STD_PERF_20']
        - X.groupby(['TS', 'GROUP'])['STD_PERF_20'].transform('mean')
    )

    # Catégories natives pour LightGBM, après les calculs par groupe.
    for column in ['GROUP', 'ALLOCATION']:
        X[column] = X[column].astype('category')

    return X
        



