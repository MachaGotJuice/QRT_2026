from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
import xgboost as xgb
import lightgbm as lgbm
from sklearn.model_selection import TimeSeriesSplit

import optuna
import numpy as np
import lightgbm as lgbm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_lgbm(X, Y, features, params: dict, num_boost_round: int = 500, early_stopping_rounds: int = 50):
    train_dates = np.array(sorted(
        X["TS"].unique(),
        key=lambda ts: int(ts.split("_")[1])
    ))
    n_splits = 6
    scores = []
    models = []

    splits = TimeSeriesSplit(n_splits=n_splits).split(train_dates)

    for i, (local_train_dates_ids, local_test_dates_ids) in enumerate(splits):
        local_train_dates = train_dates[local_train_dates_ids]
        local_test_dates  = train_dates[local_test_dates_ids]

        local_train_ids = X['TS'].isin(local_train_dates)
        local_test_ids  = X['TS'].isin(local_test_dates)

        X_local_train = X.loc[local_train_ids, features]
        y_local_train = (Y.loc[local_train_ids, 'target'] > 0).astype(int)
        X_local_test  = X.loc[local_test_ids,  features]
        y_local_test = (Y.loc[local_test_ids, 'target'] > 0).astype(int)

        train_data = lgbm.Dataset(X_local_train, label=y_local_train.values)
        valid_data = lgbm.Dataset(X_local_test, label=y_local_test.values, reference=train_data)

        model_trained = lgbm.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[valid_data],
            callbacks=[
                lgbm.early_stopping(early_stopping_rounds, verbose=False),
                lgbm.log_evaluation(period=0),
            ],
        )

        y_local_pred = np.asarray(
            model_trained.predict(X_local_test, num_threads=params.get('num_threads', -1)),
            dtype=float,
        )
        y_local_test_arr = np.asarray(y_local_test.to_numpy(), dtype=int)

        models.append(model_trained)
        score = accuracy_score(
            y_local_test_arr,
            (y_local_pred > 0.5).astype(int),
        )
        scores.append(score)
        print(f"Fold {i+1} - Accuracy: {score * 100:.2f}% (best_iter={model_trained.best_iteration})")

    mean = np.mean(scores) * 100
    std  = np.std(scores)  * 100
    print(f'Accuracy: {mean:.2f}% [{mean-std:.2f} ; {mean+std:.2f}] (+- {std:.2f})')

    return models, scores


def optimize_lgbm(
    X, Y, features,
    n_trials: int = 50,
    num_boost_round: int = 300,
    early_stopping_rounds: int = 50,
    n_splits: int = 3,
    subsample_frac: float = 1.0,
    num_threads: int = -1,
):
    """
    Lance une recherche Optuna sur les hyperparamètres LGBM.

    Paramètres pour accélérer la recherche :
        num_boost_round        – max rounds (early stopping coupe avant)
        early_stopping_rounds  – patience pour l'early stopping
        n_splits               – nombre de folds CV (3 = rapide, 5 = robuste)
        subsample_frac         – fraction des dates à garder (0.5 = 2× plus rapide)
        num_threads            – -1 = tous les cœurs dispo

    Retourne:
        best_params  – meilleurs paramètres trouvés
        study        – objet Optuna pour inspection
    """
    train_dates = np.array(sorted(
    X["TS"].unique(),
    key=lambda ts: int(ts.split("_")[1])
    ))

    # Sous-échantillonner les dates pour accélérer la recherche
    if subsample_frac < 1.0:
        rng = np.random.RandomState(42)
        n_keep = max(int(len(train_dates) * subsample_frac), n_splits + 1)
        # On garde l'ordre chronologique (important pour TimeSeriesSplit)
        keep_idx = np.sort(rng.choice(len(train_dates), size=n_keep, replace=False))
        train_dates = train_dates[keep_idx]

    def objective(trial):
        params = {
            "objective": "binary",
            "metric":    "binary_logloss",
            "verbosity":       -1,
            "num_threads":     num_threads,
            "seed":            42,
            "bagging_seed":      42,
            "feature_fraction_seed": 42,
            "data_random_seed":         42,
            # ── hyperparamètres tunés ──────────────────────────────────────
            "learning_rate":   trial.suggest_float("learning_rate",   1e-3, 0.1,  log=True),
            "max_depth":       trial.suggest_int(  "max_depth",       3,    7),
            "num_leaves":      trial.suggest_int(  "num_leaves",      8,    127),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 500),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq":    1,
            "lambda_l1":       trial.suggest_float("lambda_l1", 1e-4, 10.0, log=True),
            "lambda_l2":       trial.suggest_float("lambda_l2", 1e-4, 10.0, log=True),
        }

        fold_scores = []
        splits = TimeSeriesSplit(n_splits=n_splits).split(train_dates)

        for local_train_ids_dates, local_test_ids_dates in splits:
            local_train_dates = train_dates[local_train_ids_dates]
            local_test_dates  = train_dates[local_test_ids_dates]

            mask_train = X['TS'].isin(local_train_dates)
            mask_test  = X['TS'].isin(local_test_dates)

            X_tr = X.loc[mask_train, features]
            y_tr = (Y.loc[mask_train, 'target'] > 0).astype(int)
            X_te = X.loc[mask_test,  features]
            y_te = (Y.loc[mask_test, 'target'] > 0).astype(int)

            train_data = lgbm.Dataset(X_tr, label=y_tr.values)
            valid_data = lgbm.Dataset(X_te, label=y_te.values, reference=train_data)

            model = lgbm.train(
                params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=[valid_data],
                callbacks=[
                    lgbm.early_stopping(early_stopping_rounds, verbose=False),
                    lgbm.log_evaluation(period=0),  # silence
                ],
            )

            preds = np.asarray(model.predict(X_te, num_threads=num_threads), dtype=float)
            y_te_arr = np.asarray(y_te.to_numpy(), dtype=int)
            score = accuracy_score(
                y_te_arr,
                (preds > 0.5).astype(int),
            )
            fold_scores.append(score)

            # Pruning : coupe les essais peu prometteurs tôt
            trial.report(np.mean(fold_scores), step=len(fold_scores))
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(fold_scores)

    # Pruner : arrête les essais qui semblent mauvais au bout de 2 folds
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    sampler = optuna.samplers.TPESampler(seed=42)
    study  = optuna.create_study(direction="maximize", pruner=pruner, sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n✅ Meilleure accuracy OOF : {study.best_value * 100:.2f}%")
    print(f"   Meilleurs paramètres   : {study.best_params}")

    # Paramètres fixes à réinjecter dans best_params
    best_params = {
        "objective":    "binary",
        "metric":       "binary_logloss",
        "verbosity":    -1,
        "num_threads":  num_threads,
        "bagging_freq": 1,
        "seed":           42,
        "bagging_seed":      42,
        "feature_fraction_seed": 42,
        "data_random_seed":         42,
        **study.best_params,
    }

    return best_params, study


# ── Exemple d'utilisation ────────────────────────────────────────────────────
#
# # 1. Recherche des meilleurs hyperparamètres
# best_params, study = optimize_lgbm(X_train, y_train, features, n_trials=50)
#
# # 2. Entraînement final avec les meilleurs paramètres
# models, scores = run_lgbm(X_train, y_train, features, best_params)
#
# # 3. Inspection (optionnel)
# import optuna.visualization as vis
# vis.plot_param_importances(study).show()
# vis.plot_optimization_history(study).show()

