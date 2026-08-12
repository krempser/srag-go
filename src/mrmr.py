"""
mrmr.py — Minimum Redundancy Maximum Relevance (mRMR)
-------------------------------------------------------
Implementação usando sklearn.feature_selection.mutual_info_regression.
Não requer dependências externas além do sklearn.

ALGORITMO (Peng, Long & Ding, 2005):
  1. Calcular MI(Xi, Y) para cada feature Xi com o target Y
  2. Iniciar com a feature de maior MI com Y
  3. A cada passo, selecionar a feature que maximiza:
       score(Xi) = MI(Xi, Y) - (1/|S|) × Σ MI(Xi, Xj)  para Xj em S
     onde S é o conjunto já selecionado
  4. Repetir até selecionar n_features

A diferença em relação ao mRMR clássico é que usamos MI contínua
(mutual_info_regression) para o target e para as correlações entre features.

NOTA SOBRE APLICAÇÃO EM CV:
  Para evitar data leakage, o mRMR deve ser ajustado DENTRO de cada fold
  de treino. A função `mrmr_select_indices` recebe X e y de treino e
  retorna os índices das features selecionadas.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


def mrmr_select_indices(X: np.ndarray, y: np.ndarray,
                        n_features: int,
                        random_state: int = 42) -> list[int]:
    """
    Seleciona os índices das n_features mais relevantes por mRMR.

    Parâmetros
    ----------
    X : array (n_samples, n_total_features)
    y : array (n_samples,)
    n_features : int — número de features a selecionar
    random_state : int

    Retorna
    -------
    list[int] : índices (em X) das features selecionadas, em ordem de seleção
    """
    n_total = X.shape[1]
    n_features = min(n_features, n_total)

    if n_features == n_total:
        return list(range(n_total))

    # MI de cada feature com o target
    mi_target = mutual_info_regression(X, y, random_state=random_state)

    selected  = []
    remaining = list(range(n_total))

    # Passo 1: feature com maior relevância
    best = int(np.argmax(mi_target))
    selected.append(best)
    remaining.remove(best)

    # Cache de MI entre features (evita recalcular)
    mi_cache: dict[tuple, float] = {}

    def mi_features(i: int, j: int) -> float:
        key = (min(i, j), max(i, j))
        if key not in mi_cache:
            mi_cache[key] = mutual_info_regression(
                X[:, [j]], X[:, i], random_state=random_state
            )[0]
        return mi_cache[key]

    # Passos seguintes: maximizar relevância − redundância
    while len(selected) < n_features and remaining:
        scores = []
        for i in remaining:
            relevance  = mi_target[i]
            redundancy = np.mean([mi_features(i, j) for j in selected])
            scores.append(relevance - redundancy)

        best_pos = int(np.argmax(scores))
        best_idx = remaining[best_pos]
        selected.append(best_idx)
        remaining.pop(best_pos)

    return selected


def mrmr_select_names(X_df: pd.DataFrame, y: np.ndarray,
                      n_features: int,
                      random_state: int = 42) -> list[str]:
    """
    Versão que recebe DataFrame e retorna nomes das colunas selecionadas.
    """
    indices = mrmr_select_indices(X_df.values, y, n_features, random_state)
    return [X_df.columns[i] for i in indices]


def mrmr_report(X_df: pd.DataFrame, y: np.ndarray,
                selected_cols: list[str]) -> pd.DataFrame:
    """
    Gera um DataFrame com MI de cada feature selecionada vs target e entre si,
    útil para reportar no artigo.
    """
    mi_target = mutual_info_regression(X_df.values, y, random_state=42)
    rows = []
    for col, mi in zip(X_df.columns, mi_target):
        rows.append({
            "feature":     col,
            "MI_com_alvo": round(mi, 5),
            "selecionada": col in selected_cols,
        })
    return (pd.DataFrame(rows)
            .sort_values("MI_com_alvo", ascending=False)
            .reset_index(drop=True))
