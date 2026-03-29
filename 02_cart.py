# -*- coding: utf-8 -*-
"""
02_cart.py
─────────────────────────────────────────────────────────────────────────────
Treinamento e validação cruzada do regressor CART (Árvore de Decisão).

Entradas (features 1–10):
    idade, fc, fr, pas, spo2, temp, pr, sg, fx, queim

Saída (feature 14):
    sobr  →  probabilidade de sobrevivência no intervalo [0, 1]

Proibido usar como entrada: gcs (11), avpu (12), tri (13), sobr (14).

Etapas:
  1. Carrega dados de treino/validação
  2. GridSearchCV com KFold — varia criterion, max_depth e min_samples_leaf
  3. Exibe top-10 configurações e a melhor hiperparametrização
  4. Calcula viés (MSE médio treino vs validação) e variância por fold
  5. Salva modelo, parâmetros e gráficos em resultados/
─────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import GridSearchCV, KFold
import joblib

# ── Configurações ─────────────────────────────────────────────────────────
N_FOLDS  = 5
SEED     = 42

FEATURES = ['idade', 'fc', 'fr', 'pas', 'spo2', 'temp', 'pr', 'sg', 'fx', 'queim']
TARGET   = 'sobr'

os.makedirs("modelos",              exist_ok=True)
os.makedirs("resultados",           exist_ok=True)
os.makedirs("resultados/figuras",   exist_ok=True)

# ── Carregar dados ────────────────────────────────────────────────────────
print("=" * 65)
print("CART — REGRESSÃO — VALIDAÇÃO CRUZADA")
print("=" * 65)

df = pd.read_csv("dados/treino_validacao.csv")
X  = df[FEATURES].values
y  = df[TARGET].values

print(f"Dataset carregado: {X.shape[0]} amostras | {X.shape[1]} features")
print(f"\nEstatísticas do target (sobr):")
print(f"  Mín.: {y.min():.4f}  |  Máx.: {y.max():.4f}")
print(f"  Média: {y.mean():.4f}  |  Desvio padrão: {y.std():.4f}")

# ── Grade de hiperparâmetros ───────────────────────────────────────────────
# 3 × 5 × 4 = 60 combinações  ×  5 folds  =  300 fits
param_grid = {
    'criterion':        ['squared_error', 'friedman_mse', 'absolute_error'],
    'max_depth':        [3, 5, 10, 15, None],
    'min_samples_leaf': [4, 8, 16, 32],
}

cv    = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
model = DecisionTreeRegressor(random_state=SEED)

clf = GridSearchCV(
    model,
    param_grid,
    cv                = cv,
    scoring           = 'neg_mean_squared_error',
    return_train_score= True,
    n_jobs            = -1,
    verbose           = 1,
)

n_combinacoes = 3 * 5 * 4
print(f"\nIniciando GridSearchCV:")
print(f"  {n_combinacoes} combinações × {N_FOLDS} folds = {n_combinacoes * N_FOLDS} fits")
print(f"  Scoring: neg_mean_squared_error  |  CV: KFold(n={N_FOLDS})\n")
clf.fit(X, y)

# ── Salvar resultados completos ────────────────────────────────────────────
results = pd.DataFrame(clf.cv_results_)
results.to_csv("resultados/cart_cv_results.csv", index=False)

# ── Top-10 configurações ──────────────────────────────────────────────────
print("\n── TOP 10 CONFIGURAÇÕES (por MSE médio de validação) ──")
cols_exibir = [
    'param_criterion', 'param_max_depth', 'param_min_samples_leaf',
    'mean_train_score', 'mean_test_score', 'std_test_score',
]
top10 = (results[cols_exibir]
         .sort_values('mean_test_score', ascending=False)
         .head(10)
         .rename(columns={
             'param_criterion':        'criterion',
             'param_max_depth':        'max_depth',
             'param_min_samples_leaf': 'min_samples_leaf',
             'mean_train_score':       'neg_mse_treino',
             'mean_test_score':        'neg_mse_val',
             'std_test_score':         'neg_mse_val_std',
         }))
print(top10.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

# ── Melhor modelo ─────────────────────────────────────────────────────────
best_params = clf.best_params_
best_idx    = clf.best_index_

print(f"\n── MELHOR HIPERPARAMETRIZAÇÃO ──")
for k, v in best_params.items():
    print(f"  {k:20}: {v}")

# ── Scores por fold da melhor configuração ────────────────────────────────
# GridSearchCV armazena neg_MSE → converter para MSE positivo
fold_train_neg = np.array([results.at[best_idx, f'split{i}_train_score'] for i in range(N_FOLDS)])
fold_val_neg   = np.array([results.at[best_idx, f'split{i}_test_score']  for i in range(N_FOLDS)])

fold_train_mse = -fold_train_neg   # MSE positivo de treino por fold
fold_val_mse   = -fold_val_neg     # MSE positivo de validação por fold

mean_train_mse = fold_train_mse.mean()
mean_val_mse   = fold_val_mse.mean()
var_train_mse  = np.var(fold_train_mse)   # 1/k * Σ(mse_i − msē)²
var_val_mse    = np.var(fold_val_mse)

# ── Tabela de Viés (MSE médio) ─────────────────────────────────────────────
print("\n── TABELA DE VIÉS (MSE médio dos k-folds) ──")
print(f"{'MSE':<30} {'CART':>12}")
print(f"{'ε̄_t  (treino)':30} {mean_train_mse:>12.6f}")
print(f"{'ε̄_v  (validação)':30} {mean_val_mse:>12.6f}")
print(f"{'|ε̄_t − ε̄_v|':30} {abs(mean_train_mse - mean_val_mse):>12.6f}")

# ── Tabela de Variância ────────────────────────────────────────────────────
print("\n── TABELA DE VARIÂNCIA (MSE dos k-folds) ──")
print(f"{'VARIÂNCIA (MSE)':<30} {'CART':>14}")
print(f"{'Var_t  (treino)':30} {var_train_mse:>14.4e}")
print(f"{'Var_v  (validação)':30} {var_val_mse:>14.4e}")
print(f"{'|Var_t − Var_v|':30} {abs(var_train_mse - var_val_mse):>14.4e}")

# ── Detalhamento por fold ──────────────────────────────────────────────────
print("\n── MSE POR FOLD (melhor configuração) ──")
print(f"{'Fold':>6} {'MSE Treino':>12} {'MSE Validação':>15}")
print("-" * 36)
for i in range(N_FOLDS):
    print(f"{i+1:>6} {fold_train_mse[i]:>12.6f} {fold_val_mse[i]:>15.6f}")
print("-" * 36)
print(f"{'Média':>6} {mean_train_mse:>12.6f} {mean_val_mse:>15.6f}")
print(f"{'Var':>6} {var_train_mse:>12.4e} {var_val_mse:>15.4e}")

# ── Persistência ──────────────────────────────────────────────────────────
joblib.dump(clf.best_estimator_, "modelos/cart_melhor.pkl")

summary = {
    "best_params"      : {k: (v if v is not None else "None") for k, v in best_params.items()},
    "n_folds"          : N_FOLDS,
    "scoring"          : "neg_mean_squared_error",
    "mean_train_mse"   : float(mean_train_mse),
    "mean_val_mse"     : float(mean_val_mse),
    "vies"             : float(abs(mean_train_mse - mean_val_mse)),
    "var_train_mse"    : float(var_train_mse),
    "var_val_mse"      : float(var_val_mse),
    "fold_train_mse"   : fold_train_mse.tolist(),
    "fold_val_mse"     : fold_val_mse.tolist(),
}
with open("resultados/cart_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n→ modelos/cart_melhor.pkl")
print("→ resultados/cart_summary.json")
print("→ resultados/cart_cv_results.csv")

# ── Figura 1: MSE por fold ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(1, N_FOLDS + 1)
ax.plot(x, fold_train_mse, "o-", color="steelblue", label="Treino")
ax.plot(x, fold_val_mse,   "s-", color="tomato",    label="Validação")
ax.axhline(mean_train_mse, linestyle="--", color="steelblue", alpha=0.5,
           label=f"Média treino = {mean_train_mse:.4f}")
ax.axhline(mean_val_mse,   linestyle="--", color="tomato",    alpha=0.5,
           label=f"Média val   = {mean_val_mse:.4f}")
ax.set_xlabel("Fold")
ax.set_ylabel("MSE")
ax.set_title(f"CART — MSE por Fold\n{best_params}")
ax.set_xticks(x)
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("resultados/figuras/cart_mse_por_fold.png", dpi=150)
plt.close()
print("→ resultados/figuras/cart_mse_por_fold.png")

# ── Figura 2: Árvore de decisão (visualização limitada a depth=4) ─────────
fig2, ax2 = plt.subplots(figsize=(24, 10))
plot_tree(
    clf.best_estimator_,
    feature_names = FEATURES,
    filled        = True,
    rounded       = True,
    fontsize      = 6,
    max_depth     = 4,
    ax            = ax2,
)
ax2.set_title(f"CART — Árvore Aprendida (exibindo até depth=4)\n{best_params}", fontsize=10)
plt.tight_layout()
plt.savefig("resultados/figuras/cart_arvore.png", dpi=120)
plt.close()
print("→ resultados/figuras/cart_arvore.png")

# ── Figura 3: Importância das features ────────────────────────────────────
importances = clf.best_estimator_.feature_importances_
ordem = np.argsort(importances)[::-1]
fig3, ax3 = plt.subplots(figsize=(9, 4))
ax3.bar(range(len(FEATURES)), importances[ordem], color="steelblue")
ax3.set_xticks(range(len(FEATURES)))
ax3.set_xticklabels([FEATURES[i] for i in ordem], rotation=30, ha="right")
ax3.set_ylabel("Importância")
ax3.set_title("CART — Importância das Features")
ax3.grid(True, axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("resultados/figuras/cart_feature_importance.png", dpi=150)
plt.close()
print("→ resultados/figuras/cart_feature_importance.png")

print("\n✔  02_cart.py concluído.")
