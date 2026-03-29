# -*- coding: utf-8 -*-
"""
03_mlp.py
─────────────────────────────────────────────────────────────────────────────
Treinamento e validação cruzada da Rede Neural MLP (MLPRegressor).

Entradas (features 1–10):
    idade, fc, fr, pas, spo2, temp, pr, sg, fx, queim

Saída (feature 14):
    sobr  →  probabilidade de sobrevivência no intervalo [0, 1]

Etapas:
  1. Carrega e normaliza os dados de treino/validação
  2. GridSearchCV com KFold — varia hidden_layers, activation e
     learning_rate_init
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

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
print("MLP — REGRESSÃO — VALIDAÇÃO CRUZADA")
print("=" * 65)

df = pd.read_csv("dados/treino_validacao.csv")
X  = df[FEATURES].values
y  = df[TARGET].values

print(f"Dataset carregado: {X.shape[0]} amostras | {X.shape[1]} features")
print(f"\nEstatísticas do target (sobr):")
print(f"  Mín.: {y.min():.4f}  |  Máx.: {y.max():.4f}")
print(f"  Média: {y.mean():.4f}  |  Desvio padrão: {y.std():.4f}")

# ── Pipeline: StandardScaler + MLPRegressor ───────────────────────────────
# MLP é sensível à escala das features — normalização é essencial
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp',    MLPRegressor(
        max_iter         = 1000,
        random_state     = SEED,
        early_stopping   = True,
        n_iter_no_change = 20,
    )),
])

# ── Grade de hiperparâmetros ───────────────────────────────────────────────
# Variações:
#   hidden_layer_sizes : arquitetura da rede (camadas × neurônios)
#   activation         : função de ativação
#   learning_rate_init : taxa de aprendizado inicial
#
# Nomenclatura: (64,) = 1 camada oculta de 64 neurônios
#               (128, 64) = 2 camadas ocultas de 128 e 64 neurônios
param_grid = {
    'mlp__hidden_layer_sizes': [
        (32,),
        (64,),
        (128,),
        (64, 32),
        (128, 64),
        (128, 64, 32),
    ],
    'mlp__activation':          ['relu', 'tanh'],
    'mlp__learning_rate_init':  [0.001, 0.01],
}
# 6 × 2 × 2 = 24 combinações × 5 folds = 120 fits

cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

clf = GridSearchCV(
    pipe,
    param_grid,
    cv                 = cv,
    scoring            = 'neg_mean_squared_error',
    return_train_score = True,
    n_jobs             = -1,
    verbose            = 1,
)

n_combinacoes = 6 * 2 * 2
print(f"\nIniciando GridSearchCV:")
print(f"  {n_combinacoes} combinações × {N_FOLDS} folds = {n_combinacoes * N_FOLDS} fits")
print(f"  Scoring: neg_mean_squared_error  |  CV: KFold(n={N_FOLDS})")
print(f"  Pipeline: StandardScaler → MLPRegressor(max_iter=1000)\n")
clf.fit(X, y)

# ── Salvar resultados completos ────────────────────────────────────────────
results = pd.DataFrame(clf.cv_results_)
results.to_csv("resultados/mlp_cv_results.csv", index=False)

# ── Top-10 configurações ──────────────────────────────────────────────────
print("\n── TOP 10 CONFIGURAÇÕES (por MSE médio de validação) ──")
cols_exibir = [
    'param_mlp__hidden_layer_sizes',
    'param_mlp__activation',
    'param_mlp__learning_rate_init',
    'mean_train_score',
    'mean_test_score',
    'std_test_score',
]
top10 = (results[cols_exibir]
         .sort_values('mean_test_score', ascending=False)
         .head(10)
         .rename(columns={
             'param_mlp__hidden_layer_sizes': 'camadas',
             'param_mlp__activation':         'ativação',
             'param_mlp__learning_rate_init': 'lr',
             'mean_train_score':              'neg_mse_treino',
             'mean_test_score':              'neg_mse_val',
             'std_test_score':               'neg_mse_val_std',
         }))
print(top10.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

# ── Melhor modelo ─────────────────────────────────────────────────────────
best_params = clf.best_params_
best_idx    = clf.best_index_

print(f"\n── MELHOR HIPERPARAMETRIZAÇÃO ──")
for k, v in best_params.items():
    print(f"  {k:40}: {v}")

# ── Scores por fold da melhor configuração ────────────────────────────────
fold_train_neg = np.array([results.at[best_idx, f'split{i}_train_score'] for i in range(N_FOLDS)])
fold_val_neg   = np.array([results.at[best_idx, f'split{i}_test_score']  for i in range(N_FOLDS)])

fold_train_mse = -fold_train_neg
fold_val_mse   = -fold_val_neg

mean_train_mse = fold_train_mse.mean()
mean_val_mse   = fold_val_mse.mean()
var_train_mse  = np.var(fold_train_mse)
var_val_mse    = np.var(fold_val_mse)

# ── Tabela de Viés (MSE médio) ─────────────────────────────────────────────
print("\n── TABELA DE VIÉS (MSE médio dos k-folds) ──")
print(f"{'MSE':<30} {'MLP':>12}")
print(f"{'ε̄_t  (treino)':30} {mean_train_mse:>12.6f}")
print(f"{'ε̄_v  (validação)':30} {mean_val_mse:>12.6f}")
print(f"{'|ε̄_t − ε̄_v|':30} {abs(mean_train_mse - mean_val_mse):>12.6f}")

# ── Tabela de Variância ────────────────────────────────────────────────────
print("\n── TABELA DE VARIÂNCIA (MSE dos k-folds) ──")
print(f"{'VARIÂNCIA (MSE)':<30} {'MLP':>14}")
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
joblib.dump(clf.best_estimator_, "modelos/mlp_melhor.pkl")

summary = {
    "best_params"    : {k: str(v) for k, v in best_params.items()},
    "n_folds"        : N_FOLDS,
    "scoring"        : "neg_mean_squared_error",
    "mean_train_mse" : float(mean_train_mse),
    "mean_val_mse"   : float(mean_val_mse),
    "vies"           : float(abs(mean_train_mse - mean_val_mse)),
    "var_train_mse"  : float(var_train_mse),
    "var_val_mse"    : float(var_val_mse),
    "fold_train_mse" : fold_train_mse.tolist(),
    "fold_val_mse"   : fold_val_mse.tolist(),
}
with open("resultados/mlp_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n→ modelos/mlp_melhor.pkl")
print("→ resultados/mlp_summary.json")
print("→ resultados/mlp_cv_results.csv")

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
ax.set_title(f"MLP — MSE por Fold\n{best_params}")
ax.set_xticks(x)
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("resultados/figuras/mlp_mse_por_fold.png", dpi=150)
plt.close()
print("→ resultados/figuras/mlp_mse_por_fold.png")

# ── Figura 2: comparação CART × MLP (MSE por fold) ───────────────────────
try:
    with open("resultados/cart_summary.json", encoding="utf-8") as f:
        cart = json.load(f)

    cart_train_mse = np.array(cart["fold_train_mse"])
    cart_val_mse   = np.array(cart["fold_val_mse"])

    fig2, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, treino, val, titulo in zip(
        axes,
        [cart_train_mse, fold_train_mse],
        [cart_val_mse,   fold_val_mse],
        ["CART", "MLP"],
    ):
        ax.plot(x, treino, "o-", color="steelblue", label="Treino")
        ax.plot(x, val,    "s-", color="tomato",    label="Validação")
        ax.axhline(treino.mean(), linestyle="--", color="steelblue", alpha=0.5)
        ax.axhline(val.mean(),    linestyle="--", color="tomato",    alpha=0.5)
        ax.set_title(f"{titulo}  (treino={treino.mean():.4f} | val={val.mean():.4f})")
        ax.set_xlabel("Fold")
        ax.set_ylabel("MSE")
        ax.set_xticks(x)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

    fig2.suptitle("Comparação CART × MLP — MSE por Fold", fontsize=12)
    plt.tight_layout()
    plt.savefig("resultados/figuras/comparacao_cart_mlp_folds.png", dpi=150)
    plt.close()
    print("→ resultados/figuras/comparacao_cart_mlp_folds.png")

except FileNotFoundError:
    print("(cart_summary.json não encontrado — gráfico comparativo não gerado)")

print("\n✔  03_mlp.py concluído.")
