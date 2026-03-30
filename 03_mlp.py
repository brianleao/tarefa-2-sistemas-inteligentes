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

N_FOLDS = 5
SEED = 42

FEATURES = ['idade', 'fc', 'fr', 'pas', 'spo2', 'temp', 'pr', 'sg', 'fx', 'queim']
TARGET = 'sobr'

os.makedirs("modelos", exist_ok=True)
os.makedirs("resultados", exist_ok=True)
os.makedirs("resultados/figuras", exist_ok=True)

# carrega dados
print("=" * 65)
print("MLP - REGRESSAO - VALIDACAO CRUZADA")
print("=" * 65)

df = pd.read_csv("dados/treino_validacao.csv")
X = df[FEATURES].values
y = df[TARGET].values

print(f"Dataset carregado: {X.shape[0]} amostras | {X.shape[1]} features")
print(f"\nEstatisticas do target (sobr):")
print(f"  Min.: {y.min():.4f}  |  Max.: {y.max():.4f}")
print(f"  Media: {y.mean():.4f}  |  Desvio padrao: {y.std():.4f}")

# pipeline com normalizacao
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(
        max_iter=1000,
        random_state=SEED,
        early_stopping=True,
        n_iter_no_change=20,
    )),
])

# hiperparametros
param_grid = {
    'mlp__hidden_layer_sizes': [
        (32,),
        (64,),
        (128,),
        (64, 32),
        (128, 64),
        (128, 64, 32),
    ],
    'mlp__activation': ['relu', 'tanh'],
    'mlp__learning_rate_init': [0.001, 0.01],
}

cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

clf = GridSearchCV(
    pipe,
    param_grid,
    cv=cv,
    scoring='neg_mean_squared_error',
    return_train_score=True,
    n_jobs=-1,
    verbose=1,
)

n_combinacoes = 6 * 2 * 2
print(f"\nIniciando GridSearchCV:")
print(f"  {n_combinacoes} combinacoes x {N_FOLDS} folds = {n_combinacoes * N_FOLDS} fits")
print(f"  Scoring: neg_mean_squared_error  |  CV: KFold(n={N_FOLDS})\n")
clf.fit(X, y)

# salva resultados
results = pd.DataFrame(clf.cv_results_)
results.to_csv("resultados/mlp_cv_results.csv", index=False)

# top-10 configuracoes
print("\nTOP 10 CONFIGURACOES (por MSE medio de validacao):")
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
             'param_mlp__activation': 'ativacao',
             'param_mlp__learning_rate_init': 'lr',
             'mean_train_score': 'neg_mse_treino',
             'mean_test_score': 'neg_mse_val',
             'std_test_score': 'neg_mse_val_std',
         }))
print(top10.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

# melhor configuracao
best_params = clf.best_params_
best_idx = clf.best_index_

print(f"\nMELHOR HIPERPARAMETRIZACAO:")
for k, v in best_params.items():
    print(f"  {k:40}: {v}")

# scores por fold — GridSearchCV usa neg_MSE, converte para positivo
fold_train_neg = np.array([results.at[best_idx, f'split{i}_train_score'] for i in range(N_FOLDS)])
fold_val_neg = np.array([results.at[best_idx, f'split{i}_test_score'] for i in range(N_FOLDS)])

fold_train_mse = -fold_train_neg
fold_val_mse = -fold_val_neg

mean_train_mse = fold_train_mse.mean()
mean_val_mse = fold_val_mse.mean()
var_train_mse = np.var(fold_train_mse)
var_val_mse = np.var(fold_val_mse)

# tabela de vies
print("\nTABELA DE VIES (MSE medio dos k-folds):")
print(f"{'MSE':<30} {'MLP':>12}")
print(f"{'e_t  (treino)':30} {mean_train_mse:>12.6f}")
print(f"{'e_v  (validacao)':30} {mean_val_mse:>12.6f}")
print(f"{'|e_t - e_v|':30} {abs(mean_train_mse - mean_val_mse):>12.6f}")

# tabela de variancia
print("\nTABELA DE VARIANCIA (MSE dos k-folds):")
print(f"{'VARIANCIA (MSE)':<30} {'MLP':>14}")
print(f"{'Var_t  (treino)':30} {var_train_mse:>14.4e}")
print(f"{'Var_v  (validacao)':30} {var_val_mse:>14.4e}")
print(f"{'|Var_t - Var_v|':30} {abs(var_train_mse - var_val_mse):>14.4e}")

# mse por fold
print("\nMSE POR FOLD (melhor configuracao):")
print(f"{'Fold':>6} {'MSE Treino':>12} {'MSE Validacao':>15}")
print("-" * 36)
for i in range(N_FOLDS):
    print(f"{i+1:>6} {fold_train_mse[i]:>12.6f} {fold_val_mse[i]:>15.6f}")
print("-" * 36)
print(f"{'Media':>6} {mean_train_mse:>12.6f} {mean_val_mse:>15.6f}")
print(f"{'Var':>6} {var_train_mse:>12.4e} {var_val_mse:>15.4e}")

# salva modelo e resumo
joblib.dump(clf.best_estimator_, "modelos/mlp_melhor.pkl")

summary = {
    "best_params": {k: str(v) for k, v in best_params.items()},
    "n_folds": N_FOLDS,
    "scoring": "neg_mean_squared_error",
    "mean_train_mse": float(mean_train_mse),
    "mean_val_mse": float(mean_val_mse),
    "vies": float(abs(mean_train_mse - mean_val_mse)),
    "var_train_mse": float(var_train_mse),
    "var_val_mse": float(var_val_mse),
    "fold_train_mse": fold_train_mse.tolist(),
    "fold_val_mse": fold_val_mse.tolist(),
}
with open("resultados/mlp_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\nSalvo: modelos/mlp_melhor.pkl")
print("Salvo: resultados/mlp_summary.json")
print("Salvo: resultados/mlp_cv_results.csv")

# figura 1: mse por fold
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(1, N_FOLDS + 1)
ax.plot(x, fold_train_mse, "o-", color="steelblue", label="Treino")
ax.plot(x, fold_val_mse, "s-", color="tomato", label="Validacao")
ax.axhline(mean_train_mse, linestyle="--", color="steelblue", alpha=0.5,
           label=f"Media treino = {mean_train_mse:.4f}")
ax.axhline(mean_val_mse, linestyle="--", color="tomato", alpha=0.5,
           label=f"Media val   = {mean_val_mse:.4f}")
ax.set_xlabel("Fold")
ax.set_ylabel("MSE")
ax.set_title(f"MLP - MSE por Fold\n{best_params}")
ax.set_xticks(x)
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("resultados/figuras/mlp_mse_por_fold.png", dpi=150)
plt.close()
print("Salvo: resultados/figuras/mlp_mse_por_fold.png")

# figura 2: comparacao CART x MLP
try:
    with open("resultados/cart_summary.json", encoding="utf-8") as f:
        cart = json.load(f)

    cart_train_mse = np.array(cart["fold_train_mse"])
    cart_val_mse = np.array(cart["fold_val_mse"])

    fig2, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, treino, val, titulo in zip(
        axes,
        [cart_train_mse, fold_train_mse],
        [cart_val_mse, fold_val_mse],
        ["CART", "MLP"],
    ):
        ax.plot(x, treino, "o-", color="steelblue", label="Treino")
        ax.plot(x, val, "s-", color="tomato", label="Validacao")
        ax.axhline(treino.mean(), linestyle="--", color="steelblue", alpha=0.5)
        ax.axhline(val.mean(), linestyle="--", color="tomato", alpha=0.5)
        ax.set_title(f"{titulo}  (treino={treino.mean():.4f} | val={val.mean():.4f})")
        ax.set_xlabel("Fold")
        ax.set_ylabel("MSE")
        ax.set_xticks(x)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

    fig2.suptitle("Comparacao CART x MLP - MSE por Fold", fontsize=12)
    plt.tight_layout()
    plt.savefig("resultados/figuras/comparacao_cart_mlp_folds.png", dpi=150)
    plt.close()
    print("Salvo: resultados/figuras/comparacao_cart_mlp_folds.png")

except FileNotFoundError:
    print("cart_summary.json nao encontrado - grafico comparativo nao gerado")
