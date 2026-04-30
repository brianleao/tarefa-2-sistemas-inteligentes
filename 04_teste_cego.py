import os
import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import joblib

SEED = 42
FEATURES = ['idade', 'fc', 'fr', 'pas', 'spo2', 'temp', 'pr', 'sg', 'fx', 'queim']
TARGET = 'sobr'

os.makedirs("modelos", exist_ok=True)
os.makedirs("resultados", exist_ok=True)
os.makedirs("resultados/figuras", exist_ok=True)

# carrega datasets
print("=" * 65)
print("RETREINO + TESTE CEGO")
print("=" * 65)

df_tv = pd.read_csv("dados/treino_validacao.csv")
df_test = pd.read_csv("dados/teste_cego.csv")

X_tv = df_tv[FEATURES].values
y_tv = df_tv[TARGET].values
X_test = df_test[FEATURES].values
y_test = df_test[TARGET].values

print(f"Treino/validacao : {X_tv.shape[0]} amostras")
print(f"Teste cego       : {X_test.shape[0]} amostras")

# carrega melhores hiperparametros
with open("resultados/cart_summary.json", encoding="utf-8") as f:
    cart_summary = json.load(f)
with open("resultados/mlp_summary.json", encoding="utf-8") as f:
    mlp_summary = json.load(f)

cart_params = cart_summary["best_params"]
mlp_params = mlp_summary["best_params"]

if cart_params.get("max_depth") == "None":
    cart_params["max_depth"] = None
else:
    cart_params["max_depth"] = int(cart_params["max_depth"])

cart_params["min_samples_leaf"] = int(cart_params["min_samples_leaf"])

print(f"\nMelhor CART : {cart_params}")
print(f"Melhor MLP  : {mlp_params}")

# retreino CART
print("\n" + "-" * 65)
print("Retreinando CART com todo o dataset de treino/validacao")
print("-" * 65)

cart_final = DecisionTreeRegressor(
    criterion=cart_params["criterion"],
    max_depth=cart_params["max_depth"],
    min_samples_leaf=cart_params["min_samples_leaf"],
    random_state=SEED,
)
cart_final.fit(X_tv, y_tv)
joblib.dump(cart_final, "modelos/cart_final.pkl")
print("Salvo: modelos/cart_final.pkl")

# retreino MLP
print("\n" + "-" * 65)
print("Retreinando MLP com todo o dataset de treino/validacao")
print("-" * 65)

hidden = eval(mlp_params["mlp__hidden_layer_sizes"])
activ = mlp_params["mlp__activation"]
lr = float(mlp_params["mlp__learning_rate_init"])

mlp_final = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(
        hidden_layer_sizes=hidden,
        activation=activ,
        learning_rate_init=lr,
        max_iter=1000,
        random_state=SEED,
        early_stopping=False,
    )),
])
mlp_final.fit(X_tv, y_tv)
joblib.dump(mlp_final, "modelos/mlp_final.pkl")
print("Salvo: modelos/mlp_final.pkl")

# teste cego
print("\n" + "=" * 65)
print("TESTE CEGO")
print("=" * 65)

y_pred_cart = cart_final.predict(X_test)
y_pred_mlp = mlp_final.predict(X_test)

mse_cart = mean_squared_error(y_test, y_pred_cart)
mse_mlp = mean_squared_error(y_test, y_pred_mlp)

se_cart = (y_test - y_pred_cart) ** 2
se_mlp = (y_test - y_pred_mlp) ** 2
std_cart = np.std(se_cart)
std_mlp = np.std(se_mlp)

print(f"\n  MSE CART (teste cego) : {mse_cart:.6f}  |  std: {std_cart:.6f}")
print(f"  MSE MLP  (teste cego) : {mse_mlp:.6f}  |  std: {std_mlp:.6f}")

# tabela comparativa
print("\nTABELA COMPARATIVA - TESTE CEGO")
print(f"{'Metrica':<35} {'CART':>12} {'MLP':>12}")
print("-" * 61)
print(f"{'MSE treino (medio validacao cruzada)':<35} {cart_summary['mean_train_mse']:>12.6f} {mlp_summary['mean_train_mse']:>12.6f}")
print(f"{'MSE val   (medio validacao cruzada)':<35} {cart_summary['mean_val_mse']:>12.6f} {mlp_summary['mean_val_mse']:>12.6f}")
print(f"{'MSE teste cego':<35} {mse_cart:>12.6f} {mse_mlp:>12.6f}")
print(f"{'Diferenca |MSE_val - MSE_teste|':<35} {abs(cart_summary['mean_val_mse'] - mse_cart):>12.6f} {abs(mlp_summary['mean_val_mse'] - mse_mlp):>12.6f}")

# analise
print("\n" + "=" * 65)
print("ANALISE")
print("=" * 65)

# scatter real x predito
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, y_pred, titulo, mse_val in zip(
    axes,
    [y_pred_cart, y_pred_mlp],
    ["CART", "MLP"],
    [mse_cart, mse_mlp],
):
    ax.scatter(y_test, y_pred, s=10, alpha=0.4, color="steelblue")
    lim = [0, 1]
    ax.plot(lim, lim, "r--", linewidth=1.5, label="Predicao perfeita")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Valor Real (sobr)")
    ax.set_ylabel("Valor Predito (sobr)")
    ax.set_title(f"{titulo} - Real x Predito\nMSE = {mse_val:.6f}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

plt.suptitle("Grafico de Dispersao - Teste Cego (CART x MLP)", fontsize=13)
plt.tight_layout()
plt.savefig("resultados/figuras/scatter_real_vs_predito.png", dpi=150)
plt.close()
print("Salvo: resultados/figuras/scatter_real_vs_predito.png")

# mse por intervalo de sobr
bins = np.linspace(0, 1, 11)
bin_labels = [f"[{bins[i]:.1f},{bins[i+1]:.1f})" for i in range(len(bins) - 1)]
bin_idx = np.digitize(y_test, bins, right=False) - 1
bin_idx = np.clip(bin_idx, 0, len(bins) - 2)

mse_cart_bins = []
mse_mlp_bins = []
for b in range(len(bins) - 1):
    mask = bin_idx == b
    if mask.sum() > 0:
        mse_cart_bins.append(mean_squared_error(y_test[mask], y_pred_cart[mask]))
        mse_mlp_bins.append(mean_squared_error(y_test[mask], y_pred_mlp[mask]))
    else:
        mse_cart_bins.append(0.0)
        mse_mlp_bins.append(0.0)

x_pos = np.arange(len(bin_labels))
w = 0.35
fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.bar(x_pos - w/2, mse_cart_bins, w, label="CART", color="steelblue")
ax2.bar(x_pos + w/2, mse_mlp_bins, w, label="MLP", color="tomato")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(bin_labels, rotation=30, ha="right")
ax2.set_ylabel("MSE")
ax2.set_xlabel("Intervalo de sobr (valor real)")
ax2.set_title("MSE por Intervalo - Teste Cego")
ax2.legend()
ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("resultados/figuras/mse_por_intervalo.png", dpi=150)
plt.close()
print("Salvo: resultados/figuras/mse_por_intervalo.png")

print("\nMSE POR INTERVALO DE sobr:")
print(f"{'Intervalo':<20} {'N amostras':>12} {'MSE CART':>12} {'MSE MLP':>12}")
print("-" * 58)
for b, label in enumerate(bin_labels):
    mask = bin_idx == b
    n = mask.sum()
    print(f"{label:<20} {n:>12} {mse_cart_bins[b]:>12.6f} {mse_mlp_bins[b]:>12.6f}")

# salva resumo final
final = {
    "cart": {
        "mse_teste": float(mse_cart),
        "std_teste": float(std_cart),
        "mean_val_mse": cart_summary["mean_val_mse"],
        "mse_por_intervalo": mse_cart_bins,
    },
    "mlp": {
        "mse_teste": float(mse_mlp),
        "std_teste": float(std_mlp),
        "mean_val_mse": mlp_summary["mean_val_mse"],
        "mse_por_intervalo": mse_mlp_bins,
    },
    "intervalos": bin_labels,
}
with open("resultados/teste_cego_summary.json", "w", encoding="utf-8") as f:
    json.dump(final, f, indent=2, ensure_ascii=False)

print("\nSalvo: resultados/teste_cego_summary.json")
