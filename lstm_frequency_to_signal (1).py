import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings("ignore")

# ─── Load Data ───────────────────────────────────────────────────────────────
data = pd.read_csv("sample_frequency_data_100.csv")

# ✅ 1 Input (frequency) → 1 Output (signal_strength), nothing else
X = data[["frequency"]].values
y = data[["signal_strength"]].values

print(f"Dataset shape: {data.shape}")
print(f"Frequency range  : {X.min():.2f} – {X.max():.2f} MHz")
print(f"Signal Str range : {y.min():.2f} – {y.max():.2f} dBm")

# ─── Scale both to [0,1] ──────────────────────────────────────────────────────
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# ─── Sequence Builder ─────────────────────────────────────────────────────────
def create_sequences(X, y, look_back):
    Xs, ys = [], []
    for i in range(len(X) - look_back):
        Xs.append(X[i:i+look_back])
        ys.append(y[i+look_back])
    return np.array(Xs), np.array(ys)

# ─── Key fix: look_back=3 to preserve more samples on 100-row data ────────────
LOOK_BACK = 3
X_seq, y_seq = create_sequences(X_scaled, y_scaled, LOOK_BACK)

# ─── Key fix: temporal split — NO random shuffle for time series! ─────────────
split = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

print(f"\nSequences total: {len(X_seq)} | Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Input shape: {X_train.shape}  → (samples, look_back=3, features=1)")

# ─── LSTM Model ───────────────────────────────────────────────────────────────
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(LOOK_BACK, 1)),
    BatchNormalization(),
    Dropout(0.1),
    LSTM(32, return_sequences=False),
    Dropout(0.1),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.005), loss="mse")
model.summary()

# ─── Callbacks ────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=1e-5, verbose=0)
]

history = model.fit(
    X_train, y_train,
    epochs=1000,
    batch_size=4,           # small batch = better gradient updates on tiny dataset
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# ─── Predictions ─────────────────────────────────────────────────────────────
pred      = model.predict(X_test, verbose=0)
actual    = scaler_y.inverse_transform(y_test).flatten()
predicted = scaler_y.inverse_transform(pred).flatten()

# ─── Metrics ─────────────────────────────────────────────────────────────────
mae  = mean_absolute_error(actual, predicted)
mse  = mean_squared_error(actual, predicted)
rmse = np.sqrt(mse)
r2   = r2_score(actual, predicted)
acc1 = np.mean(np.abs(actual - predicted) <= 1) * 100
acc2 = np.mean(np.abs(actual - predicted) <= 2) * 100
acc5 = np.mean(np.abs(actual - predicted) <= 5) * 100

print("\n" + "="*52)
print("    RESULTS  —  Frequency → Signal Strength LSTM")
print("="*52)
print(f"  MAE           : {mae:.4f} dBm")
print(f"  RMSE          : {rmse:.4f} dBm")
print(f"  MSE           : {mse:.4f}")
print(f"  R² Score      : {r2:.4f}  {'✅ POSITIVE' if r2 > 0 else '❌ NEGATIVE'}")
print(f"  Accuracy ±1dBm: {acc1:.1f}%")
print(f"  Accuracy ±2dBm: {acc2:.1f}%")
print(f"  Accuracy ±5dBm: {acc5:.1f}%")
print("="*52)

# ─── Plots ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("LSTM: Frequency (Input) → Signal Strength (Output)", fontsize=14, fontweight="bold")

# 1. Actual vs Predicted line plot
axes[0].plot(actual,    label="Actual",    marker="o", markersize=4, linewidth=2)
axes[0].plot(predicted, label="Predicted", marker="x", markersize=5, linewidth=2, linestyle="--")
axes[0].set_title(f"Actual vs Predicted\nR²={r2:.4f}  |  MAE={mae:.2f} dBm")
axes[0].set_xlabel("Test Sample Index")
axes[0].set_ylabel("Signal Strength (dBm)")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# 2. Scatter
lim = [min(actual.min(), predicted.min())-1, max(actual.max(), predicted.max())+1]
axes[1].scatter(actual, predicted, alpha=0.85, color="steelblue", edgecolors="white", s=70)
axes[1].plot(lim, lim, 'r--', linewidth=2, label="Perfect Fit")
axes[1].set_xlim(lim); axes[1].set_ylim(lim)
axes[1].set_title(f"Scatter: Actual vs Predicted\nR²={r2:.4f}")
axes[1].set_xlabel("Actual (dBm)"); axes[1].set_ylabel("Predicted (dBm)")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

# 3. Training loss
axes[2].plot(history.history["loss"],     label="Train Loss", linewidth=2)
axes[2].plot(history.history["val_loss"], label="Val Loss",   linewidth=2)
axes[2].set_title("Training & Validation Loss")
axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("MSE Loss")
axes[2].legend(); axes[2].grid(True, alpha=0.3)
axes[2].set_yscale("log")

plt.tight_layout()
plt.savefig("lstm_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved → lstm_results.png")
