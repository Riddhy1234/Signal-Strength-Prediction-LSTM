# Signal-Strength-Prediction-LSTM
# Signal Strength Prediction using LSTM

## Project Overview

This project explores the use of Long Short-Term Memory (LSTM) Neural Networks for predicting signal strength (dBm) from frequency measurements.

The objective was to investigate whether frequency alone can be used to predict signal strength using a deep learning approach.

---

## Dataset

Dataset File:
sample_frequency_data_100.csv

Dataset Characteristics:

- Total Samples: 100
- Input Feature: Frequency
- Output Feature: Signal Strength (dBm)

Frequency was used as the only input variable and signal strength was used as the prediction target.

---

## Model Architecture

The implemented LSTM model consists of:

- LSTM Layer (64 units)
- Batch Normalization
- Dropout Layer
- LSTM Layer (32 units)
- Dense Layer (16 neurons)
- Output Layer (1 neuron)

Additional techniques:

- Early Stopping
- ReduceLROnPlateau
- Min-Max Normalization

---

## Libraries Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn
- TensorFlow / Keras

---

## Evaluation Metrics

The model was evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score
- Accuracy within ±1 dB
- Accuracy within ±2 dB
- Accuracy within ±5 dB

---

## Results

| Metric | Value |
|----------|----------|
| MAE | 1.91 dBm |
| RMSE | 2.45 dBm |
| MSE | 6.01 |
| R² Score | -0.30 |
| Accuracy ±1 dB | 40% |
| Accuracy ±2 dB | 65% |
| Accuracy ±5 dB | 95% |

---

## Result Visualization

The repository includes:

- Actual vs Predicted Signal Strength Plot
- Scatter Plot
- Training vs Validation Loss Curve

File:

lstm_results.png

---

## Observations

The model achieved strong performance within ±5 dB.

However, the negative R² score indicates that frequency alone does not contain sufficient information to accurately explain variations in signal strength for this dataset.

Increasing model complexity and neuron count did not significantly improve performance, suggesting limitations in the available input feature.

---

## Future Work

Possible improvements include:

- Larger datasets
- Additional signal-related features
- Time-series measurements
- Feature engineering
- Alternative deep learning architectures such as GRU and Bi-LSTM

---

## Author

Riddhy Gupta

B.Tech CSE (AI & ML)

College of Engineering Roorkee

