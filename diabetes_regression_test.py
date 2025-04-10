import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Veri Kümesi Oluşturuluyor
np.random.seed(42)

n = 100
age = np.random.randint(20, 70, size=n)
bmi = np.random.normal(25, 4, size=n)  # Ortalama BMI 25, sapma 4
bp = np.random.normal(120, 10, size=n)  # Ortalama tansiyon 120
cholesterol = np.random.normal(200, 15, size=n)  # Ortalama kolesterol 200

# Hedef değişken target (y)
# Basit formül: yaşın %0.5 etkisi, BMI %1.5, bp %0.4, cholesterol %0.3 + ufak rastgele gürültü
target = (
    0.5 * age +
    1.5 * bmi +
    0.4 * bp +
    0.3 * cholesterol +
    np.random.normal(0, 10, size=n)  # Gürültü ekleyerek daha gerçekçi sonuç
)

# Veriyi DataFrame'e dönüştür
df = pd.DataFrame({
    "age": age,
    "bmi": bmi,
    "bp": bp,
    "cholesterol": cholesterol,
    "target": target
})

print(" Veri Kümesinin İlk 5 Satırı:")
print(df.head(), "\n")

# 2. Basit Lineer Regresyon (Bağımsız Değişken: bmi)
X_simple = df[["bmi"]]
y = df["target"]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple, y, test_size=0.2, random_state=42)

model_simple = LinearRegression()
model_simple.fit(X_train_s, y_train_s)
y_pred_s = model_simple.predict(X_test_s)

r2_s = r2_score(y_test_s, y_pred_s)
mae_s = mean_absolute_error(y_test_s, y_pred_s)
mse_s = mean_squared_error(y_test_s, y_pred_s)

# 3. Çoklu Lineer Regresyon (Tüm Bağımsız Değişkenler)
X_multi = df[["age", "bmi", "bp", "cholesterol"]]

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y, test_size=0.2, random_state=42)

model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)
y_pred_m = model_multi.predict(X_test_m)

r2_m = r2_score(y_test_m, y_pred_m)
mae_m = mean_absolute_error(y_test_m, y_pred_m)
mse_m = mean_squared_error(y_test_m, y_pred_m)

# 4. Sonuçları Yazdır
print(" Basit Lineer Regresyon (Sadece BMI):")
print(f"R² Skoru: {r2_s:.4f}")
print(f"MAE: {mae_s:.4f}")
print(f"MSE: {mse_s:.4f}\n")

print(" Çoklu Lineer Regresyon (age, bmi, bp, cholesterol):")
print(f"R² Skoru: {r2_m:.4f}")
print(f"MAE: {mae_m:.4f}")
print(f"MSE: {mse_m:.4f}")
