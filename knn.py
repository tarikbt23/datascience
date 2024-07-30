import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, confusion_matrix
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Adım 1: Veri okuma (header = 1 ilk satırı atlamak için)
data = pd.read_excel('C:/Users/ttuna/Downloads/veriler.xls', sheet_name='VERİ_1', engine='xlrd', header=1)

# Veri setindeki ondalıklı değerler virgül ile ayrılmış, noktaya çevirdim
data = data.apply(lambda x: x.astype(str).str.replace(',', '.')).astype(float)

X = data[['P', 'n', 'Ty']] # giriş sütunları
y = data[['Rs', 'R2', 'XS', 'X2', 'RM', 'XM', 'ΔT', 's']] # çıkış sütunları

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Adım 2: KNN modeli oluşturma ve en iyi k değerini bulma
best_k = 1
best_score = float('inf')

for k in range(1, 21):
    knn = KNeighborsRegressor(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
    mean_score = -np.mean(scores)  # MSE negatif değer döner, bu yüzden negatifini alıyoruz
    
    if mean_score < best_score:
        best_score = mean_score
        best_k = k

# En iyi k değeri ile modelin eğitilmesi
knn = KNeighborsRegressor(n_neighbors=best_k)
knn.fit(X_train, y_train)

# Test verisi ile tahmin yapılması
y_pred = knn.predict(X_test)

# Performans metriklerinin raporlanması
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
r2 = r2_score(y_test, y_pred, multioutput='raw_values')

# Sınıflandırma metriklerini hesaplamak için bir eşik değeri belirleniyor
# Burada y_test ve y_pred sürekli değerlerdir. Sınıflandırma için ikili değerlere dönüştürüyoruz.
threshold = 0.5
y_test_binary = (y_test > threshold).astype(int)
y_pred_binary = (y_pred > threshold).astype(int)

# Doğruluk, hassasiyet, özgüllük gibi metriklerin hesaplanması
accuracy = accuracy_score(y_test_binary, y_pred_binary)
precision = precision_score(y_test_binary, y_pred_binary, average='weighted')
recall = recall_score(y_test_binary, y_pred_binary, average='weighted')
conf_matrix = confusion_matrix(y_test_binary.values.argmax(axis=1), y_pred_binary.argmax(axis=1))

# Metrikleri yüzdesel olarak hesapla
accuracy_percent = accuracy * 100
precision_percent = precision * 100
recall_percent = recall * 100

# Tkinter 
def predict():
    try:
        values = [float(entry.get()) for entry in entries]
        user_input = np.array(values).reshape(1, -1)
        prediction = knn.predict(user_input)
        output = "\n".join([f"{col}: {val:.2f}" for col, val in zip(y.columns, prediction[0])])
        
        result_window = tk.Toplevel(root)
        result_window.title("Tahmin Sonuçları")
        result_label = tk.Label(result_window, text=output, font=("Helvetica", 16), justify=tk.LEFT)
        result_label.pack(padx=20, pady=20)
    except ValueError:
        messagebox.showerror("Hata", "Lütfen tüm alanlara geçerli sayılar girin")

def show_metrics():
    metrics_output = (f"Ortalama Kare Hatası (MSE): {mse}\n"
                      f"R2 Skoru: {r2}\n"
                      f"Doğruluk: {accuracy_percent:.2f}%\n"
                      f"Hassasiyet: {precision_percent:.2f}%\n"
                      f"Özgüllük: {recall_percent:.2f}%\n"
                      f"Karışıklık Matrisi:\n{conf_matrix}")
    messagebox.showinfo("Performans Metrikleri", metrics_output)

def plot_metrics():
    labels = ['Doğruluk', 'Hassasiyet', 'Özgüllük']
    values = [accuracy_percent, precision_percent, recall_percent]
    
    fig, ax = plt.subplots()
    bars = ax.bar(labels, values, color=['blue', 'green', 'red'])
    ax.set_ylim(0, 100)
    ax.set_ylabel('Yüzde (%)')
        
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.2f}%", ha='center', va='bottom')
    
    fig.tight_layout()
    
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

root = tk.Tk()
root.title("KNN Tahmin Arayüzü")

feature_names = ['P', 'n', 'Ty']
entries = []

for feature in feature_names:
    label = tk.Label(root, text=f"{feature}")
    label.pack()
    entry = tk.Entry(root)
    entry.pack()
    entries.append(entry)

predict_button = tk.Button(root, text="Tahmin Yap", command=predict)
predict_button.pack()

metrics_button = tk.Button(root, text="Performans Metriklerini Göster", command=show_metrics)
metrics_button.pack()

plot_button = tk.Button(root, text="Metrikleri Görselleştir", command=plot_metrics)
plot_button.pack()

root.mainloop()
