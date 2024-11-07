import numpy as np
import pandas as pd

traindata = pd.read_csv("/kaggle/input/ml-course-project-2/train.csv")
test_data = pd.read_csv("/kaggle/input/ml-course-project-2/test.csv")

categorical_cols = ["Marka", "Seri", "Model", "Yakıt Tipi", "Vites Tipi", "Renk", "Çekiş", 
                    "Kasa Tipi", "İl", "Garanti", "Plaka Uyruğu", "Arka Kaput", "Arka Tampon", 
                    "Motor Kaputu", "Sağ Arka Çamurluk", "Sağ Ön Çamurluk", "Sağ Arka Kapı", 
                    "Sol Arka Çamurluk", "Sol Ön Çamurluk", "Sol Arka Kapı", "Sol Ön Kapı", 
                    "Sağ Ön Kapı", "Tavan", "Ön Tampon"]

numerical_cols = ["Yıl", "Motor Hacmi", "Motor Gücü", "Kilometre", "Orijinal Sayısı", 
                  "Boyanmış Sayısı", "Değişmiş Sayısı", "Belirsiz Sayısı"]


def normalize(column):
    return (column - column.min()) / (column.max() - column.min())

combined_data = pd.concat([traindata, test_data], axis=0)

combined_data_dummies = pd.get_dummies(combined_data[categorical_cols], dtype=int)

combined_data_numerical = combined_data[numerical_cols].apply(normalize)

combined_data_processed = pd.concat([combined_data_numerical, combined_data_dummies], axis=1)


x_train = combined_data_processed.iloc[:len(traindata), :]
x_test = combined_data_processed.iloc[len(traindata):, :]
price = traindata["Fiyat"].to_numpy()


def compute_cost(w, x, b, y):
    m = len(y)
    predictions = x.dot(w) + b
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

def gradient_descent(x, y, w, b, learning_rate, iterations):
    m = len(y)
    for _ in range(iterations):
        predictions = x.dot(w) + b
        error = predictions - y
        dw = (1 / m) * x.T.dot(error)
        db = (1 / m) * np.sum(error)
        
        
        w -= learning_rate * dw
        b -= learning_rate * db
    return w, b, compute_cost(w, x, b, y)

x_train = x_train.to_numpy()
w_init = np.zeros(x_train.shape[1])
b_init = 0
learning_rate = 0.01
iterations = 10000

w_fin, b_fin, final_cost = gradient_descent(x_train, price, w_init, b_init, learning_rate, iterations)

x_test = x_test.to_numpy()
result = x_test.dot(w_fin) + b_fin

df = pd.DataFrame(result, columns=["Predicted Price"])
df.to_csv("/kaggle/working/deneme3.csv", index=True)
