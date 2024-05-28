import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import os
import librosa

# Hàm để trích xuất đặc trưng MFCC từ tệp âm thanh
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs

# Đường dẫn tới thư mục test
test_path = "dataset/test"

# Hàm để tải và trích xuất đặc trưng từ các tệp âm thanh
def load_data(data_path):
    features = []
    labels = []
    file_paths = []
    for label, animal in enumerate(['cat', 'dog']):
        animal_path = os.path.join(data_path, animal)
        if not os.path.exists(animal_path):
            print(f"Danh mục {animal_path} không tồn tại.")
            continue
        for file_name in os.listdir(animal_path):
            file_path = os.path.join(animal_path, file_name)
            if file_path.endswith('.wav'):
                features.append(extract_features(file_path))
                labels.append(label)
                file_paths.append(file_path)
    return np.array(features), np.array(labels), file_paths

# Tải và trích xuất đặc trưng từ dữ liệu test
X_test, y_test, test_file_paths = load_data(test_path)

# Kiểm tra xem có dữ liệu trong X_test không
if len(X_test) == 0:
    print("Không tìm thấy tập tin âm thanh trong thư mục.")
else:
    # Tải mô hình đã lưu
    model = joblib.load('best_svm_model.pkl')
    print("Đã lưu file best_svm_model.pkl")

    # Chuẩn hóa dữ liệu kiểm tra
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)  # Thực hiện chuẩn hóa dữ liệu kiểm tra

    # Dự đoán với mô hình đã tải
    y_pred = model.predict(X_test)

    # Đánh giá mô hình
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print(classification_report(y_test, y_pred, target_names=['cat', 'dog']))

    # In ra nhãn dự đoán cho từng tệp âm thanh
    for file_path, label in zip(test_file_paths, y_pred):
        print(f'File: {file_path} => dự đoán Label: {"cat" if label == 0 else "dog"}')
