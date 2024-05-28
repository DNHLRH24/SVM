import os
import librosa
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import joblib

# Hàm để trích xuất đặc trưng MFCC từ tệp âm thanh
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs

# Đường dẫn tới các thư mục train và test
train_path = "dataset/train"
test_path = "dataset/test"

# Hàm để tải và trích xuất đặc trưng từ các tệp âm thanh
def load_data(data_path):
    features = []
    labels = []
    for label, animal in enumerate(['cat', 'dog']):
        animal_path = os.path.join(data_path, animal)
        if not os.path.exists(animal_path):
            print(f"Thư mục {animal_path} không tồn tại.")
            continue
        for file_name in os.listdir(animal_path):
            file_path = os.path.join(animal_path, file_name)
            if file_path.endswith('.wav'):
                features.append(extract_features(file_path))
                labels.append(label)
    return np.array(features), np.array(labels)

# Tải và trích xuất đặc trưng từ dữ liệu train và test
X_train, y_train = load_data(train_path)
X_test, y_test = load_data(test_path)

# Kiểm tra xem có dữ liệu trong X_train và X_test không
if len(X_train) == 0 or len(X_test) == 0:
    print("Không tìm thấy tập tin âm thanh nào trong thư mục đào tạo.")
else:
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Khởi tạo mô hình SVM
    model = svm.SVC()

    # Định nghĩa lưới tham số
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf']
    }

    # Khởi tạo GridSearchCV
    grid_search = GridSearchCV(model, param_grid, refit=True, verbose=2, cv=5)
    grid_search.fit(X_train, y_train)

    # In kết quả tốt nhất
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best estimator: {grid_search.best_estimator_}")

    # Lưu mô hình tốt nhất
    joblib.dump(grid_search.best_estimator_, 'best_svm_model.pkl')
    print("Đã lưu file best_svm_model.pkl")

    # Dự đoán trên tập kiểm tra
    y_pred = grid_search.best_estimator_.predict(X_test)

    # Đánh giá mô hình
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print(classification_report(y_test, y_pred, target_names=['cat', 'dog']))
