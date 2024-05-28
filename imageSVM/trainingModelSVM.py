import numpy as np
import os
import cv2
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog
import joblib

# Đường dẫn đến thư mục chứa dữ liệu ảnh
train_data_dir = 'dataset/train'
test_data_dir = 'dataset/test'

# Danh sách các lớp (chó và mèo)
classes = ["dogs", "cats"]

# Số lượng ảnh muốn sử dụng từ mỗi lớp
num_images_per_class = 1000

# Khởi tạo các danh sách để lưu trữ dữ liệu và nhãn
X_train = []
y_train = []
X_test = []
y_test = []

# Định nghĩa kích thước ảnh
image_size = (64, 64)  # Kích thước cố định để đảm bảo nhất quán

# Đọc và tiền xử lý dữ liệu huấn luyện
for class_index, class_name in enumerate(classes):
    class_dir = os.path.join(train_data_dir, class_name)
    image_files = os.listdir(class_dir)[:num_images_per_class]
    for image_file in image_files:
        image_path = os.path.join(class_dir, image_file)
        image = cv2.imread(image_path)
        image = cv2.resize(image, image_size)  # Resize ảnh thành kích thước cố định
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Chuyển đổi ảnh sang ảnh grayscale
        
        # Trích xuất đặc trưng HOG từ ảnh
        features = hog(gray_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
        
        X_train.append(features)  # Lưu trữ đặc trưng
        y_train.append(class_index)  # Lưu nhãn của ảnh

# Đọc và tiền xử lý dữ liệu kiểm tra
for class_index, class_name in enumerate(classes):
    class_dir = os.path.join(test_data_dir, class_name)
    image_files = os.listdir(class_dir)[:num_images_per_class]
    for image_file in image_files:
        image_path = os.path.join(class_dir, image_file)
        image = cv2.imread(image_path)
        image = cv2.resize(image, image_size)  # Resize ảnh thành kích thước cố định
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Chuyển đổi ảnh sang ảnh grayscale
        
        # Trích xuất đặc trưng HOG từ ảnh kiểm tra
        features = hog(gray_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
        
        X_test.append(features)  # Lưu trữ đặc trưng
        y_test.append(class_index)  # Lưu nhãn của ảnh

# Chuyển danh sách thành mảng numpy
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Tạo mô hình SVM
svm_model = SVC()

# Thiết lập lựa chọn tham số
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': [0.001, 0.01, 0.1, 1]}

# Tinh chỉnh tham số sử dụng Grid Search Cross Validation
grid_search = GridSearchCV(svm_model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# In ra tham số tốt nhất tìm được
print("Best parameters found:", grid_search.best_params_)

# Dự đoán nhãn của dữ liệu kiểm tra bằng mô hình đã tinh chỉnh
y_pred = grid_search.predict(X_test)

# Đánh giá độ chính xác của mô hình
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Lưu mô hình đã huấn luyện
model_filename = 'svm_dog_cat_model.pkl'
joblib.dump(grid_search, model_filename)
print(f"Model saved to {model_filename}")
