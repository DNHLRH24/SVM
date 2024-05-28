import numpy as np
import os
import cv2
from skimage.feature import hog
import joblib

# Định nghĩa kích thước ảnh
image_size = (64, 64)  # Kích thước cố định để đảm bảo nhất quán

# Tải mô hình đã lưu
model_filename = 'svm_dog_cat_model.pkl'
loaded_model = joblib.load(model_filename)
print("Model loaded from", model_filename)

# Đường dẫn đến thư mục chứa ảnh mới
new_images_dir = 'dataset/imagePredictModel'

# Danh sách các lớp (chó và mèo)
classes = ["dog", "cat"]

# Duyệt qua tất cả các tệp trong thư mục
for image_file in os.listdir(new_images_dir):
    image_path = os.path.join(new_images_dir, image_file)

    # Đọc và tiền xử lý ảnh
    new_image = cv2.imread(image_path)
    
    if new_image is None:
        print(f"Error: Could not read image from {image_path}. Skipping this file.")
        continue

    new_image_resized = cv2.resize(new_image, image_size)  # Resize ảnh thành kích thước cố định
    gray_image = cv2.cvtColor(new_image_resized, cv2.COLOR_BGR2GRAY)  # Chuyển đổi ảnh sang ảnh grayscale

    # Trích xuất đặc trưng HOG từ ảnh mới
    features = hog(gray_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')

    # Chuyển đặc trưng thành mảng numpy và reshape để phù hợp với mô hình
    features = np.array(features).reshape(1, -1)

    # Sử dụng mô hình đã huấn luyện để dự đoán nhãn của ảnh mới
    predicted_class = loaded_model.predict(features)[0]

    # In ra lớp dự đoán
    predicted_label = classes[predicted_class]
    print(f"Image: {image_file}, Predicted class: {predicted_label}")

    # Vẽ nhãn dự đoán lên ảnh
    cv2.putText(new_image, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Hiển thị ảnh với nhãn dự đoán
    cv2.imshow("Predicted Image", new_image)
    cv2.waitKey(0)  # Nhấn phím bất kỳ để tiếp tục hiển thị ảnh tiếp theo

# Đóng tất cả cửa sổ hiển thị
cv2.destroyAllWindows()
