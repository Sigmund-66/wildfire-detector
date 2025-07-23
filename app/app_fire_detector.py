class FireImagePreprocessor:
    def __init__(self, target_size=(640, 640)):
        self.target_size = target_size

    def resize(self, image):
        return cv2.resize(image, self.target_size)

    def adjust_contrast(self, image, factor=1.2):
        return cv2.convertScaleAbs(image, alpha=factor, beta=0)

    def equalize_histogram(self, image):
        img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    def increase_saturation(self, image, scale=1.3):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] *= scale
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def gaussian_blur(self, image, kernel_size=(3, 3)):
        return cv2.GaussianBlur(image, kernel_size, 0)

    def normalize(self, image):
        return image / 255.0

    def full_pipeline(self, image):
        image = self.resize(image)
        image = self.adjust_contrast(image)
        image = self.equalize_histogram(image)
        image = self.increase_saturation(image)
        image = self.gaussian_blur(image)
        return self.normalize(image)

    def fire_mask_overlay(self, image):
        # Uma máscara simples baseada em tons avermelhados
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower_fire = np.array([0, 100, 100])
        upper_fire = np.array([20, 255, 255])
        mask = cv2.inRange(hsv, lower_fire, upper_fire)

        overlay = image.copy()
        overlay[mask > 0] = [255, 0, 0]  # Destaca em vermelho

        return overlay
        
# Configurações do model e da imagem
model = YOLO("/content/runs/detect/train2/weights/best.pt")
IMAGE_PATH = "/content/pexels-andreas-geissler-320379030-17356102.jpg"

preprocessor = FireImagePreprocessor(target_size=(640, 640))

image = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
preprocessed_image = preprocessor.full_pipeline(image_rgb)
preprocessed_uint8 = (preprocessed_image * 255).astype(np.uint8)

results = model.predict(IMAGE_PATH, save=True, imgsz=736, conf=0.25)[0]

# 5. Verificar classes detectadas
detected_classes = results.boxes.cls.cpu().numpy().astype(int)
class_names = model.names
detected_labels = [class_names[i] for i in detected_classes]

# 6. Mensagem
has_fire = 'fire' in detected_labels
has_smoke = 'smoke' in detected_labels

results.show();

if has_fire and has_smoke:
    print("A imagem contém **fogo e fumaça**.")
elif has_fire:
    print("A imagem contém apenas **fogo**.")
elif has_smoke:
    print("A imagem contém apenas **fumaça**.")
else:
    print("Nenhuma ocorrência de fogo ou fumaça foi detectada na imagem.")
