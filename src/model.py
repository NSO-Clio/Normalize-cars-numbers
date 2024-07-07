import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from skimage.transform import resize
import segmentation_models_pytorch as smp
from tqdm import tqdm
from glob import glob
import os, cv2
import matplotlib.pyplot as plt


def transform(image: np.ndarray, mas_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Применяет перспективное преобразование к изображению на основе заданных точек.

    Args:
        image (np.ndarray): Входное изображение.
        mas_points (np.ndarray): Массив точек для перспективного преобразования.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Преобразованное изображение и измененное исходное изображение.
    """
    width, height = 512, 112
    image = cv2.resize(image, (192, 64), interpolation=cv2.INTER_LINEAR)
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    matrix = cv2.getPerspectiveTransform(mas_points, pts2)
    result = cv2.warpPerspective(image, matrix, (width, height))
    image = cv2.resize(image, (width, height))
    if image.shape[-1] < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if result.shape[-1] < 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return result, image


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Упорядочивает точки в последовательности: верхний левый, верхний правый, нижний правый, нижний левый.

    Args:
        pts (np.ndarray): Массив точек.

    Returns:
        np.ndarray: Упорядоченные точки.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


class SegmentCarNumber:
    def __init__(self, unet_path: str, encoder_path: str) -> None:
        """
        Инициализирует модель SegmentCarNumber с заданными путями к моделям UNet и энкодеру.

        Args:
            unet_path (str): Путь к контрольной точке модели UNet.
            encoder_path (str): Путь к контрольной точке энкодера.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = smp.Unet(
            encoder_name='resnet34',
            encoder_weights=None,
            classes=1,
            activation=None,
        )

        encoder_state_dict = torch.load(encoder_path, map_location=torch.device(self.device))
        self.model.encoder.load_state_dict(encoder_state_dict)

        checkpoint = torch.load(unet_path, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model = self.model.to(self.device)

        self.image_transform = transforms.Compose([
            transforms.Resize((64, 192)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict_dir(self, dir_imgs: str, output_path: str) -> None:
        """
        Предсказывает и выравнивает изображения из заданного каталога, сохраняет результаты в выходной каталог.

        Args:
            dir_imgs (str): Каталог с входными изображениями.
            output_path (str): Каталог для сохранения выходных изображений.
        """
        self.model.eval()
        path_list = [os.path.join(dir_imgs, f) for f in os.listdir(dir_imgs)]
        ind = 0
        for i in tqdm(path_list):
            image = Image.open(i).convert('RGB')
            image = self.image_transform(image)
            with torch.no_grad():
                prediction = self.model(image.to(self.device).unsqueeze(0))
            prediction = torch.sigmoid(prediction[0]).cpu().permute(1, 2, 0).detach().numpy()
            prediction = (prediction * 255).astype(np.uint8)
            image = cv2.imread(i)
            res_img = self.binary_model(prediction, image)
            cv2.imwrite(os.path.join(output_path, str(ind) + '.jpg'), res_img[0])
            ind += 1

    def predict(self, image: str | Image.Image) -> np.ndarray:
        """
        Предсказывает и выравнивает одно изображение.

        Args:
            image (Union[str, Image.Image]): Путь к изображению или PIL Image.

        Returns:
            np.ndarray: Предсказанная маска для изображения.
        """
        self.model.eval()
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
            image = self.image_transform(image)
        elif isinstance(image, Image.Image):
            image = self.image_transform(image)
        with torch.no_grad():
            prediction = self.model(image.to(self.device).unsqueeze(0))
        prediction = torch.sigmoid(prediction[0]).cpu().permute(1, 2, 0).detach().numpy()
        prediction = (prediction * 255).astype(np.uint8)
        return prediction
    
    @staticmethod
    def perspective_transform_rgb(image_rgb: np.ndarray, mask_img: np.ndarray) -> np.ndarray:
        """
        Применяет перспективное преобразование к данному RGB изображению на основе маски изображения.

        Args:
            image_rgb (np.ndarray): Входное RGB изображение.
            mask_img (np.ndarray): Маска изображения для преобразования.

        Returns:
            np.ndarray: Преобразованное RGB изображение.
        """
        image_rgb = cv2.resize(image_rgb, (192, 64), interpolation=cv2.INTER_LINEAR)
        _, binary_mask = cv2.threshold(mask_img, 200, 255, cv2.THRESH_BINARY)
        binary_mask = cv2.convertScaleAbs(binary_mask)
        try:
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        except:
            print(binary_mask.dtype)  # Should output uint8 (CV_8UC1)
            print(binary_mask.shape)
            plt.imshow(binary_mask)
        if len(contours) == 0:
            return image_rgb
        
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        rect = np.zeros((4, 2), dtype="float32")
        s = box.sum(axis=1)
        rect[0] = box[np.argmin(s)]
        rect[2] = box[np.argmax(s)]
        diff = np.diff(box, axis=1)
        rect[1] = box[np.argmin(diff)]
        rect[3] = box[np.argmax(diff)]
        
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        transformed_image = cv2.warpPerspective(image_rgb, M, (maxWidth, maxHeight))
        transformed_image_resized = cv2.resize(transformed_image, (512, 112))
        if transformed_image_resized.shape[-1] < 3:
            transformed_image_resized = cv2.cvtColor(transformed_image_resized, cv2.COLOR_GRAY2BGR)
        return [transformed_image_resized]
    
    def binary_model(self, image, your_image):
        """
        Применяет бинарное пороговое значение и находит контуры для выполнения перспективного преобразования.

        Args:
            image (np.ndarray): Предсказанная бинарная маска.
            your_image (np.ndarray): Исходное изображение.

        Returns:
            List[np.ndarray]: Список, содержащий преобразованное изображение.
        """
        plug_image = image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output_image = image.copy()
        mas_points = []
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                cv2.drawContours(output_image, [approx], -1, (0, 255, 0), 3)
                for point in approx:
                    mas_points.append(point[0])
        if len(mas_points) == 4:
            mas_points = np.array(mas_points, dtype=np.float32)
            mas_points = order_points(mas_points)
            return transform(your_image, mas_points)
        else:
            return self.perspective_transform_rgb(your_image, plug_image)
