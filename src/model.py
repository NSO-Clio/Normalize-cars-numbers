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


def rle_decode(coordinates: str, image_shape=(64, 192)) -> np.ndarray:
    points = []
    dt = coordinates.split()
    for i in range(0, len(dt), 2):
        points.append((int(dt[i]), int(dt[i + 1])))
    mask = np.zeros(image_shape, dtype=np.uint8)
    for coord in points:
        mask[coord[1] - 1, coord[0] - 1] = 1
    for i in range(len(points) - 1):
        cv2.line(mask, points[i], points[i + 1], (1, 1, 1), 1)
    cv2.fillPoly(mask, [np.array(points)], (1, 1, 1))
    return mask


class SegmentationDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None, mask_transform=None, train=False) -> None:
        self.dataframe = dataframe
        self.transform = transform
        self.mask_transform = mask_transform
        self.train = train

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file {img_path} not found.")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.train:
            mask_rle = self.dataframe.iloc[idx, 1]
            mask = rle_decode(mask_rle, (image.height, image.width))
            if self.mask_transform:
                mask = self.mask_transform(mask)
            else:
                mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return image, mask if self.train else image


class Metrics:
    @staticmethod
    def f1_score(y_true, y_pred, threshold=0.5):
        y_pred = (y_pred > threshold).float()
        tp = (y_true * y_pred).sum().to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        return f1.item()

    @staticmethod
    def precision(y_true, y_pred, threshold=0.5):
        y_pred = (y_pred > threshold).float()
        tp = (y_true * y_pred).sum().to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
        precision = tp / (tp + fp + 1e-6)
        return precision.item()

    @staticmethod
    def recall(y_true, y_pred, threshold=0.5):
        y_pred = (y_pred > threshold).float()
        tp = (y_true * y_pred).sum().to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
        recall = tp / (tp + fn + 1e-6)
        return recall.item()

    @staticmethod
    def accuracy(y_true, y_pred, threshold=0.5):
        y_pred = (y_pred > threshold).float()
        correct = (y_true == y_pred).float().sum()
        accuracy = correct / y_true.numel()
        return accuracy.item()


class SegmentCarNumber:
    def __init__(self, unet_path: str, encoder_path: str) -> None:
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
    
    def predict(self, image: str | Image.Image, batch_size=None, shuffle=False, num_workers=2) -> np.ndarray:
        self.model.eval()
        if batch_size:
            result_pred = []
            data = pd.DataFrame({'image_path': [os.path.join(image, f) for f in os.listdir(image)]})
            dataset = SegmentationDataset(dataframe=data, transform=self.image_transform)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
            
            for indx, batch in enumerate(tqdm(data_loader)):
                images, _ = batch
                with torch.no_grad():
                    prediction = self.model(images.to(self.device))
                result_pred.extend([i for i in 
                    (torch.sigmoid(prediction).cpu().permute(0, 2, 3, 1).detach().numpy()* 255).astype(np.uint8)
                    ])
            return result_pred
        else:
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
    def perspective_transform_rgb(image_rgb, mask_img) -> np.ndarray:
        _, binary_mask = cv2.threshold(mask_img, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return image_rgb
        
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Ensure that box points are in the correct order (top-left, top-right, bottom-right, bottom-left)
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
        
        return transformed_image_resized
