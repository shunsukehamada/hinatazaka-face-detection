import os
from typing import List, TypedDict, cast

import cv2
import dlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision import models


class DetectData(TypedDict):
    id: int
    img_height: int
    img_width: int
    top: int
    right: int
    bottom: int
    left: int


class TopPrediction(TypedDict):
    predict_class: int
    probability: float
    name: str


class AllPrediction(TypedDict):
    predict_classes: List[int]
    probabilities: List[float]
    names: List[str]


class Prediction(TypedDict):
    predict: TopPrediction
    all: AllPrediction


class DataAndPrediction(TypedDict):
    data: DetectData
    prediction: Prediction


IMAGE_SIZE = (124, 124)
detector_file = "mmod_human_face_detector.dat"
detector_path = os.path.join(os.path.dirname(__file__), "data", detector_file)
cnn_face_detector = dlib.cnn_face_detection_model_v1(detector_path)  # type: ignore


def detect(img: cv2.Mat, detector) -> list[DetectData]:
    faces: cv2.Mat = cnn_face_detector(img, 0)
    detect_data: List[DetectData] = []
    for i, face in enumerate(faces):
        img_h, img_w, _ = img.shape
        rect_top = int(face.rect.top())
        if rect_top < 0:
            rect_top = 0
        rect_bottom = int(face.rect.bottom())
        if rect_bottom > img_h:
            rect_bottom = img_h
        rect_left = int(face.rect.left())
        if rect_left < 0:
            rect_left = 0
        rect_right = int(face.rect.right())
        if rect_right > img_w:
            rect_right = img_w
        detect_data.append(
            {
                "id": i,
                "img_height": img_h,
                "img_width": img_w,
                "top": rect_top,
                "right": rect_right,
                "bottom": rect_bottom,
                "left": rect_left,
            }
        )
    return detect_data


def trim(img: cv2.Mat, detect_data: DetectData) -> cv2.Mat:
    face_img = img[
        int(detect_data["top"]) : int(detect_data["bottom"]),  # noqa:
        int(detect_data["left"]) : int(detect_data["right"]),  # noqa:
    ]
    return face_img


preprocess = T.Compose(
    [
        T.Resize(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Model(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.mobilenet = base_model
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(576, 100)
        self.fc3 = nn.Linear(100, 22)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        x = self.mobilenet(x)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x


mobile_net = models.mobilenet_v3_small()
mobile_net.classifier = nn.Sequential(Identity())

model = Model(mobile_net)
model.load_state_dict(
    torch.load(
        os.path.join(
            os.path.dirname(__file__), "data", "mobilenet_transform_weights.pth"
        )
    )
)
for param in model.parameters():
    param.requires_grad = False
model.eval()

classes = {
    0: "その他",
    1: "上村ひなの",
    2: "丹生明里",
    3: "佐々木久美",
    4: "佐々木美玲",
    5: "加藤史帆",
    6: "宮田愛萌",
    7: "富田鈴花",
    8: "小坂菜緒",
    9: "山口陽世",
    10: "影山優佳",
    11: "東村芽依",
    12: "松田好花",
    13: "森本茉莉",
    14: "河田陽菜",
    15: "潮紗理菜",
    16: "濱岸ひより",
    17: "金村美玖",
    18: "高本彩花",
    19: "高瀬愛奈",
    20: "髙橋未来虹",
    21: "齊藤京子",
}


def opencv2pil(img: cv2.Mat) -> Image.Image:
    new_image = img.copy()
    if new_image.shape[2] == 3:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    new_image = Image.fromarray(new_image)
    return new_image


def predict(
    face: Image.Image,
    transform: T.Compose = preprocess,
    model: Model = model,
    classes: dict[int, str] = classes,
) -> Prediction:
    # 1バッチとして拡張
    tensor: torch.Tensor = transform(face).unsqueeze(0)  # type: ignore
    with torch.no_grad():
        outputs = model.forward(tensor)
    probability: torch.Tensor
    predicted: torch.Tensor
    probability, predicted = torch.max(outputs.data, 1)
    predict_class: int = cast(int, predicted.item())
    all_probabilities: torch.Tensor
    all_classes: torch.Tensor
    all_probabilities, all_classes = torch.topk(outputs.data, 22)
    return {
        "predict": {
            "predict_class": predict_class,
            "probability": probability.item(),
            "name": classes[predict_class],
        },
        "all": {
            "predict_classes": all_classes.tolist(),
            "probabilities": all_probabilities[0].tolist(),
            "names": [classes[i] for i in all_classes[0].tolist()],
        },
    }


def get_predictions(img: cv2.Mat) -> List[DataAndPrediction]:
    detect_data = detect(img, cnn_face_detector)
    predictions: List[DataAndPrediction] = []
    for data in detect_data:
        face = trim(img, data)
        prediction = predict(opencv2pil(face))
        predictions.append({"data": data, "prediction": prediction})
    return predictions
