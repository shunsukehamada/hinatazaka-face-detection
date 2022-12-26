"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
const fileInput = document.getElementById("file");
fileInput === null || fileInput === void 0 ? void 0 : fileInput.addEventListener("change", (e) => __awaiter(void 0, void 0, void 0, function* () {
    const files = fileInput.files;
    if (files) {
        if (files.length === 0) {
            return;
        }
        const file = files[0];
        const imageUrl = yield getImageUrl(file);
        previewImage(imageUrl);
    }
}));
const getImageUrl = (blob) => __awaiter(void 0, void 0, void 0, function* () {
    const reader = new FileReader();
    reader.readAsDataURL(blob);
    reader.addEventListener("error", (e) => {
        var _a;
        console.error((_a = e.target) === null || _a === void 0 ? void 0 : _a.error);
    });
    return new Promise((resolve) => {
        reader.addEventListener("load", () => {
            const result = reader.result;
            resolve(result);
        });
    });
});
const previewImage = (imageUrl) => {
    const imgTag = document.getElementById("preview");
    imgTag.setAttribute("src", imageUrl);
};
const predict = (image_base, controller) => __awaiter(void 0, void 0, void 0, function* () {
    const data = { image_base };
    const res = yield fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
        mode: "cors",
        signal: controller === null || controller === void 0 ? void 0 : controller.signal,
    });
    return (yield res.json());
});
const predictButton = document.getElementById("predict");
predictButton.addEventListener("click", () => __awaiter(void 0, void 0, void 0, function* () {
    const files = fileInput.files;
    if (files) {
        if (files.length === 0) {
            return;
        }
        displayLoader();
        const controller = new AbortController();
        setCancel(controller);
        const file = files[0];
        const imageUrl = yield getImageUrl(file);
        const image_base = imageUrl.replace(/data:.*\/.*;base64,/, "");
        try {
            const predictions = yield predict(image_base, controller);
            if (predictions.length === 0) {
                alert("顔が検出できませんでした");
            }
            setPrediction(predictions, imageUrl);
        }
        finally {
            hideLoader();
        }
    }
}));
const setPrediction = (predictions, imageUrl) => {
    const predictionField = document.getElementById("prediction");
    while (predictionField.firstChild) {
        predictionField.removeChild(predictionField.firstChild);
    }
    predictions.forEach((predict) => {
        const allPredict = predict.prediction.all;
        const top3Name = allPredict.names.slice(0, 3);
        const top3Prob = allPredict.probabilities.slice(0, 3);
        const block = document.createElement("div");
        block.classList.add("predict-block");
        const img = new Image();
        img.src = imageUrl;
        const data = predict.data;
        const croppedCanvas = createCroppedCanvas(img, data.top, data.right, data.bottom, data.left);
        block.appendChild(croppedCanvas);
        for (let i = 0; i < 3; i++) {
            const name = top3Name[i];
            const prob = top3Prob[i];
            const raw = document.createElement("div");
            raw.classList.add("predict-raw");
            const nameSpan = document.createElement("span");
            nameSpan.innerHTML = `${i + 1}. ${name}: `;
            const probSpan = document.createElement("span");
            probSpan.innerHTML = `${(Math.round(prob * 10000) / 100).toFixed(2)}  %`;
            raw.appendChild(nameSpan);
            raw.appendChild(probSpan);
            block.appendChild(raw);
        }
        predictionField.appendChild(block);
    });
};
const createCroppedCanvas = (img, top, right, bottom, left) => {
    const canvas = document.createElement("canvas");
    canvas.width = 150;
    canvas.height = 150;
    const ctx = canvas.getContext("2d");
    const imgHeight = bottom - top;
    const imgWidth = right - left;
    const sx = left - imgWidth * 0.1 >= 0 ? left - imgWidth * 0.1 : 0;
    const sy = top - imgHeight * 0.1 >= 0 ? top - imgHeight * 0.1 : 0;
    ctx === null || ctx === void 0 ? void 0 : ctx.drawImage(img, sx, sy, imgWidth * 1.2, imgHeight * 1.2, 0, 0, 150, 150);
    return canvas;
};
const displayLoader = () => {
    const loader = document.querySelector(".loader-wrap");
    loader.classList.remove("hidden");
};
const hideLoader = () => {
    const loader = document.querySelector(".loader-wrap");
    loader.classList.add("hidden");
};
const setCancel = (controller) => {
    const abortButton = document.getElementById("abort");
    abortButton.onclick = () => {
        hideLoader();
        controller.abort();
    };
};
