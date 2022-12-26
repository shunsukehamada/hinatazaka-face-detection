type RequestBody = {
    image_base: string | ArrayBuffer | null;
};

type DetectData = {
    id: number;
    img_height: number;
    img_width: number;
    top: number;
    right: number;
    bottom: number;
    left: number;
};

type TopPrediction = { predict_class: number; probability: number; name: string };

type AllPrediction = { predict_classes: number[]; probabilities: number[]; names: string[] };

type Prediction = { predict: TopPrediction; all: AllPrediction };

type PredictResponse = { data: DetectData; prediction: Prediction };

const fileInput = document.getElementById("file") as HTMLInputElement;
fileInput?.addEventListener("change", async (e) => {
    const files = fileInput.files;
    if (files) {
        if (files.length === 0) {
            return;
        }
        const file = files[0];
        const imageUrl = await getImageUrl(file);
        previewImage(imageUrl);
    }
});

const getImageUrl = async (blob: Blob): Promise<string> => {
    const reader = new FileReader();
    reader.readAsDataURL(blob);
    reader.addEventListener("error", (e) => {
        console.error(e.target?.error);
    });
    return new Promise<string>((resolve) => {
        reader.addEventListener("load", () => {
            const result = reader.result as string;
            resolve(result);
        });
    });
};

const previewImage = (imageUrl: string): void => {
    const imgTag = document.getElementById("preview") as HTMLImageElement;
    imgTag.setAttribute("src", imageUrl);
};

const predict = async (
    image_base: string | ArrayBuffer | null,
    controller?: AbortController
): Promise<PredictResponse[]> => {
    const data: RequestBody = { image_base };
    const res = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
        mode: "cors",
        signal: controller?.signal,
    });
    return (await res.json()) as PredictResponse[];
};
const predictButton = document.getElementById("predict") as HTMLDivElement;
predictButton.addEventListener("click", async () => {
    const files = fileInput.files;
    if (files) {
        if (files.length === 0) {
            return;
        }
        displayLoader();
        const controller = new AbortController();
        setCancel(controller);
        const file = files[0];
        const imageUrl = await getImageUrl(file);
        const image_base = imageUrl.replace(/data:.*\/.*;base64,/, "");
        try {
            const predictions = await predict(image_base, controller);
            if (predictions.length === 0) {
                alert("顔が検出できませんでした");
            }
            setPrediction(predictions, imageUrl);
        } finally {
            hideLoader();
        }
    }
});

const setPrediction = (predictions: PredictResponse[], imageUrl: string): void => {
    const predictionField = document.getElementById("prediction") as HTMLDivElement;
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

const createCroppedCanvas = (
    img: HTMLImageElement,
    top: number,
    right: number,
    bottom: number,
    left: number
): HTMLCanvasElement => {
    const canvas = document.createElement("canvas");
    canvas.width = 150;
    canvas.height = 150;
    const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;
    const imgHeight = bottom - top;
    const imgWidth = right - left;
    const sx = left - imgWidth * 0.1 >= 0 ? left - imgWidth * 0.1 : 0;
    const sy = top - imgHeight * 0.1 >= 0 ? top - imgHeight * 0.1 : 0;
    ctx?.drawImage(img, sx, sy, imgWidth * 1.2, imgHeight * 1.2, 0, 0, 150, 150);
    return canvas;
};

const displayLoader = (): void => {
    const loader = document.querySelector(".loader-wrap") as HTMLDivElement;
    loader.classList.remove("hidden");
};

const hideLoader = (): void => {
    const loader = document.querySelector(".loader-wrap") as HTMLDivElement;
    loader.classList.add("hidden");
};

const setCancel = (controller: AbortController): void => {
    const abortButton = document.getElementById("abort") as HTMLDivElement;
    abortButton.onclick = () => {
        hideLoader();
        controller.abort();
    };
};
