import cv2
import predict
from flask import Flask, request
from flask_cors import CORS
from typing import TypedDict, cast
import numpy as np
import base64
import json
import os


class RequestBody(TypedDict):
    image_base: str


PORT = cast(int, os.environ["PORT"]) if "PORT" in os.environ else 5000
app = Flask(__name__)
cors = CORS(app)


@app.after_request
def after_request(response):
    # allowed_origins = ["http://127.0.0.1:5500/"]
    # origin = request.headers.get("Origin")
    # if origin in allowed_origins:
    #     response.headers.add("Access-Control-Allow-Origin", origin)
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Headers", "Authorization")
    response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")  # OPTIONSは必要
    response.headers.add("Access-Control-Allow-Credentials", "true")
    return response


@app.route("/predict", methods=["POST"])
def post_predict() -> str:
    body = cast(RequestBody, request.json)
    img_raw = np.frombuffer(base64.b64decode(body["image_base"]), np.uint8)
    img: cv2.Mat = cv2.imdecode(img_raw, cv2.IMREAD_UNCHANGED)
    return json.dumps(predict.get_predictions(img))  # type:ignore


if __name__ == "__main__":
    app.run(debug=False, port=PORT)
