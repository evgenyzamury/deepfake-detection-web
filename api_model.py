from PIL import Image
import torch
from flask import Flask, request, jsonify

from model import inception_model, preprocess

TARGET_SIZE = (256, 256)

api_app = Flask(__name__)


def predict_image(image: Image.Image) -> dict:
    """
    Выполняет предсказание класса изображения с помощью модели.

    Функция принимает изображение PIL.Image, приводит его к RGB,
    изменяет размер до TARGET_SIZE, применяет preprocess, добавляет
    batch-размерность и передаёт изображение в модель inception_model.

    Возвращает словарь с результатом классификации:
    - result: числовой класс, где 0 — real, 1 — fake
    - label: текстовая метка класса, "real" или "fake"
    - prob_real: вероятность принадлежности к классу real в процентах
    - prob_fake: вероятность принадлежности к классу fake в процентах
    """

    # ---- preprocess ----
    image = image.convert("RGB")
    image_resized = image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

    image_tensor = preprocess(image_resized)
    image_tensor = image_tensor.unsqueeze(0)

    # ---- model use ----
    with torch.no_grad():
        output = inception_model(image_tensor)

        result = int(torch.argmax(output))
        probs = torch.softmax(output, dim=1)

        prob_real = round(probs[0][0].item() * 100, 2)
        prob_fake = round(probs[0][1].item() * 100, 2)

    return {
        "result": result,
        "label": "fake" if result == 1 else "real",
        "prob_real": prob_real,
        "prob_fake": prob_fake,
    }


@api_app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        if "image" not in request.files:
            return jsonify({
                "error": "Файл изображения не был отправлен"
            }), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({
                "error": "Пустое имя файла"
            }), 400

        image = Image.open(file.stream).convert("RGB")
        prediction = predict_image(image)

        return jsonify(prediction), 200

    except Exception as e:
        api_app.logger.exception("Ошибка при обработке изображения")
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    api_app.run(host="127.0.0.1", port=8081)