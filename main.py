from flask import Flask, render_template, request, jsonify
from model import inception_model, preprocess
import torch
import base64
from io import BytesIO
from PIL import Image

# предпочитаемый размер изображения для модели
TARGET_SIZE = (256, 256)

app = Flask(__name__)


def image_to_base64(image: Image.Image) -> str:
    """Преобразует PIL Image в base64-строку для вставки в src."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    encoded = base64.b64encode(img_bytes).decode('utf-8')
    return "data:image/jpeg;base64," + encoded


@app.route('/', methods=['GET'])
def main_route():
    return render_template('./index.html')


@app.route('/upload', methods=['POST'])
def upload():

    file = request.files['image']
    if 'image' not in request.files or file.filename == '':
        return render_template('./index.html', error='Пожалуйста, выберите изображение')


    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')
    image_resized = image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    uploaded_image = image_to_base64(image)

    # preprocess image
    image_tensor = preprocess(image_resized)
    image_tensor = image_tensor.unsqueeze(0)

    # result model
    with torch.no_grad():
        output = inception_model(image_tensor)
        result = int(torch.argmax(output))
        probs = torch.softmax(output, dim=1)
        prob_real = round(probs[0][0].item() * 100, 2)
        prob_fake = round(probs[0][1].item() * 100, 2)
        print(f'{prob_real=}, {prob_fake=}')

    # fake = 1       real = 0


    return render_template('./index.html', result=result, is_image_load=1,
                           uploaded_image=uploaded_image, prob_fake=prob_fake, prob_real=prob_real)


if __name__ == "__main__":
    app.run()
