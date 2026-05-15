from flask import Flask, render_template, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import requests

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

    image_bytes = file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    uploaded_image = image_to_base64(image)

    files = {
        'image': (file.filename, BytesIO(image_bytes), file.content_type)
    }

    response = requests.post(
        'http://127.0.0.1:8081/api/predict',
        files=files,
        timeout=60
    )

    data = response.json()
    result = data['result']
    prob_fake = data['prob_fake']
    prob_real = data['prob_real']

    return render_template('./index.html', result=result, is_image_load=1,
                           uploaded_image=uploaded_image, prob_fake=prob_fake, prob_real=prob_real)


if __name__ == "__main__":
    app.run()
