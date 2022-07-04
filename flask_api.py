import os
from flask import Flask, request
from dotenv import load_dotenv

from api.controllers.image_inference_controller import ImageInferenceController

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

imageInferenceController = ImageInferenceController()


@app.route('/api/v1/get-image-inference', methods=['POST'])
def get_image_inference():
    req = request

    return imageInferenceController.handle_inference_image(req)


@app.route('/')
def index():
    return 'Lisam API'


if __name__ == '__main__':
    load_dotenv()
    HOST = os.getenv('HOST')
    port = int(os.environ.get("PORT", 5000))
    app.run(host=HOST, port=port)
