from flask import Flask, request

from api.controllers.image_inference_controller import ImageInferenceController

app = Flask(__name__)
imageInferenceController = ImageInferenceController()


@app.route('/api/v1/get-image-inference/', methods=['POST'])
def get_image_inference():
    req = request

    return imageInferenceController.handle_inference_image(req)


if __name__ == '__main__':
    app.run(debug=True)
