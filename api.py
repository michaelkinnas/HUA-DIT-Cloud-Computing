from flask import request, Flask
from flask import send_file
from application import classify_CIFAR10
from plotting import classification_result

app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify():
    imagefile = request.data
    result = classify_CIFAR10(imagefile)
    plot = classification_result(result.keys(), result.values())
    return f"<img src='data:image/png;base64,{plot}'/>"