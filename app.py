from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from inference import Inference

infer = Inference()
# instantiate the app
app = Flask(__name__, static_folder='dist', template_folder='dist')
app.config.from_object(__name__)
# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})
@app.route('/')
def serve():
    return send_from_directory(app.template_folder, 'index.html')
@app.route('/<path:path>')
def static_proxy(path):
    try:
        return send_from_directory(app.static_folder, path)
    except Exception as e:
        print(f"Error serving {path}: {e}")
        return "File not found", 404
@app.route('/predict', methods=['GET'])
def predict():
    idx = request.args.get('idx', default=None, type=int)
    y_pred,y_true = infer.inference(idx)
    return jsonify({'pred':y_pred,'true':y_true})
if __name__ == '__main__':
    app.run()