from flask import Flask, jsonify, request
from flask_cors import CORS
from inference import Inference

infer = Inference()
# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)
# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

# sanity check route
@app.route('/ping', methods=['GET'])
def ping_pong():
    print(infer.inference(idx=0))
    return jsonify('pong!')


if __name__ == '__main__':
    app.run()