from flask import Flask, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from inference import Inference

class Server(object):
    def __init__(self):
        super(Server, self).__init__()
        self.IS_PREDICT,self.IS_DISGNOSIS = False,False
        self.infer = Inference()
        # instantiate the app
        self.app = Flask(__name__, static_folder='dist', template_folder='dist')
        self.app.config.from_object(__name__)
        self.predict_idx = 0
        self.diagnosis_idx = 0
        # enable CORS
        CORS(self.app, resources={r'/*': {'origins': '*'}})
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        @self.app.route('/')
        def serve():
            return send_from_directory(self.app.template_folder, 'index.html')
        @self.app.route('/<path:path>')
        def static_proxy(path):
            try:
                return send_from_directory(self.app.static_folder, path)
            except Exception as e:
                print(f"Error serving {path}: {e}")
                return "File not found", 404
        @self.socketio.on('railway_server')
        def handle_message(message):  # 其中的message是前端传过来的
            print('Railway server received message: ' , message)  # 打印前端传来的信息
            if message=="predict":
                self.predict()
                self.IS_PREDICT = True
            elif message=="diagnosis":
                self.diagnosis()
                self.IS_DISGNOSIS = True
        @self.socketio.on('disconnect')
        def test_disconnect():
            print('Client disconnected:')
            if self.IS_PREDICT:
                self.IS_PREDICT = False
            if self.IS_DISGNOSIS:
                self.IS_DISGNOSIS = False
    def predict(self):
        y_pred,y_true = self.infer.predict(self.predict_idx)
        self.predict_idx += 12
        self.socketio.emit('predict', {'pred':y_pred,'true':y_true, 'index':self.predict_idx})

    def diagnosis(self):
        y_pred,y_true = self.infer.diagnosis(self.diagnosis_idx)
        self.diagnosis_idx += 500
        self.socketio.emit('diagnosis', {'pred':y_pred,'true':y_true})

    def background_thread(self):
        """Example of how to send server generated events to clients."""
        while True:
            if self.IS_PREDICT:
                self.socketio.sleep(3)  # pause
                self.predict()
            if self.IS_DISGNOSIS:
                self.socketio.sleep(3)  # pause
                self.diagnosis()
    def run(self):
        from threading import Thread
        thread = Thread(target=self.background_thread)
        thread.daemon = True  # thread dies when main thread (app thread) dies.
        thread.start()
        self.socketio.run(self.app)

if __name__ == '__main__':
    server = Server()
    server.run()