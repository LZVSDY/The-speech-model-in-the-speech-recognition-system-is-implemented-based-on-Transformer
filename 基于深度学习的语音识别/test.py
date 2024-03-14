from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests
import json
from SpeechModel import SpeechModel
from LanguageModel import LanguageModel


app = Flask(__name__)
CORS(app)

@app.route('/recognize', methods=['POST'])
def process_audio():
    if 'audio' in request.files:
        audio_file = request.files['audio']

        # 声学模型
        model = SpeechModel('')
        model.load_model('model_speech/train/weight.ckpt')
        result_pinyin = model.recognize_speech(audio_file)
        # 语言模型
        lm = LanguageModel('model_language/')
        lm.load_model()
        result_chs = lm.speech2text(result_pinyin)
        print(result_chs)

        return result_chs
    else:
        return '未找到音频文件'

if __name__ == '__main__':
    app.run()



