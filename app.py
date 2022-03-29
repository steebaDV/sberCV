from flask import Flask, jsonify, make_response, request

import json
import io
import base64
from PIL import Image
import pandas as pd

from pipeline import Model, Pipeline, transform

pipeline = Pipeline(Model, transform, 'cpu')
breed_list = pipeline.get_breeds_list()

app = Flask(__name__)


def get_preds_df(img):
    probas = pipeline.predict_proba(img)[0]
    rounded_probas = ['%.3f' % proba for proba in probas]
    preds_concat = pd.concat([pd.Series(breed_list), pd.Series(rounded_probas)], axis=1)
    preds_df = pd.DataFrame(data=preds_concat.values, columns=['label', 'probability'])
    return preds_df


def launch_task(b64string):
    try:  # Не придумал как это if-ом обработать
        f = io.BytesIO(base64.b64decode(b64string))
        pil_image = Image.open(f)
        preds_df = get_preds_df(pil_image)
        res_dict = {
            'result': json.loads(preds_df.to_json(orient='records'))
        }
    except:
        res_dict = {'error': 'No images found. Use encoded byte string'}
    return res_dict


@app.route('/classify', methods=['GET'])
def get_task():
    result = launch_task(request.args.get('img'))
    return make_response(jsonify(result), 200)


if __name__ == '__main__':
    app.run(port=5000, debug=False)
