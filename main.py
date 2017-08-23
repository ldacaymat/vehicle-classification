import os
#import matlab.engine
import tensorflow as tf
import numpy as np
import os
import glob
import numpy as np
import cv2
from uuid import uuid4

from flask import Flask, request, render_template, send_from_directory

#sess = tf.Session()

app = Flask(__name__)
#eng = matlab.engine.start_matlab()

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    target2 = os.path.join(APP_ROOT, 'out/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    if not os.path.isdir(target2):
            os.mkdir(target2)
    else:
        print("Couldn't create upload directory: {}".format(target2))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)

     ###################################   
    #ag="agnes"
    imagePath = 'car.jpg'
    modelFullPath = '/tmp/output_graph.pb'
    labelsFullPath = '/tmp/output_labels.txt'

    answer =None
    strMessage=""
    if not tf.gfile.Exists(imagePath):
        tf.logging.fatal('File does not exist %s', imagePath)
        return answer

    image_data = tf.gfile.FastGFile(imagePath, 'rb').read()

    # Creates graph from saved GraphDef.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-5:][::-1]  # Getting top 5 predictions
        f = open(labelsFullPath, 'rb')
        lines = f.readlines()
        labels = [str(w).replace("\n", "tj") for w in lines]
        labels = [str(w).replace("b'", "") for w in lines]
        labels2 = [str(w).replace("\\n'", "") for w in labels]
        for node_id in top_k:
            human_string = labels2[node_id]
            score = predictions[node_id]
            strMessage=strMessage+('%s (score = %.5f)' % (human_string, score)) + " || "
            
        answer = strMessage
    
    return render_template("complete.html", image_name=filename, answer=answer)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

@app.route('/gallery')
def gallery(name=None):
    return render_template("gallery.html",name=name)
if __name__ == "__main__":
    app.run(port=8080, debug=True)
