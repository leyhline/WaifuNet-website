# -*- coding: utf-8 -*-
"""
A simple Flask web app for showing the results of my WaifuNet deep learning project.

Created on Mon Feb 6 13:59:38 2017

@copyright: 2017 Thomas Leyh
@licence: GPLv3
"""

import numpy as np
import cv2
import os
from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from predict import SimpleConvNet

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config.update({"MODEL":SimpleConvNet(),
                   "ALLOWED_EXTENSIONS":{".png", ".jpg", ".jpeg"}})

def allowed_file(filename):
    return os.path.splitext(filename)[1] in app.config["ALLOWED_EXTENSIONS"]

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/display", methods=["POST"])
def display_image():
    if request.method == "POST" and "image" in request.files:
        img = request.files["image"]
        if img and allowed_file(img.filename):
            img = np.frombuffer(img.read(), dtype=np.int8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            if img is not None:
                plot = app.config["MODEL"].plot_prediction(img)
                return send_file(plot, mimetype='image/png', cache_timeout=1, add_etags=False)
    return redirect(url_for("index"))
