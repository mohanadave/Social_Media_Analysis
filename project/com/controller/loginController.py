from flask import *
import os
import datetime
from project import app

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/register', methods=["POST","GET"])
def register():
    return render_template('register.html')