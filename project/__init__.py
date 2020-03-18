from flask import *
import os
app = Flask(__name__)
app.secret_key='abc'
print('inside project init')
import project.com.controller
