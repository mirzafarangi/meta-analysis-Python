import os
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import math
import copy
import requests
from IPython.display import display, HTML
from statsmodels import api as sm
from sklearn import datasets
import scipy.stats
from decimal import Decimal
import statistics
import random

#r converts
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter






DEBUG = True
app = Flask(__name__)





@app.route('/')
def index():
    return render_template('about.html')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=DEBUG, host='0.0.0.0', port=port)



