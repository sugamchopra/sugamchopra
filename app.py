# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 09:25:34 2021

@author: sugam
"""
import flask
from flask import Flask, request , jsonify, render_template
#import jinja2
import numpy as np
import pandas as pd
import pickle

from flask import Flask, request,jsonify,render_template
import pickle
import pandas as pd
import numpy as np
from flask import Flask , render_template,request
app = Flask(__name__)
filename = 'model.pkl'
model = pickle.load(open(filename, 'rb'))
@app.route('/')
def main():
    return render_template('home.html')


@app.route('/after', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    arr = np.array([[data1, data2, data3]])
    pred = model.predict(arr)
    return render_template('after.html',data=pred)
    
    
@app.route('/after', methods=['POST'])
def home1():
    data1 = request.form['a']
    arr1 = np.array([[data1]])
    pred1 = model.predict(arr1)
    return render_template('after1.html',data=pred1)

