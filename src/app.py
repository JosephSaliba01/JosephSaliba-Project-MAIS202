from flask import Flask, request, render_template
from matplotlib.pyplot import text
from helper import get_prediced_class

app = Flask(__name__)

@app.route('/')
def display_form():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def my_form_post():
    u, g, r, i, z = request.form['u'], request.form['g'] ,request.form['r'], request.form['i'], request.form['z']
    predicted_class = get_prediced_class([u, g, r, i, z])
    return render_template('prediction.html', text=predicted_class)