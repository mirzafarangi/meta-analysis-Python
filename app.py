import os
from flask import Flask, render_template

DEBUG = True
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("metamar.html")


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=DEBUG, host='0.0.0.0', port=port)
