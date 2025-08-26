from flask import Flask, request, jsonify, render_template
import gspread
from oauth2client.service_account import ServiceAccountCredentials

app = Flask(__name__) 

@app.route('/')
def index():
    return render_template('index.html')



if __name__ == "__main__":
    app.run(debug=True, port=5001)