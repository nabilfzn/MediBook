from flask import Flask, request, jsonify, render_template
import gspread
from oauth2client.service_account import ServiceAccountCredentials

app = Flask(__name__) 

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/chatmedi')
def chatmedi():
    return render_template('chatmedi.html')
@app.route('/chatbook')
def chatbook():
    return render_template('chatbook.html')



if __name__ == "__main__":
    app.run(debug=True, port=5001)