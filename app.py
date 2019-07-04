from flask import Flask, render_template
import sqlite3
from flask import g

DATABASE = 'database/twitter.db'

def get_db():
    db = getattr(g, '_database', None)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('sort_by_sentiment_app.html')

if __name__ == '__main__':
    app.run(debug=True)