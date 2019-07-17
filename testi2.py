from flask import Flask, render_template
import os
import numpy as np
import pickle 

app = Flask(__name__)

# setting up the classifier
cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir,
                   'tweetclassifier_outofcore',
                   'pickled_objects',
                   'classifier.pkl'
                   ), 'rb'))

@app.route('/')
def index():
    return f'<h1>hello world{clf.predict(["lovely!"])}</h1>'


if __name__ == '__main__':
    app.run(debug=True, host=os.getenv('IP', '0.0.0.0'), 
            port=int(os.getenv('PORT', 4445)))