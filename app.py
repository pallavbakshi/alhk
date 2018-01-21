import os
import datetime
from flask import Flask, render_template, request, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from xgboostmodel import predict, train_model, reset_table
from multiprocessing import Pool

# START HERE ------


# Initialize application
app = Flask(__name__)

# Enabling cors
CORS(app)

# app configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ['DATABASE_URL']
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

DATABASE_URI = os.environ['DATABASE_URL']

# Initialize Flask Sql Alchemy
db = SQLAlchemy(app)

@app.route('/')
def homepage():
    return render_template('index.html')

class Learn(db.Model):
    """
    Class to represent the BucketList model
    """
    __tablename__ = 'learn'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    species = db.Column(db.String(255), nullable=False)
    age = db.Column(db.Numeric(255), nullable=False)
    create_at = db.Column(db.DateTime, nullable=False)
    score = db.Column(db.Numeric(255), nullable=False)

    def __init__(self, species, age, score):
        self.species = species
        self.age = age
        self.create_at = datetime.datetime.utcnow()
        self.score = score

    def save(self):
        """
        Persist a bucket in the database
        :return:
        """
        db.session.add(self)
        db.session.commit()

    def json(self):
        """
        Json representation of the bucket model.
        :return:
        """
        return {
            'id': self.id,
            'species': self.species,
            'age': self.age,
            'createdAt': self.create_at.isoformat(),
            'score': self.score
        }

# db.create_all()


@app.route('/predict/', methods=['POST'])
def send_score():
    """
    Return a score based on the datapoint from the sent json data.
    :return:
    """
    if request.content_type == 'application/json':
        data = request.get_json()
        species = str(data.get('species')).strip()
        age = str(data.get('age')).strip()
        if species and age:
            # call model_predict
            prediction = predict(species.lower(), float(age))
            return presp(str(prediction), 201)
        return presp('Missing some attribute', 400)
    return presp('Content-type must be json', 202)

@app.route('/train/', methods=['POST'])
def train():
    # run training on saved data

    # Async processes req cannot work with heroku free version
    # otherwise I would have used async calls
    # pool = Pool(processes=1)
    # pool.apply_async(train_model, [db, Learn])
    # train_model(db, Learn)
    train_model(db, Learn)
    return response('ok', 201)


@app.route('/reset/', methods=['POST'])
def reset():
    # run training on saved data
    r = reset_table(db, Learn)
    return response(r, 201)

@app.route('/learn/', methods=['POST'])
def create_datapoint():
    """
    Create a datapoint from the sent json data.
    :return:
    """
    if request.content_type == 'application/json':
        data = request.get_json()
        species = str(data.get('species')).strip()
        age = str(data.get('age')).strip()
        score = str(data.get('score')).strip()
        if species and age and score:
            datapoint = Learn(species.lower(), float(age), float(score))
            datapoint.save()
            return response('ok', 201)
        return response('Missing some attribute', 400)
    return response('Content-type must be json', 202)

def presp(mes, stat):
    resp = make_response(jsonify({'score': float(mes)}), stat)
    return resp

def response(mes, stat):
    resp = make_response(mes, stat)
    resp.headers['Content-type'] = 'text'
    return resp

# app.register_blueprint(learn, url_prefix='/v1')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
