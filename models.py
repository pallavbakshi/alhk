import datetime
from app import db

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
