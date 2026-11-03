"""Database models"""
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    predictions = db.relationship('Prediction', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<User {self.username}>'

class Prediction(db.Model):
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    image_filename = db.Column(db.String(255), nullable=False)
    
    # Static model results
    static_prediction = db.Column(db.String(20), nullable=False)
    static_hc_confidence = db.Column(db.Float, nullable=False)
    static_sz_confidence = db.Column(db.Float, nullable=False)
    
    # Dynamic model results
    dynamic_prediction = db.Column(db.String(20), nullable=False)
    dynamic_hc_confidence = db.Column(db.Float, nullable=False)
    dynamic_sz_confidence = db.Column(db.Float, nullable=False)
    
    models_agree = db.Column(db.Boolean, nullable=False)
    most_attended_window = db.Column(db.Integer, nullable=False)
    
    result_json = db.Column(db.Text, nullable=False)  # JSON string
    result_images = db.Column(db.Text, nullable=True)  # JSON string with base64 images
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Prediction {self.id} - {self.static_prediction}/{self.dynamic_prediction}>'