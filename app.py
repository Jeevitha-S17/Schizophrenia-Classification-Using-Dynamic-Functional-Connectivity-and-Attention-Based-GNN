"""
Flask application for Schizophrenia DFC prediction
"""
import os
import sys
import uuid
import json
from datetime import datetime

# Add parent directory to path to import original ML modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import base64
import matplotlib
matplotlib.use("Agg")

from app_config import AppConfig
from models import db, User, Prediction
from inference import InferenceEngine
from utils import save_plot_as_base64, get_result_images, cleanup_old_results

# Initialize extensions
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def create_app():
    """Application factory"""
    app = Flask(__name__)
    app.config.from_object(AppConfig)
    
    # Initialize extensions with app
    db.init_app(app)
    login_manager.init_app(app)
    
    # Create tables
    with app.app_context():
        db.create_all()
        # Initialize inference engine
        app.inference_engine = InferenceEngine()
    
    # Register routes
    register_routes(app)
    
    return app

def register_routes(app):
    """Register all routes"""
    
    @app.route('/')
    def index():
        """Home page"""
        return render_template('index.html')
    
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        """User login"""
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
        
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            user = User.query.filter_by(username=username).first()
            
            if user and check_password_hash(user.password_hash, password):
                login_user(user)
                flash('Logged in successfully!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid username or password.', 'danger')
        
        return render_template('login.html')
    
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        """User registration"""
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
        
        if request.method == 'POST':
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')
            
            # Validation
            if not username or not email or not password:
                flash('All fields are required.', 'danger')
                return render_template('register.html')
            
            if password != confirm_password:
                flash('Passwords do not match.', 'danger')
                return render_template('register.html')
            
            if User.query.filter_by(username=username).first():
                flash('Username already exists.', 'danger')
                return render_template('register.html')
            
            if User.query.filter_by(email=email).first():
                flash('Email already registered.', 'danger')
                return render_template('register.html')
            
            # Create user
            user = User(
                username=username,
                email=email,
                password_hash=generate_password_hash(password)
            )
            db.session.add(user)
            db.session.commit()
            
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        
        return render_template('register.html')
    
    @app.route('/logout')
    @login_required
    def logout():
        """User logout"""
        logout_user()
        flash('Logged out successfully.', 'info')
        return redirect(url_for('index'))
    
    @app.route('/dashboard')
    @login_required
    def dashboard():
        """Dashboard showing user's predictions"""
        page = request.args.get('page', 1, type=int)
        per_page = 10
        
        predictions = Prediction.query.filter_by(user_id=current_user.id)\
            .order_by(Prediction.created_at.desc())\
            .paginate(page=page, per_page=per_page, error_out=False)
        
        print(f"\n\n\n{predictions.items}\n\n\n")
        return render_template('dashboard.html', predictions=predictions)
    
    @app.route('/predict', methods=['GET'])
    @login_required
    def predict_page():
        """Prediction input page"""
        return render_template('predict.html')
    
    @app.route('/predict', methods=['POST'])
    @login_required
    def predict():
        """Handle prediction request"""
        if 'image' not in request.files:
            flash('No file uploaded.', 'danger')
            return redirect(url_for('predict_page'))
        
        file = request.files['image']
        if file.filename == '':
            flash('No file selected.', 'danger')
            return redirect(url_for('predict_page'))
        
        # Check file extension
        allowed_extensions = {'.png', '.jpg', '.jpeg'}
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_extensions:
            flash('Please upload a PNG or JPG image.', 'danger')
            return redirect(url_for('predict_page'))
        
        try:
            # Save uploaded file temporarily
            temp_dir = os.path.join(app.config['TEMP_FOLDER'], str(uuid.uuid4()))
            os.makedirs(temp_dir, exist_ok=True)
            
            filename = secure_filename(file.filename)
            temp_path = os.path.join(temp_dir, filename)
            file.save(temp_path)
            
            # Run prediction
            result = app.inference_engine.predict(temp_path)
            
            # Debug: Log the result structure
            print(f"[app.py] Result contains top_rois_dynamic: {bool(result.get('top_rois_dynamic'))}")
            print(f"[app.py] Result contains top_rois_static: {bool(result.get('top_rois_static'))}")
            if result.get('top_rois_dynamic'):
                print(f"[app.py] top_rois_dynamic sample: {result['top_rois_dynamic'][:3]}")
            
            # Store prediction in database
            prediction = Prediction(
                user_id=current_user.id,
                image_filename=filename,
                static_prediction=result['static_prediction'],
                static_hc_confidence=result['static_confidence']['Healthy'],
                static_sz_confidence=result['static_confidence']['Schizophrenia'],
                dynamic_prediction=result['static_prediction'],
                dynamic_hc_confidence=result['static_confidence']['Healthy'],
                dynamic_sz_confidence=result['static_confidence']['Schizophrenia'],
                models_agree=result['models_agree'],
                most_attended_window=result['most_attended_window'],
                result_json=json.dumps(result)
            )
            db.session.add(prediction)
            db.session.commit()
            
            # Generate images and store as base64
            result_images = get_result_images(result, temp_dir, prediction.id)
            prediction.result_images = json.dumps(result_images)
            db.session.commit()
            
            # Cleanup temp files
            cleanup_old_results(temp_dir)
            
            # Store result in session for display
            result['images'] = result_images
            
            # Debug: Log images generated
            print(f"[app.py] Images generated: {list(result_images.keys())}")
            
            # Return JSON if AJAX request
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify(result)
            
            return render_template('result.html', result=result)
            
        except FileNotFoundError as e:
            flash(f'Model checkpoint not found: {str(e)}. Please run train.py first.', 'danger')
            return redirect(url_for('predict_page'))
        except Exception as e:
            import traceback
            traceback.print_exc()
            flash(f'Prediction error: {str(e)}', 'danger')
            return redirect(url_for('predict_page'))
    
    @app.route('/prediction/<int:prediction_id>')
    @login_required
    def view_prediction(prediction_id):
        """View a specific prediction"""
        prediction = Prediction.query.filter_by(
            id=prediction_id,
            user_id=current_user.id
        ).first_or_404()
        
        result = json.loads(prediction.result_json)
        result['images'] = json.loads(prediction.result_images)
        
        return render_template('result.html', result=result)
    
    @app.route('/prediction/<int:prediction_id>/delete', methods=['POST'])
    @login_required
    def delete_prediction(prediction_id):
        """Delete a prediction"""
        prediction = Prediction.query.filter_by(
            id=prediction_id,
            user_id=current_user.id
        ).first_or_404()
        
        db.session.delete(prediction)
        db.session.commit()
        
        flash('Prediction deleted.', 'success')
        return redirect(url_for('dashboard'))

    
    @app.route('/about')
    def about():
        """About page with project details"""
        return render_template('about.html')

    
    @app.route('/health')
    def health():
        """Health check endpoint"""
        return jsonify({'status': 'ok', 'models_loaded': app.inference_engine.models_loaded})


    


    @app.route('/api/predict', methods=['POST'])
    @login_required
    def api_predict():
        """JSON API for prediction"""
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        try:
            temp_dir = os.path.join(app.config['TEMP_FOLDER'], str(uuid.uuid4()))
            os.makedirs(temp_dir, exist_ok=True)
            
            filename = secure_filename(file.filename)
            temp_path = os.path.join(temp_dir, filename)
            file.save(temp_path)
            
            result = app.inference_engine.predict(temp_path)
            
            # Generate images
            result_images = get_result_images(result, temp_dir)
            result['images'] = result_images
            
            cleanup_old_results(temp_dir)
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)