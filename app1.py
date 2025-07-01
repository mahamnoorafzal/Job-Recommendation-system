from flask import Flask, render_template, request, redirect, url_for, flash, session, g, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_paginate import Pagination
from datetime import datetime
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from bson import ObjectId
from bson.errors import InvalidId
from pymongo import MongoClient, ReturnDocument
from pymongo import errors as pymongo_errors
from dotenv import load_dotenv
from job_prediction import JobPredictionModel
from flask_mail import Mail, Message
import os
import time
import traceback
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Allowed file extensions for resume uploads
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Create app
app = Flask(__name__, 
           static_url_path='/static',
           static_folder='static',
           template_folder='templates')
app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key_here')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['MONGO_URI'] = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')

# Email configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # Or your SMTP server
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER')

# Create upload folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize MongoDB connection with better error handling
print("\nInitializing MongoDB connection...")

# Initialize variables
client = None
db = None
users_collection = None
jobs_collection = None
applications_collection = None
reviews_collection = None
employer_profiles_collection = None
job_seeker_profiles_collection = None
job_model = None

try:
    # Establish MongoDB connection
    client = MongoClient(app.config['MONGO_URI'], serverSelectionTimeoutMS=5000)
    print(f"Using MongoDB URI: {app.config['MONGO_URI']}")
    
    # Force a connection check
    client.admin.command('ping')
    print("MongoDB server ping successful")
    
    # Get database
    db = client['job_portal']
    print(f"Connected to database: job_portal")
    
    # Initialize collections only if connection is successful
    if db is not None:
        users_collection = db.users
        jobs_collection = db.jobs
        applications_collection = db.applications
        reviews_collection = db.reviews
        employer_profiles_collection = db.employer_profiles
        job_seeker_profiles_collection = db.job_seeker_profiles
        
        # Print collection statistics
        print("\nCollection statistics:")
        for collection_name in ['users', 'jobs', 'applications', 'reviews', 'employer_profiles', 'job_seeker_profiles']:
            count = db[collection_name].count_documents({})
            print(f"{collection_name}: {count} documents")
            
    else:
        print("Error: Failed to connect to database")
    
    # Verify collections exist
    collections = db.list_collection_names()
    print(f"Available collections: {collections}")
    
    required_collections = [
        'users', 'jobs', 'applications', 'reviews', 
        'employer_profiles', 'job_seeker_profiles'
    ]
    
    # Create missing collections and their indexes
    for collection_name in required_collections:
        if collection_name not in collections:
            print(f"Creating collection: {collection_name}")
            db.create_collection(collection_name)
            
            # Create indexes based on collection
            if collection_name == 'users':
                db[collection_name].create_index('email', unique=True)
            elif collection_name == 'jobs':
                db[collection_name].create_index([('title', 'text'), ('description', 'text')])
                db[collection_name].create_index('employer_id')
                db[collection_name].create_index('status')
            elif collection_name == 'applications':
                db[collection_name].create_index([('job_id', 1), ('job_seeker_id', 1)], unique=True)
                db[collection_name].create_index('job_id')
                db[collection_name].create_index('job_seeker_id')
            elif collection_name == 'employer_profiles':
                db[collection_name].create_index('user_id', unique=True)
            elif collection_name == 'job_seeker_profiles':
                db[collection_name].create_index('user_id', unique=True)
            elif collection_name == 'reviews':
                db[collection_name].create_index('reviewee_id')
                db[collection_name].create_index('reviewer_id')
    
    print("✓ Successfully connected to MongoDB and initialized collections")
      # Initialize job prediction model
    try:
        from hybrid_job_prediction import HybridJobPredictionModel
        job_model = HybridJobPredictionModel(db)
        print("✓ Successfully loaded hybrid job prediction model")
    except Exception as e:
        print(f"✕ Error initializing hybrid job prediction model: {str(e)}")
        job_model = None
    
except pymongo_errors.ConnectionFailure as e:
    print(f"✕ MongoDB Connection Error: {str(e)}")
    print("Traceback:")
    traceback.print_exc()
    client = None
    db = None
    users_collection = None
    jobs_collection = None
    applications_collection = None
    reviews_collection = None
    employer_profiles_collection = None
    job_seeker_profiles_collection = None
    
except Exception as e:
    print(f"✕ Unexpected error while connecting to MongoDB: {str(e)}")
    print("Traceback:")
    traceback.print_exc()
    client = None
    db = None
    users_collection = None
    jobs_collection = None
    applications_collection = None
    reviews_collection = None
    employer_profiles_collection = None
    job_seeker_profiles_collection = None

# Initialize mail
mail = Mail(app)

# Add connection check middleware
@app.before_request
def check_db_connection():
    if client is None or db is None:
        flash('Database connection is not available. Please try again later.', 'danger')
        return render_template('error.html', 
                            error_message="Database connection is not available. Please try again later.",
                            retry_url=request.referrer)

# Add database connection check middleware
@app.before_request
def check_database_connection():
    """Check if database connection is available before processing requests"""
    if request.path.startswith('/static/'):
        return  # Skip check for static files

    global client, db, users_collection, jobs_collection, applications_collection, reviews_collection, employer_profiles_collection, job_seeker_profiles_collection
    
    # Check each database component
    components = [
        (client, "MongoDB client"),
        (db, "Database"),
        (users_collection, "Users collection"),
        (jobs_collection, "Jobs collection"),
        (applications_collection, "Applications collection"),
        (reviews_collection, "Reviews collection"),
        (employer_profiles_collection, "Employer profiles collection"),
        (job_seeker_profiles_collection, "Job seeker profiles collection")
    ]
    
    for component, name in components:
        if component is None:
            app.logger.error(f"Database component not available: {name}")
            flash('Database connection error. Please try again later.', 'danger')
            return redirect(url_for('error', message=f'Database component not available: {name}'))

# Request logging
@app.before_request
def before_request():
    print(f"\nIncoming request: {request.method} {request.url}")
    print(f"Headers: {dict(request.headers)}")

# Error handling
@app.errorhandler(Exception)
def handle_error(error):
    print(f"Error: {str(error)}")
    return render_template('error.html', 
                         error_message="An unexpected error occurred. Please try again.",
                         retry_url=request.referrer)

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html',
                         error_message="The requested page was not found.",
                         retry_url=url_for('index'))

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html',
                         error_message="An internal server error occurred. Please try again later.",
                         retry_url=request.referrer)

# --- User Class for Flask-Login ---
class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.name = f"{user_data.get('first_name', '')} {user_data.get('last_name', '')}"
        self.email = user_data.get('email', '')
        self.user_type = user_data.get('user_type', '')

@login_manager.user_loader
def load_user(user_id):
    user_data = users_collection.find_one({'_id': ObjectId(user_id)})
    if user_data:
        return User(user_data)
    return None

# --- Constants ---
APPLICATION_STATES = {
    'pending': 'pending',
    'accepted': 'accepted',
    'rejected': 'rejected',
    'withdrawn': 'withdrawn'
}

# --- Routes ---

@app.route('/')
def index():
    # Get featured jobs (most recent with complete information)
    featured_jobs = list(jobs_collection.aggregate([
        {'$match': {'status': 'open'}},
        {'$lookup': {
            'from': 'users',
            'localField': 'employer_id',
            'foreignField': '_id',
            'as': 'employer'
        }},
        {'$unwind': '$employer'},
        {'$lookup': {
            'from': 'employer_profiles',
            'localField': 'employer_id',
            'foreignField': 'user_id',
            'as': 'employer_profile'
        }},
        {'$unwind': '$employer_profile'},
        {'$sort': {'date_posted': -1}},
        {'$limit': 6},
        {'$project': {
            '_id': 1,
            'title': 1,
            'description': 1,
            'salary': 1,
            'location': 1,
            'date_posted': 1,
            'status': 1,
            'company_name': '$employer_profile.company_name',
            'formatted_date': {'$dateToString': {'format': '%Y-%m-%d', 'date': '$date_posted'}}
        }}
    ]))
    
    # Calculate platform statistics
    stats = {
        'total_jobs': jobs_collection.count_documents({'status': 'open'}),
        'total_companies': users_collection.count_documents({'user_type': 'employer'}),
        'total_applicants': users_collection.count_documents({'user_type': 'job_seeker'})
    }
    
    # Overall success rate
    pipeline = [
        {'$group': {
            '_id': None,
            'total_applications': {'$sum': 1},
            'accepted_applications': {'$sum': {'$cond': [{'$eq': ['$status', 'accepted']}, 1, 0]}}
        }}
    ]
    success_data = list(applications_collection.aggregate(pipeline))
    
    if success_data and success_data[0]['total_applications'] > 0:
        stats['success_rate'] = round((success_data[0]['accepted_applications'] / success_data[0]['total_applications']) * 100)
    else:
        stats['success_rate'] = 0
    
    return render_template('index.html', featured_jobs=featured_jobs, stats=stats)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user_data = users_collection.find_one({'email': email})
        
        if user_data and check_password_hash(user_data['password'], password):
            user = User(user_data)
            login_user(user)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        user_type = request.form.get('user_type')

        # Validate required fields
        if not all([first_name, last_name, email, password, confirm_password, user_type]):
            flash('Please fill out all required fields.', 'danger')
            return redirect(url_for('register'))

        # Validate password confirmation
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('register'))

        # Check if email exists
        if users_collection.find_one({'email': email}):
            flash('Email already exists. Please use a different email.', 'danger')
            return redirect(url_for('register'))

        try:
            # Insert user
            user_data = {
                'first_name': first_name,
                'last_name': last_name,
                'email': email,
                'password': generate_password_hash(password),
                'user_type': user_type,
                'created_at': datetime.utcnow()
            }
            
            result = users_collection.insert_one(user_data)
            
            # Create user-type specific profile
            if user_type == 'employer':
                db.employer_profiles.insert_one({
                    'user_id': result.inserted_id,
                    'company_name': '',
                    'company_description': '',
                    'industry': '',
                    'company_size': ''
                })
            elif user_type == 'job_seeker':
                # Handle CV upload
                cv_path = None
                if 'cv' in request.files:
                    cv_file = request.files['cv']
                    if cv_file and allowed_file(cv_file.filename):
                        filename = secure_filename(f"{result.inserted_id}_{cv_file.filename}")
                        cv_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        cv_file.save(cv_path)

                # Create job seeker profile with enhanced information
                job_seeker_profile = {
                    'user_id': result.inserted_id,
                    'skills': [skill.strip() for skill in request.form.get('skills', '').split(',') if skill.strip()],
                    'experience_years': float(request.form.get('experience_years', 0)),
                    'education_level': request.form.get('education_level', 'high_school'),                    'cv_path': cv_path,
                    'created_at': datetime.utcnow()
                }
                
                db.job_seeker_profiles.insert_one(job_seeker_profile)
            
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        
        except Exception as e:
            logger.error(f"Registration error: {str(e)}")
            flash(f'Registration failed: {str(e)}', 'danger')
            return redirect(url_for('register'))
            
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        
        except Exception as e:
            flash('An error occurred during registration.', 'danger')
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():    
    try:
        print("\n=== Dashboard Access ===")
        print(f"User: {current_user.id} (Type: {current_user.user_type})")
        
        if current_user.user_type == 'job_seeker':
            # Get recent applications with formatted dates
            applications = list(applications_collection.aggregate([
                {'$match': {'job_seeker_id': ObjectId(current_user.id)}},
                {'$sort': {'date_applied': -1}},
                {'$limit': 5},
                {'$lookup': {
                    'from': 'jobs',
                    'localField': 'job_id',
                    'foreignField': '_id',
                    'as': 'job'
                }},
                {'$unwind': '$job'},
                {'$lookup': {
                    'from': 'users',
                    'localField': 'job.employer_id',
                    'foreignField': '_id',
                    'as': 'employer'
                }},
                {'$unwind': '$employer'},
                {'$lookup': {
                    'from': 'employer_profiles',
                    'localField': 'job.employer_id',
                    'foreignField': 'user_id',
                    'as': 'employer_profile'
                }},
                {'$unwind': '$employer_profile'},
                {'$project': {
                '_id': 1,
                'status': 1,
                'date_applied': {'$ifNull': ['$date_applied', '$$NOW']},
                'formatted_date': {'$dateToString': {'format': '%Y-%m-%d', 'date': {'$ifNull': ['$date_applied', '$$NOW']}}},
                'job_title': '$job.title',
                'company_name': '$employer_profile.company_name'
            }}
            ]))
            
            # Get application statistics
            stats_pipeline = [
                {'$match': {'job_seeker_id': ObjectId(current_user.id)}},
                {'$group': {
                    '_id': None,
                    'total_applications': {'$sum': 1},
                    'accepted_applications': {'$sum': {'$cond': [{'$eq': ['$status', 'accepted']}, 1, 0]}}
                }}
            ]
            
            stats = list(applications_collection.aggregate(stats_pipeline))
            if stats:
                stats = stats[0]
            else:                stats = {
                    'total_applications': 0,
                    'accepted_applications': 0,
                    'avg_rating': 0
                }

            # Get average rating from reviews
            rating_pipeline = [
                {'$match': {'reviewee_id': ObjectId(current_user.id)}},
                {'$group': {
                    '_id': None,
                    'avg_rating': {'$avg': '$rating'}
                }}
            ]
            ratings = list(reviews_collection.aggregate(rating_pipeline))
            if ratings:
                stats['avg_rating'] = ratings[0].get('avg_rating', 0)
            else:
                stats['avg_rating'] = 0
            
            return render_template('dashboard.html', applications=applications, stats=stats)
        
        else:
            # Employer dashboard            print("\nRetrieving employer's posted jobs...")
            posted_jobs = list(jobs_collection.aggregate([
                {'$match': {'employer_id': ObjectId(current_user.id)}},
                {'$lookup': {
                    'from': 'applications',
                    'localField': '_id',
                    'foreignField': 'job_id',
                    'as': 'applications'
                }},                {'$project': {
                    '_id': {'$toString': '$_id'},  # Convert ObjectId to string
                    'title': 1,
                    'status': 1,
                    'date_posted': 1,
                    'application_count': {'$size': '$applications'},
                    'formatted_date': {'$dateToString': {'format': '%Y-%m-%d', 'date': '$date_posted'}}
                }},
                {'$sort': {'date_posted': -1}}
            ]))
            
            # Employer stats pipeline
            stats_pipeline = [
                {'$match': {'employer_id': ObjectId(current_user.id)}},
                {'$group': {
                    '_id': None,
                    'active_jobs': {'$sum': 1},
                    'open_jobs': {'$sum': {'$cond': [{'$eq': ['$status', 'open']}, 1, 0]}}
                }}
            ]
            
            stats = list(jobs_collection.aggregate(stats_pipeline))
            
            # Get total applications
            apps_pipeline = [
                {'$match': {'employer_id': ObjectId(current_user.id)}},
                {'$group': {
                    '_id': None,
                    'total_applications': {'$sum': 1}
                }}
            ]
            
            apps_stats = list(applications_collection.aggregate(apps_pipeline))
            
            # Get company rating
            rating_pipeline = [
                {'$match': {'reviewee_id': ObjectId(current_user.id)}},
                {'$group': {
                    '_id': None,
                    'company_rating': {'$avg': '$rating'}
                }}
            ]
            
            ratings = list(reviews_collection.aggregate(rating_pipeline))
            
            # Combine all stats
            final_stats = {
                'active_jobs': stats[0]['active_jobs'] if stats else 0,
                'open_jobs': stats[0]['open_jobs'] if stats else 0,
                'total_applications': apps_stats[0]['total_applications'] if apps_stats else 0,
                'company_rating': ratings[0]['company_rating'] if ratings else 0
            }
            
            print(f"\nFound {len(posted_jobs)} jobs")
            for job in posted_jobs:
                print(f"Job: {job['title']} - ID: {job['_id']}")
            
            return render_template('dashboard.html', posted_jobs=posted_jobs, stats=final_stats)
    
    except Exception as e:
        print(f"Error in dashboard: {str(e)}")
        import traceback
        traceback.print_exc()
        flash('Error loading dashboard data. Please try again.', 'danger')
        return redirect(url_for('index'))

@app.route('/jobs')
def jobs():
    try:
        page = request.args.get('page', 1, type=int)
        per_page = 10
        skip = (page - 1) * per_page

        # Get filter parameters
        search = request.args.get('search', '')
        min_salary = request.args.get('min_salary', type=int)
        location = request.args.get('location', '')
        sort_by = request.args.get('sort_by', 'date')
        
        # Build query
        query = {'status': 'open'}
        
        if search:
            query['$or'] = [
                {'title': {'$regex': search, '$options': 'i'}},
                {'description': {'$regex': search, '$options': 'i'}}
            ]
        
        if min_salary:
            query['salary'] = {'$gte': min_salary}
        
        if location:
            query['location'] = {'$regex': location, '$options': 'i'}
        
        # Determine sort
        if sort_by == 'salary':
            sort = [('salary', -1)]
        else:  # Default to date
            sort = [('date_posted', -1)]        # Convert ObjectId strings to ObjectIds in the query if needed
        if '_id' in query and isinstance(query['_id'], str):
            query['_id'] = ObjectId(query['_id'])
        if 'employer_id' in query and isinstance(query['employer_id'], str):
            query['employer_id'] = ObjectId(query['employer_id'])        # Get jobs with company information
        jobs = list(jobs_collection.aggregate([
            {'$match': query},
            {'$sort': dict(sort)},
            {'$skip': skip},
            {'$limit': per_page},
            {'$lookup': {
                'from': 'employer_profiles',
                'localField': 'employer_id',
                'foreignField': 'user_id',
                'as': 'employer_profile'
            }},
            {'$unwind': {'path': '$employer_profile', 'preserveNullAndEmptyArrays': True}},
            {'$project': {
                '_id': {'$toString': '$_id'},
                'title': 1,
                'description': 1,
                'salary': 1,
                'location': 1,
                'date_posted': 1,
                'employer_id': {'$toString': '$employer_id'},
                'company_name': {'$ifNull': ['$employer_profile.company_name', 'Company Not Available']},
                'formatted_date': {'$dateToString': {'format': '%Y-%m-%d', 'date': '$date_posted'}}
            }}
        ]))

        # Debug print
        print(f"Found {len(jobs)} jobs")
        for job in jobs:
            print(f"Job: {job['title']} - ID: {job['_id']}")

        # Count total jobs for pagination
        total_jobs = jobs_collection.count_documents(query)
        
        # Create pagination object
        pagination = Pagination(
            page=page,
            per_page=per_page,
            total=total_jobs,
            record_name='jobs',
            css_framework='bootstrap5'
        )

        # Print debug info about pagination
        print(f"Pagination: Page {page}, Per page {per_page}, Total {total_jobs}")

        return render_template('jobs.html', jobs=jobs, pagination=pagination)
        
    except pymongo_errors.ConnectionFailure as e:
        print(f"Database connection error: {str(e)}")
        flash('Unable to connect to database. Please try again later.', 'danger')
        return render_template('jobs.html', jobs=[], pagination=None)

    except Exception as e:
        print(f"Error in jobs route: {str(e)}")
        traceback.print_exc()  # Print full traceback for debugging
        flash('Error loading jobs.', 'danger')
        return render_template('jobs.html', jobs=[], pagination=None)

@app.route('/job/<job_id>')
def job_details(job_id):
    try:
        print(f"Received job_id: {job_id}")
        
        # Convert string ID to ObjectId and validate in one step
        try:
            job_object_id = ObjectId(job_id)
            print(f"Successfully converted to ObjectId: {job_object_id}")
        except (InvalidId, TypeError) as e:
            print(f"Error converting to ObjectId: {str(e)}")
            flash('Invalid job ID format.', 'danger')
            return redirect(url_for('jobs'))
            
        # Get job details with company information
        job = list(jobs_collection.aggregate([
            {'$match': {'_id': job_object_id}},
            {'$lookup': {
                'from': 'users',
                'localField': 'employer_id',
                'foreignField': '_id',
                'as': 'employer'
            }},
            {'$unwind': {'path': '$employer', 'preserveNullAndEmptyArrays': True}},
            {'$lookup': {
                'from': 'employer_profiles',
                'localField': 'employer_id',
                'foreignField': 'user_id',
                'as': 'employer_profile'
            }},
            {'$unwind': {'path': '$employer_profile', 'preserveNullAndEmptyArrays': True}},
            {'$project': {
                '_id': {'$toString': '$_id'},
                'title': 1,
                'description': 1,
                'salary': 1,
                'location': 1,
                'status': 1,
                'date_posted': 1,
                'employer_id': {'$toString': '$employer_id'},
                'company_name': {'$ifNull': ['$employer_profile.company_name', 'Company Not Available']},
                'company_description': '$employer_profile.company_description',
                'industry': '$employer_profile.industry',
                'company_size': '$employer_profile.company_size',
                'formatted_date': {'$dateToString': {'format': '%Y-%m-%d', 'date': '$date_posted'}}
            }}
        ]))

        if not job:
            print(f"No job found with ID: {job_id}")
            flash('Job not found.', 'danger')
            return redirect(url_for('jobs'))

        job = job[0]  # Get first (and only) job from aggregation result
        print(f"Found job: {job['title']}")
        
        # Get application status if user is logged in and is a job seeker
        application_status = None
        application_date = None
        has_applied = False
        prediction_score = None
        
        if current_user.is_authenticated and current_user.user_type == 'job_seeker':
            # Check current application status
            current_application = applications_collection.find_one({
                'job_id': job_object_id,
                'job_seeker_id': ObjectId(current_user.id)
            })
            
            if current_application:
                has_applied = True
                application_status = current_application.get('status')
                if current_application.get('date_applied'):
                    application_date = current_application['date_applied'].strftime('%Y-%m-%d')
                    
            # Get prediction score if not applied yet and model is available
            if not has_applied and job_model:
                try:
                    cover_letter = request.args.get('cover_letter', '')
                    prediction = job_model.predict(str(current_user.id), job_id, cover_letter)
                    if prediction:
                        prediction_score = {
                            'overall': prediction['success_probability'],
                            'skills': prediction['skills_match'],
                            'experience': prediction['experience_match'],
                            'education': prediction['education_match'],
                            'cv': prediction.get('cv_relevance', 0.0),
                            'cover_letter': prediction.get('cover_letter_match', 0.0),
                            'matched_skills': prediction.get('matched_skills', []),
                            'missing_skills': prediction.get('missing_skills', [])
                        }
                        print(f"Prediction score: {prediction_score['overall'] * 100}%")
                except Exception as e:
                    print(f"Error getting prediction: {str(e)}")
                    traceback.print_exc()
                    prediction_score = None

        # Initialize variables that might not be set in some code paths
        similar_jobs = []
        reviews = []

        # Get similar jobs regardless of user authentication
        similar_jobs = list(jobs_collection.aggregate([
            {'$match': {
                '_id': {'$ne': job_object_id},
                'status': 'open',
                '$or': [
                    {'title': {'$regex': job['title'], '$options': 'i'}},
                    {'description': {'$regex': job['title'], '$options': 'i'}}
                ]
            }},
            {'$limit': 3},
            {'$lookup': {
                'from': 'employer_profiles',
                'localField': 'employer_id',
                'foreignField': 'user_id',
                'as': 'employer_profile'
            }},
            {'$unwind': {'path': '$employer_profile', 'preserveNullAndEmptyArrays': True}},
            {'$project': {
                '_id': {'$toString': '$_id'},
                'title': 1,
                'company_name': {'$ifNull': ['$employer_profile.company_name', 'Company Not Available']},
                'salary': 1
            }}
        ]))

        # Get company reviews if there's an employer_id
        if job.get('employer_id'):
            try:
                employer_id = ObjectId(job['employer_id'])
                reviews = list(reviews_collection.find({
                    'reviewee_id': employer_id
                }).sort('date', -1).limit(5))
            except (InvalidId, TypeError):
                # Keep reviews as empty list
                pass
                
        return render_template('job_details.html',
                            job=job,
                            has_applied=has_applied,
                            application_status=application_status,
                            application_date=application_date,
                            similar_jobs=similar_jobs,
                            prediction_score=prediction_score,
                            reviews=reviews)
                            
    except pymongo_errors.ConnectionError as e:
        print(f"MongoDB connection error: {str(e)}")
        flash('Database connection error. Please try again later.', 'danger')
        return redirect(url_for('jobs'))
    except Exception as e:
        print(f"Error in job_details: {str(e)}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        flash('Error loading job details.', 'danger')
        return redirect(url_for('jobs'))

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    try:
        if request.method == 'POST':
            # Handle profile update
            user_id = ObjectId(current_user.id)
            first_name = request.form.get('first_name')
            last_name = request.form.get('last_name')
            email = request.form.get('email')
            
            # Update basic user information
            users_collection.update_one(
                {'_id': user_id},
                {'$set': {
                    'first_name': first_name,
                    'last_name': last_name,
                    'email': email
                }}
            )
            
            # Update user type specific information
            if current_user.user_type == 'job_seeker':
                # Process skills as a list
                skills = request.form.get('skills', '').split(',')
                skills = [skill.strip() for skill in skills if skill.strip()]
                
                # Get other form fields
                education = request.form.get('education')
                years_experience = int(request.form.get('years_experience', 0))
                industry_preferences = request.form.getlist('industry_preferences[]')
                experience_details = request.form.get('experience_details')
                resume = request.files.get('resume')
                
                # Handle resume upload if provided
                update_data = {
                    'skills': skills,
                    'education': education,
                    'years_experience': years_experience,
                    'industry_preferences': industry_preferences,
                    'experience_details': experience_details
                }
                
                if resume and resume.filename:
                    if allowed_file(resume.filename):
                        filename = secure_filename(resume.filename)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"{current_user.id}_{timestamp}_{filename}"
                        resume.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                        update_data['resume_link'] = filename
                    else:
                        flash('Invalid file type. Please upload a PDF or Word document.', 'danger')
                
                db.job_seeker_profiles.update_one(
                    {'user_id': user_id},
                    {'$set': update_data},
                    upsert=True
                )
            
            elif current_user.user_type == 'employer':
                company_name = request.form.get('company_name')
                company_description = request.form.get('company_description')
                industry = request.form.get('industry')
                company_size = request.form.get('company_size')
                
                db.employer_profiles.update_one(
                    {'user_id': user_id},
                    {'$set': {
                        'company_name': company_name,
                        'company_description': company_description,
                        'industry': industry,
                        'company_size': company_size
                    }},
                    upsert=True
                )
            
            flash('Profile updated successfully!', 'success')
            return redirect(url_for('profile'))
        
        # Get user data for display
        user_data = users_collection.find_one({'_id': ObjectId(current_user.id)})
        
        if current_user.user_type == 'job_seeker':
            profile_data = db.job_seeker_profiles.find_one({'user_id': ObjectId(current_user.id)})
            
            # Get application stats
            pipeline = [
                {'$match': {'job_seeker_id': ObjectId(current_user.id)}},
                {'$group': {
                    '_id': None,
                    'total': {'$sum': 1},
                    'accepted': {'$sum': {'$cond': [{'$eq': ['$status', 'accepted']}, 1, 0]}}
                }}
            ]
            stats = list(applications_collection.aggregate(pipeline))
            
            success_rate = 0
            if stats and stats[0]['total'] > 0:
                success_rate = (stats[0]['accepted'] / stats[0]['total']) * 100
        else:
            profile_data = db.employer_profiles.find_one({'user_id': ObjectId(current_user.id)})
            
            # Get hiring stats
            pipeline = [
                {'$match': {'employer_id': ObjectId(current_user.id)}},
                {'$lookup': {
                    'from': 'applications',
                    'localField': '_id',
                    'foreignField': 'job_id',
                    'as': 'applications'
                }},
                {'$unwind': '$applications'},
                {'$group': {
                    '_id': None,
                    'total': {'$sum': 1},
                    'hired': {'$sum': {'$cond': [{'$eq': ['$applications.status', 'accepted']}, 1, 0]}}
                }}
            ]
            stats = list(jobs_collection.aggregate(pipeline))
            
            hire_rate = 0
            if stats and stats[0]['total'] > 0:
                hire_rate = (stats[0]['hired'] / stats[0]['total']) * 100
        
        # Get reviews
        reviews = list(reviews_collection.aggregate([
            {'$match': {'reviewee_id': ObjectId(current_user.id)}},
            {'$lookup': {
                'from': 'users',
                'localField': 'reviewer_id',
                'foreignField': '_id',
                'as': 'reviewer'
            }},
            {'$unwind': '$reviewer'},
            {'$project': {
                '_id': 1,
                'rating': 1,
                'review_text': 1,
                'reviewer_name': {'$concat': ['$reviewer.first_name', ' ', '$reviewer.last_name']},
                'review_date': {'$dateToString': {'format': '%Y-%m-%d', 'date': '$review_date'}}
            }},
            {'$sort': {'review_date': -1}},
            {'$limit': 5}
        ]))
        
        # Calculate average rating
        rating_data = list(reviews_collection.aggregate([
            {'$match': {'reviewee_id': ObjectId(current_user.id)}},
            {'$group': {
                '_id': None,
                'avg_rating': {'$avg': '$rating'},
                'review_count': {'$sum': 1}
            }}
        ]))
        
        if rating_data:
            rating_data = rating_data[0]
            avg_rating = rating_data['avg_rating']
            review_count = rating_data['review_count']
        else:
            avg_rating = 0
            review_count = 0
        
        return render_template('profile.html',
                            user_data=user_data,
                            profile_data=profile_data,
                            reviews=reviews,
                            success_rate=success_rate if current_user.user_type == 'job_seeker' else None,
                            hire_rate=hire_rate if current_user.user_type == 'employer' else None,
                            avg_rating=avg_rating,
                            review_count=review_count)
    
    except Exception as e:
        print(f"Error in profile: {str(e)}")
        traceback.print_exc()
        flash('Error loading profile.', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/post-job', methods=['GET', 'POST'])
@login_required
def post_job():
    if current_user.user_type != 'employer':
        flash('Only employers can post jobs', 'danger')
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        try:
            # Get form data
            title = request.form.get('title')
            description = request.form.get('description')
            salary = float(request.form.get('salary'))
            location = request.form.get('location')
            required_skills = request.form.get('required_skills', '').split(',')
            required_skills = [skill.strip() for skill in required_skills if skill.strip()]
            experience_level = int(request.form.get('experience_level', 0))
            education = request.form.get('education')
            industry = request.form.get('industry')
            
            # Convert employer_id to ObjectId safely
            try:
                employer_id = ObjectId(current_user.id)
            except InvalidId as e:
                print(f"Error: Invalid user ID format: {str(e)}")
                flash('Error posting job: Invalid user ID format', 'danger')
                return redirect(url_for('post_job'))
                
            # Create job data with ML-required fields
            job_data = {
                'employer_id': employer_id,
                'title': title,
                'description': description,
                'salary': salary,
                'location': location,
                'date_posted': datetime.utcnow(),
                'status': 'open',
                'required_skills': required_skills,
                'experience_level': experience_level,
                'education': education,
                'industry': industry
            }
            
            # Insert the job
            result = jobs_collection.insert_one(job_data)
            if result.inserted_id:
                flash('Job posted successfully!', 'success')
                return redirect(url_for('job_details', job_id=str(result.inserted_id)))
            else:
                flash('Error posting job: Job creation failed', 'danger')
                return redirect(url_for('post_job'))
            
        except ValueError as e:
            print(f"Error: Invalid input format: {str(e)}")
            flash('Please check your input values and try again', 'danger')
            return redirect(url_for('post_job'))
            
        except Exception as e:
            print(f"Error posting job: {str(e)}")
            flash('Error posting job. Please try again.', 'danger')
            return redirect(url_for('post_job'))
    
    return render_template('post_job.html')

@app.route('/manage-applications/<job_id>', endpoint='manage_applications_job')
@login_required
def manage_applications_job(job_id):
    try:
        print("\n====== Manage Applications for Job ======")
        print(f"Job ID: {job_id}")
        print(f"User ID: {current_user.id}")
        print(f"User Type: {current_user.user_type}")
        
        # Check if MongoDB connection is available
        if client is None or db is None:
            raise pymongo_errors.ConnectionFailure("MongoDB connection is not available")
            
        if current_user.user_type != 'employer':
            print("✕ Error: User is not an employer")
            flash('Only employers can manage applications.', 'warning')
            return redirect(url_for('dashboard'))
            
        # Verify job exists and belongs to current employer
        try:
            job_object_id = ObjectId(job_id)
            employer_object_id = ObjectId(current_user.id)
            
            job = jobs_collection.find_one({
                '_id': job_object_id,
                'employer_id': employer_object_id
            })
            
            if not job:
                print("✕ Error: Job not found or unauthorized access")
                flash('Job not found or you do not have permission to manage its applications.', 'danger')
                return redirect(url_for('dashboard'))
            
            # Fetch applications with predictions and applicant details
            applications = list(applications_collection.aggregate([
                {'$match': {'job_id': job_object_id}},
                {'$lookup': {
                    'from': 'users',
                    'localField': 'job_seeker_id',
                    'foreignField': '_id',
                    'as': 'applicant'
                }},
                {'$unwind': {'path': '$applicant', 'preserveNullAndEmptyArrays': True}},
                {'$lookup': {
                    'from': 'job_seeker_profiles',
                    'localField': 'job_seeker_id',
                    'foreignField': 'user_id',
                    'as': 'applicant_profile'
                }},
                {'$unwind': {'path': '$applicant_profile', 'preserveNullAndEmptyArrays': True}},
                {'$project': {
                    '_id': {'$toString': '$_id'},
                    'applicant_name': {
                        '$concat': [
                            {'$ifNull': ['$applicant.first_name', '']},
                            ' ',
                            {'$ifNull': ['$applicant.last_name', '']}
                        ]
                    },
                    'applicant_email': '$applicant.email',
                    'cover_letter': 1,
                    'date_applied': 1,
                    'status': 1,
                    'prediction_score': 1,
                    'skills_match': 1,
                    'experience_match': 1,
                    'education_match': 1,
                    'industry_match': 1,
                    'formatted_date': {'$dateToString': {'format': '%Y-%m-%d', 'date': '$date_applied'}}
                }},
                {'$sort': {
                    'prediction_score': -1,
                    'date_applied': -1
                }}
            ]))
            
            # Convert the job ID to string for the template
            job['_id'] = str(job['_id'])
            job['employer_id'] = str(job['employer_id'])
            
            return render_template('manage_applications.html', 
                                job=job, 
                                applications=applications)
                                
        except InvalidId:
            print("✕ Error: Invalid job ID format")
            flash('Invalid job ID format.', 'danger')
            return redirect(url_for('dashboard'))
            
    except Exception as e:
        print(f"✕ Unexpected error in manage_applications: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        flash('An unexpected error occurred. Please try again.', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/my-applications')
@login_required
def my_applications():
    print("\n=== Loading My Applications ===")
    print(f"User: {current_user.id} (Type: {current_user.user_type})")

    if current_user.user_type != 'job_seeker':
        print("Error: User is not a job seeker")
        flash('Only job seekers can view applications', 'warning')
        return redirect(url_for('dashboard'))
    
    try:
        # Get all applications with job and company details
        applications = list(applications_collection.aggregate([
            {'$match': {'job_seeker_id': current_user.id}},
            {'$lookup': {
                'from': 'jobs',
                'localField': 'job_id',
                'foreignField': '_id',
                'as': 'job'
            }},
            {'$unwind': {
                'path': '$job',
                'preserveNullAndEmptyArrays': True  # Keep applications even if job is deleted
            }},
            {'$lookup': {
                'from': 'employer_profiles',
                'localField': 'job.employer_id',
                'foreignField': 'user_id',
                'as': 'employer'
            }},
            {'$unwind': {
                'path': '$employer',
                'preserveNullAndEmptyArrays': True  # Keep applications even if employer profile is missing
            }},
            {'$sort': {'date_applied': -1}},  # Most recent first
            {'$project': {
                '_id': {'$toString': '$_id'},
                'status': 1, 
                'date_applied': {'$ifNull': ['$date_applied', '$$NOW']},
                'cover_letter': 1,
                'job_id': {'$toString': '$job._id'},
                'job_title': {'$ifNull': ['$job.title', 'Job No Longer Available']},
                'company_name': {'$ifNull': ['$employer.company_name', 'Company Not Available']},
                'salary': {'$ifNull': ['$job.salary', 0]},
                'location': {'$ifNull': ['$job.location', 'Location Not Available']},
                'formatted_date': {'$dateToString': {'format': '%Y-%m-%d', 'date': {'$ifNull': ['$date_applied', '$$NOW']}}}
            }}
        ]))
        
        # Calculate statistics
        stats_pipeline = [
            {'$match': {'job_seeker_id': current_user.id}},
            {'$group': {
                '_id': None,
                'total_applications': {'$sum': 1},
                'accepted': {'$sum': {'$cond': [{'$eq': ['$status', 'accepted']}, 1, 0]}},
                'pending': {'$sum': {'$cond': [{'$eq': ['$status', 'pending']}, 1, 0]}},
                'rejected': {'$sum': {'$cond': [{'$eq': ['$status', 'rejected']}, 1, 0]}}
            }}
        ]
        
        stats = list(applications_collection.aggregate(stats_pipeline))
        stats = stats[0] if stats else {
            '_id': None,
            'total_applications': 0,
            'accepted': 0,
            'pending': 0,
            'rejected': 0
        }
        
        return render_template('my_applications.html',
                             applications=applications,
                             stats=stats)
                             
    except Exception as e:
        print(f"Error in my_applications: {str(e)}")
        flash('Error loading your applications.', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/apply/<job_id>', methods=['POST'])
@login_required
def apply_job(job_id):
    try:
        if current_user.user_type != 'job_seeker':
            flash('Only job seekers can apply for jobs', 'danger')
            return redirect(url_for('jobs'))
        
        # Calculate match score during application submission
        prediction_model = JobPredictionModel()
        prediction_result = prediction_model.predict(current_user.id, job_id)
        
        cover_letter = request.form.get('cover_letter', '')
        
        # Include prediction in application document
        application = {
            'job_id': ObjectId(job_id),
            'job_seeker_id': ObjectId(current_user.id),
            'cover_letter': cover_letter,
            'date_applied': datetime.utcnow(),
            'status': 'pending',
            'prediction_score': prediction_result['success_probability'],
            'skills_match': prediction_result['skills_match'],
            'experience_match': prediction_result['experience_relevance'],
            'education_match': prediction_result['education_match'],
            'industry_match': prediction_result['industry_match']
        }
        
        # Check if already applied
        existing_application = applications_collection.find_one({
            'job_id': ObjectId(job_id),
            'job_seeker_id': ObjectId(current_user.id)
        })
        
        if existing_application:
            flash('You have already applied for this job', 'warning')
            if request.content_type == 'application/json':
                return jsonify({'status': 'error', 'message': 'Already applied'}), 400
            return redirect(url_for('job_details', job_id=job_id))
        
        # Insert the application
        result = applications_collection.insert_one(application)
        
        if result.inserted_id:
            flash('Application submitted successfully!', 'success')
            if request.content_type == 'application/json':
                return jsonify({'status': 'success', 'message': 'Application submitted successfully'})
            return redirect(url_for('my_applications'))
        else:
            flash('Error submitting application', 'danger')
            if request.content_type == 'application/json':
                return jsonify({'status': 'error', 'message': 'Application submission failed'}), 500
            return redirect(url_for('job_details', job_id=job_id))
            
    except Exception as e:
        logger.error(f"Error in apply_job: {str(e)}")
        flash('Error submitting application', 'danger')
        if request.content_type == 'application/json':
            return jsonify({'status': 'error', 'message': str(e)}), 500
        return redirect(url_for('job_details', job_id=job_id))

@app.route('/update_application_status/<application_id>', methods=['POST'])
@login_required
def update_application_status(application_id):
    if current_user.user_type != 'employer':
        return jsonify({'error': 'Unauthorized access'}), 403

    try:        # Get JSON data from request
        data = request.get_json()
        new_status = data.get('status') if data else None
        if not new_status:
            return jsonify({'error': 'Status is required'}), 400
        
        # Convert IDs to ObjectId
        try:
            application_object_id = ObjectId(application_id)
            employer_object_id = ObjectId(current_user.id)
        except InvalidId as e:
            print(f"Error: Invalid ID format: {str(e)}")
            return jsonify({'error': 'Invalid application ID format'}), 400
            
        # Verify the application exists
        application = applications_collection.find_one({
            '_id': application_object_id
        })

        if not application:
            return jsonify({'error': 'Application not found'}), 404

        # Convert job_id to ObjectId for query
        try:
            job_object_id = ObjectId(application['job_id'])
        except InvalidId as e:
            print(f"Error: Invalid job ID in application: {str(e)}")
            return jsonify({'error': 'Invalid job ID format in application'}), 500

        # Verify the job belongs to the current employer
        job = jobs_collection.find_one({
            '_id': job_object_id,
            'employer_id': employer_object_id
        })

        if not job:
            return jsonify({'error': 'Unauthorized access: Job does not belong to current employer'}), 403

        # Update the application status
        result = applications_collection.update_one(
            {'_id': application_object_id},
            {'$set': {
                'status': new_status,
                'updated_at': datetime.utcnow()
            }}
        )

        if result.modified_count == 1:
            # Send email notification to job seeker
            try:
                job_seeker = users_collection.find_one({'_id': application['job_seeker_id']})
                if job_seeker and job_seeker.get('email'):                    # Get employer profile
                    employer_profile = employer_profiles_collection.find_one({
                        'user_id': job['employer_id']
                    })

                    # Get ML insights for the email
                    prediction_model = JobPredictionModel()
                    try:
                        prediction = prediction_model.predict(
                            str(application['job_seeker_id']),
                            str(job['_id'])
                        )
                        match_score = prediction['success_probability'] * 100
                        skills_match = prediction.get('skills_match', 0)
                        experience_match = prediction.get('experience_relevance', 0)
                    except Exception as e:
                        print(f"Warning: Could not get prediction for email: {str(e)}")
                        match_score = None
                        skills_match = None
                        experience_match = None

                    # Get company name
                    company_name = employer_profile.get('company_name') if employer_profile else 'The Hiring Team'

                    msg = Message(
                        'Application Status Updated',
                        recipients=[job_seeker['email']],
                        html=render_template(
                            'email/application_status_update.html',
                            applicant_name=f"{job_seeker.get('first_name', '')} {job_seeker.get('last_name', '')}",
                            job_title=job['title'],
                            new_status=new_status,
                            company_name=company_name,
                            match_score=match_score,
                            skills_match=skills_match,
                            experience_match=experience_match
                        )
                    )
                    mail.send(msg)
            except Exception as e:
                print(f"Warning: Failed to send email notification: {str(e)}")
                
            return jsonify({
                'success': True, 
                'message': 'Application status updated successfully'
            })
        else:
            return jsonify({
                'error': 'No changes were made to the application'
            }), 400

    except pymongo_errors.ConnectionFailure as e:
        print(f"Database connection error: {str(e)}")
        return jsonify({'error': 'Database connection error'}), 500
        
    except Exception as e:
        print(f"Error updating application status: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': 'An error occurred while updating the application status'}), 500

# Initialize ML model
job_prediction_model = JobPredictionModel()

# =======================
# PREDICTION API ENDPOINTS
# =======================

@app.route('/api/predict/success-score', methods=['POST'])
@login_required
def predict_success_score():
    """Predict success score for a job seeker-job pair"""
    try:
        data = request.get_json()
        
        if not data or 'job_id' not in data:
            return jsonify({
                'error': 'job_id is required'
            }), 400
        
        # If no job_seeker_id provided, use current user
        job_seeker_id = data.get('job_seeker_id', current_user.id)
        
        # Get job seeker data
        job_seeker = users_collection.find_one({
            '_id': ObjectId(job_seeker_id),
            'user_type': 'job_seeker'
        })
        
        if not job_seeker:
            return jsonify({'error': 'Job seeker not found'}), 404
        
        # Get job data
        job = jobs_collection.find_one({'_id': ObjectId(data['job_id'])})
        
        if not job:
            return jsonify({'error': 'Job not found'}), 404
          # Make prediction
        prediction = job_prediction_model.predict(job_seeker_id, data['job_id'])
        
        if prediction is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        response = {
            'success_score': round(float(prediction['success_probability']) * 100, 2),
            'skills_match': round(float(prediction['skills_match']) * 100, 2),
            'experience_match': round(float(prediction['experience_relevance']) * 100, 2),
            'education_match': round(float(prediction['education_match']) * 100, 2),
            'industry_match': round(float(prediction['industry_match']) * 100, 2),
            'job_seeker_id': job_seeker_id,
            'job_id': data['job_id'],
            'job_title': job.get('title', 'Unknown'),
            'company_name': job.get('company_name', 'Unknown'),
            'prediction_timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in predict_success_score: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/predict/job-recommendations', methods=['GET'])
@login_required
def get_job_recommendations():
    """Get job recommendations for the current job seeker"""
    try:
        if current_user.user_type != 'job_seeker':
            return jsonify({'error': 'Only job seekers can get recommendations'}), 403
        
        # Get active jobs
        limit = int(request.args.get('limit', 20))
        jobs = list(jobs_collection.find({
            'status': 'open'
        }).limit(limit))
        
        job_recommendations = []
        
        for job in jobs:
            try:
                # Check if already applied
                existing_application = applications_collection.find_one({
                    'job_seeker_id': ObjectId(current_user.id),
                    'job_id': job['_id']
                })
                
                if existing_application:
                    continue  # Skip jobs already applied to
                
                # Make prediction
                success_score = job_prediction_model.predict_success_probability(current_user.id, str(job['_id']))
                
                if success_score is not None and success_score >= 0.5:  # Only recommend jobs with 50%+ success rate
                    job_info = {
                        'job_id': str(job['_id']),
                        'title': job.get('title', 'Unknown'),
                        'company_name': job.get('company_name', 'Unknown'),
                        'description': job.get('description', ''),
                        'location': job.get('location', ''),
                        'salary': job.get('salary', 0),
                        'predicted_success_score': round(float(success_score) * 100, 2),
                        'posted_at': job.get('date_posted', '').isoformat() if isinstance(job.get('date_posted'), datetime) else str(job.get('date_posted', ''))
                    }
                    
                    job_recommendations.append(job_info)
                    
            except Exception as e:
                logger.error(f"Error processing job {job.get('_id')}: {e}")
                continue
        
        # Sort by success score (highest first)
        job_recommendations.sort(key=lambda x: x['predicted_success_score'], reverse=True)
        
        response = {
            'job_seeker_id': current_user.id,
            'recommended_jobs': job_recommendations,
            'total_recommendations': len(job_recommendations),
            'recommendation_timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in get_job_recommendations: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/predict/dashboard', methods=['GET'])
@login_required
def job_seeker_prediction_dashboard():
    """Get dashboard data with predictions for all applications"""
    try:
        if current_user.user_type != 'job_seeker':
            return jsonify({'error': 'Only job seekers can access this dashboard'}), 403
        
        # Get all applications by this job seeker
        applications = list(applications_collection.find({
            'job_seeker_id': ObjectId(current_user.id)
        }))
        
        application_predictions = []
        
        for app in applications:
            try:
                # Get job data
                job = jobs_collection.find_one({'_id': app['job_id']})
                
                if not job:
                    continue
                
                # Make prediction
                success_score = job_prediction_model.predict_success_probability(
                    current_user.id, 
                    str(app['job_id'])
                )
                
                if success_score is not None:
                    app_info = {
                        'application_id': str(app['_id']),
                        'job_id': str(app['job_id']),
                        'job_title': job.get('title', 'Unknown'),
                        'company_name': job.get('company_name', 'Unknown'),
                        'success_score': round(float(success_score) * 100, 2),
                        'application_status': app.get('status', 'pending'),
                        'applied_at': app.get('date_applied', '').isoformat() if isinstance(app.get('date_applied'), datetime) else str(app.get('date_applied', ''))
                    }
                    
                    application_predictions.append(app_info)
                    
            except Exception as e:
                logger.error(f"Error processing application {app.get('_id')}: {e}")
                continue
        
        # Calculate statistics
        if application_predictions:
            avg_score = sum(app['success_score'] for app in application_predictions) / len(application_predictions)
            max_score = max(app['success_score'] for app in application_predictions)
            min_score = min(app['success_score'] for app in application_predictions)
        else:
            avg_score = max_score = min_score = 0
        
        response = {
            'job_seeker_id': current_user.id,
            'name': current_user.name,
            'email': current_user.email,
            'application_predictions': application_predictions,
            'statistics': {
                'total_applications': len(application_predictions),
                'average_success_score': round(avg_score, 2),
                'highest_success_score': round(max_score, 2),
                'lowest_success_score': round(min_score, 2)
            },
            'dashboard_timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in job_seeker_prediction_dashboard: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/manage_applications', endpoint='manage_applications_all')
@login_required
def manage_applications_all():
    try:
        # Verify user is an employer
        if current_user.user_type != 'employer':
            flash('Access denied. Only employers can view this page.', 'danger')
            return redirect(url_for('dashboard'))
            
        # Verify database connection
        if client is None or db is None:            raise Exception("Database connection is not available")
            
        # Get employer's jobs with proper error handling
        employer_jobs = list(jobs_collection.aggregate([
            {
                '$match': {
                    'employer_id': current_user.id  # IDs are stored as strings
                }
            },
            {
                '$lookup': {
                    'from': 'applications',
                    'localField': '_id',
                    'foreignField': 'job_id',
                    'as': 'applications'
                }
            },
            {
                '$lookup': {
                    'from': 'users',
                    'localField': 'applications.job_seeker_id',
                    'foreignField': '_id',
                    'as': 'applicants'
                }
            },
            {
                '$project': {
                    '_id': 1,  # Keep original string ID
                    'title': 1,
                    'applications': 1,
                    'applicants': 1
                }
            }
        ]))

        if employer_jobs is None:            raise Exception("Failed to fetch employer jobs")
            
        # Process applications data
        jobs_with_applications = []
        for job in employer_jobs:
            applications_data = []
            for application in job.get('applications', []):
                applicant = next(
                    (user for user in job.get('applicants', []) 
                     if user['_id'] == application['job_seeker_id']), 
                    None)
                
                if applicant:
                    application_info = {
                        'application_id': str(application['_id']),
                        'job_title': job['title'],
                        'applicant_name': f"{applicant.get('first_name', '')} {applicant.get('last_name', '')}",
                        'applicant_email': applicant.get('email', ''),
                        'status': application.get('status', 'Pending'),
                        'date_applied': application.get('date_applied', '').strftime('%Y-%m-%d') if application.get('date_applied') else 'N/A'
                    }
                    applications_data.append(application_info)

            if applications_data:
                jobs_with_applications.append({
                    'job_id': str(job['_id']),
                    'title': job['title'],
                    'applications': applications_data
                })

        return render_template('manage_applications.html', jobs=jobs_with_applications)

    except pymongo_errors.ConnectionFailure as e:
        flash('Database connection error. Please try again later.', 'danger')
        return redirect(url_for('dashboard'))
        
    except Exception as e:
        print(f"Error in manage_applications_all: {str(e)}")
        flash('An error occurred while loading applications.', 'danger')
        return redirect(url_for('dashboard'))

# Background prediction update task
def update_all_predictions():
    """Background task to update predictions"""
    try:
        print("Starting prediction update task...")
        prediction_model = JobPredictionModel()
        applications = applications_collection.find({'status': 'pending'})
        
        updated_count = 0
        for application in applications:
            try:
                prediction = prediction_model.predict(
                    str(application['job_seeker_id']), 
                    str(application['job_id'])
                )
                
                applications_collection.update_one(
                    {'_id': application['_id']},
                    {'$set': {
                        'prediction_score': prediction['success_probability'],
                        'skills_match': prediction['skills_match'],
                        'experience_match': prediction['experience_relevance'],
                        'education_match': prediction['education_match'],
                        'industry_match': prediction['industry_match']
                    }}
                )
                updated_count += 1
                
            except Exception as e:
                logger.error(f"Error updating prediction for application {application['_id']}: {str(e)}")
                
        print(f"Updated predictions for {updated_count} applications")
                
    except Exception as e:
        logger.error(f"Error in update_all_predictions: {str(e)}")

# Schedule the prediction update task (run every hour)
def schedule_prediction_updates():
    while True:
        try:
            update_all_predictions()
        except Exception as e:
            logger.error(f"Error in prediction update scheduler: {str(e)}")
        time.sleep(3600)  # Sleep for 1 hour

# Start the prediction update scheduler in a background thread
from threading import Thread
prediction_update_thread = Thread(target=schedule_prediction_updates, daemon=True)
prediction_update_thread.start()

if __name__ == '__main__':
    port = 5000
    print(f"\nServer is running!")
    print(f"* Local URL: http://localhost:{port}")
    print(f"* Network URL: http://127.0.0.1:{port}")
    print("\nPress CTRL+C to quit")
    app.run(host='0.0.0.0', port=port, debug=True)