import os
from pymongo import MongoClient, errors
from datetime import datetime
import time
from dotenv import load_dotenv

def get_db_connection(max_retries=3, retry_delay=1):
    """Get MongoDB connection with retry logic"""
    load_dotenv()
    mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
    client = None
    
    for attempt in range(max_retries):
        try:
            client = MongoClient(mongo_uri, 
                               serverSelectionTimeoutMS=5000,
                               connectTimeoutMS=5000,
                               socketTimeoutMS=5000)
            # Verify connection
            client.server_info()
            return client, None
        except errors.ServerSelectionTimeoutError as e:
            if client is not None:
                client.close()
            if attempt == max_retries - 1:
                return None, f"Could not connect to MongoDB after {max_retries} attempts: {str(e)}"
            print(f"Connection attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        except Exception as e:
            if client is not None:
                client.close()
            return None, f"Error connecting to MongoDB: {str(e)}"

def initialize_collections(db):
    """Initialize MongoDB collections if they don't exist"""
    if db is None:
        print("Error: Database object is None")
        return False
        
    try:
        collections = ['users', 'jobs', 'applications', 'reviews', 'employer_profiles', 'job_seeker_profiles']
        existing_collections = db.list_collection_names()
        
        for collection in collections:
            if collection not in existing_collections:
                db.create_collection(collection)
                print(f"Created collection: {collection}")
                
                # Add indexes for common queries
                if collection == 'users':
                    db[collection].create_index('email', unique=True)
                elif collection == 'jobs':
                    db[collection].create_index([('title', 'text'), ('description', 'text')])
                    db[collection].create_index('employer_id')
                    db[collection].create_index('status')
                elif collection == 'applications':
                    db[collection].create_index([('job_id', 1), ('job_seeker_id', 1)], unique=True)
                    db[collection].create_index('status')
                elif collection == 'reviews':
                    db[collection].create_index('reviewee_id')
        return True
    except Exception as e:
        print(f"Error initializing collections: {str(e)}")
        return False

def check_and_setup_database():
    """Initialize database and verify connection"""
    client, error = get_db_connection()
    if client is None:
        return False, error
    
    try:
        db = client['job_prediction_db']
        if db is None:
            return False, "Could not access database"
            
        collections = ['users', 'jobs', 'applications', 'reviews', 'employer_profiles', 'job_seeker_profiles']
        
        # Create collections if they don't exist
        for collection in collections:
            if collection not in db.list_collection_names():
                db.create_collection(collection)
        
        # Create indexes
        db.users.create_index('email', unique=True)
        db.jobs.create_index([('title', 'text'), ('description', 'text')])
        db.applications.create_index([('job_id', 1), ('job_seeker_id', 1)], unique=True)
        
        return True, "Database setup completed successfully"
    except Exception as e:
        return False, f"Error setting up database: {str(e)}"
    finally:
        if client is not None:
            client.close()

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_database_connection():
    """
    Create a connection to MongoDB and return the database instance
    """
    try:
        # Get MongoDB URI from environment variable or use default
        mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
        
        # Create client with timeout
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        
        # Force a connection to verify it works
        client.server_info()
        
        # Get database
        db = client['job_prediction_db']
        return True, db, "Successfully connected to database"
    
    except ConnectionFailure as e:
        return False, None, f"Failed to connect to MongoDB. Error: {str(e)}"
    except Exception as e:
        return False, None, f"An error occurred while connecting to database: {str(e)}"

def check_and_setup_database():
    """
    Check database connection and setup initial collections if needed
    """
    success, db, message = get_database_connection()
    
    if not success:
        return False, message
    
    try:
        # Create collections if they don't exist
        collections = ['users', 'jobs', 'applications', 'reviews', 'employer_profiles', 'job_seeker_profiles']
        for collection in collections:
            if collection not in db.list_collection_names():
                db.create_collection(collection)
        
        return True, "Database setup completed successfully"
    except Exception as e:
        return False, f"Error setting up database: {str(e)}"

def validate_job_data(job_data):
    """
    Validate job data before insertion
    Returns (is_valid: bool, errors: list)
    """
    errors = []
    required_fields = {
        'employer_id': str,
        'title': str,
        'description': str,
        'required_skills': list,
        'salary': (int, float),
        'location': str,
        'experience_level': str,
        'deadline': str,
        'status': str
    }
    
    # Check required fields
    for field, field_type in required_fields.items():
        if field not in job_data:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(job_data[field], field_type):
            errors.append(f"Invalid type for {field}: expected {field_type}, got {type(job_data[field])}")

    # Additional validation
    if len(errors) == 0:
        # Validate status
        valid_statuses = ['open', 'closed', 'draft', 'archived']
        if job_data['status'] not in valid_statuses:
            errors.append(f"Invalid status: {job_data['status']}. Must be one of {valid_statuses}")

        # Validate experience level
        valid_exp_levels = ['Entry Level', 'Mid Level', 'Senior Level', 'Lead', 'Executive']
        if job_data['experience_level'] not in valid_exp_levels:
            errors.append(f"Invalid experience_level: {job_data['experience_level']}. Must be one of {valid_exp_levels}")

        # Validate salary
        if job_data['salary'] < 0:
            errors.append("Salary cannot be negative")

        # Validate skills
        if not job_data['required_skills']:
            errors.append("At least one required skill must be specified")
        elif not all(isinstance(skill, str) for skill in job_data['required_skills']):
            errors.append("All skills must be strings")

        # Basic title/description validation  
        if len(job_data['title'].strip()) < 3:
            errors.append("Title must be at least 3 characters")
        if len(job_data['description'].strip()) < 50:
            errors.append("Description must be at least 50 characters")

    return len(errors) == 0, errors

def prepare_job_data(job_data, existing_job=None):
    """
    Prepare job data for insertion/update by:
    - Adding default values
    - Converting types
    - Sanitizing strings
    etc.
    """
    from datetime import datetime
    import re

    prepared_data = {}
    
    if existing_job:
        # Start with existing data for updates
        prepared_data = existing_job.copy()

    # Update with new data
    prepared_data.update({
        k: v for k, v in job_data.items() 
        if k in ['employer_id', 'title', 'description', 'required_skills', 
                 'salary', 'location', 'experience_level', 'deadline', 'status']
    })

    # Clean strings
    for field in ['title', 'description', 'location']:
        if field in prepared_data:
            prepared_data[field] = prepared_data[field].strip()
            
    # Ensure required_skills is a list
    if 'required_skills' in prepared_data:
        if isinstance(prepared_data['required_skills'], str):
            prepared_data['required_skills'] = [s.strip() for s in prepared_data['required_skills'].split(',')]
        prepared_data['required_skills'] = list(set(prepared_data['required_skills']))  # Remove duplicates

    # Convert salary to integer
    if 'salary' in prepared_data:
        prepared_data['salary'] = int(float(prepared_data['salary']))

    # Set defaults for new jobs
    if not existing_job:
        prepared_data.setdefault('created_at', datetime.utcnow())
        prepared_data.setdefault('status', 'draft')

    return prepared_data

def get_job_by_id(job_id, db=None):
    """
    Get a job by its ID with proper error handling
    Returns (job, error)
    """
    from bson import ObjectId
    from pymongo.errors import InvalidId

    try:
        if not db:
            success, db, message = get_database_connection()
            if not success:
                return None, message

        # Validate ObjectId
        if not ObjectId.is_valid(job_id):
            return None, "Invalid job ID format"

        job = db.jobs.find_one({'_id': ObjectId(job_id)})
        if not job:
            return None, "Job not found"
            
        return job, None

    except InvalidId:
        return None, "Invalid job ID format"
    except Exception as e:
        return None, f"Error retrieving job: {str(e)}"

def insert_job(job_data, db=None):
    """
    Insert a new job with validation
    Returns (inserted_id, error)
    """
    try:
        if not db:
            success, db, message = get_database_connection()
            if not success:
                return None, message

        # Validate job data
        is_valid, errors = validate_job_data(job_data)
        if not is_valid:
            return None, f"Validation failed: {', '.join(errors)}"

        # Prepare data
        prepared_data = prepare_job_data(job_data)

        # Insert into database
        result = db.jobs.insert_one(prepared_data)
        if not result.inserted_id:
            return None, "Failed to insert job"

        return result.inserted_id, None

    except Exception as e:
        return None, f"Error inserting job: {str(e)}"
