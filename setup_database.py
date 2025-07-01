from pymongo import MongoClient
from datetime import datetime
def init_mongodb():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['job_prediction_db']  # Consistent database name
    
    # Users collection (basic info)
    db.create_collection('users', validator={
        '$jsonSchema': {
            'bsonType': 'object',
            'required': ['email', 'password', 'user_type', 'created_at'],
            'properties': {
                'first_name': {'bsonType': 'string'},
                'last_name': {'bsonType': 'string'},
                'email': {
                    'bsonType': 'string',
                    'pattern': '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
                },
                'password': {'bsonType': 'string'},
                'user_type': {
                    'bsonType': 'string',
                    'enum': ['employer', 'job_seeker']
                },
                'created_at': {'bsonType': 'date'}
            }
        }
    })
    db.users.create_index('email', unique=True)

    # Employer profiles (separate collection)
    db.create_collection('employer_profiles')
    db.employer_profiles.create_index('user_id', unique=True)

    # Job seeker profiles (separate collection)
    db.create_collection('job_seeker_profiles')
    db.job_seeker_profiles.create_index('user_id', unique=True)

    # Jobs collection
    db.create_collection('jobs')
    db.jobs.create_index('employer_id')
    db.jobs.create_index('status')

    # Applications collection
    db.create_collection('applications')
    db.applications.create_index('job_id')
    db.applications.create_index('job_seeker_id')
    db.applications.create_index([('job_id', 1), ('job_seeker_id', 1)], unique=True)

    # Reviews collection
    db.create_collection('reviews')
    db.reviews.create_index('reviewee_id')
    db.reviews.create_index('reviewer_id')

    print("Database initialized successfully!")

if __name__ == '__main__':
    init_mongodb()