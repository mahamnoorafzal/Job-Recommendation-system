from cover_letter_analyzer import CoverLetterAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from bson import ObjectId
import logging
import os
import joblib
import json
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class JobPredictionModel:
    def __init__(self):
        self.mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
        self.db_name = 'job_portal'
        self.analyzer = CoverLetterAnalyzer()
        
        # Initialize database connection
        try:
            self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            self.db = self.client[self.db_name]
            logger.info("Successfully connected to MongoDB")
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            self.db = None
            
        # Define model paths
        self.models_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        self.model_path = os.path.join(self.models_dir, 'job_prediction_model.pkl')
        self.scaler_path = os.path.join(self.models_dir, 'job_prediction_scaler.pkl')
            
        # Initialize model and scaler
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.tfidf_skills = TfidfVectorizer(max_features=1000)
        self.tfidf_description = TfidfVectorizer(max_features=1000)
        
        # Load or create the model and scaler
        self._load_or_create_model()

    def _get_training_data(self):
        """Collect training data from historical applications."""
        try:
            # Get all completed applications
            applications = list(self.db.applications.find({
                'status': {'$in': ['accepted', 'rejected']}
            }))
            
            if not applications:
                logger.warning("No historical applications found for training")
                return None, None
            
            features_list = []
            labels = []
            
            for app in applications:
                try:
                    # Get job seeker profile
                    job_seeker_profile = self.db.job_seeker_profiles.find_one({
                        'user_id': app['job_seeker_id']
                    })
                    
                    # Get job data
                    job = self.db.jobs.find_one({'_id': app['job_id']})
                    
                    if not job_seeker_profile or not job:
                        continue
                        
                    # Calculate features without relying on created_at
                    previous_applications = list(self.db.applications.find({
                        'job_seeker_id': app['job_seeker_id']
                    }))
                    
                    # Get cover letter score if available
                    cover_letter_score = 0.0
                    if 'cover_letter' in app:
                        cover_letter_score = self.analyze_cover_letter(
                            app['cover_letter'],
                            job.get('description', '')
                        )
                    
                    # Create feature vector
                    features = [
                        len(previous_applications),  # total applications
                        float(job_seeker_profile.get('experience_years', 0)),  # experience
                        len(job_seeker_profile.get('skills', [])),  # number of skills
                        float(job.get('salary', 0)),  # job salary
                        sum(1 for prev_app in previous_applications if prev_app.get('status') == 'accepted'),  # accepted apps
                        0.0,  # default success rate
                        cover_letter_score  # cover letter relevance
                    ]
                    
                    # Calculate success rate if there are previous applications
                    if previous_applications:
                        features[5] = features[4] / features[0]
                    
                    features_list.append(features)
                    labels.append(1 if app['status'] == 'accepted' else 0)
                    
                except Exception as e:
                    logger.error(f"Error processing application {app['_id']}: {str(e)}")
                    continue
            
            if not features_list:
                # Create dummy training data if no real data exists
                logger.warning("No valid applications found, creating dummy training data")
                features_list = [
                    [0, 0, 0, 0, 0, 0, 0.0],  # Rejected case
                    [5, 3, 10, 50000, 3, 0.6, 0.8]  # Accepted case
                ]
                labels = [0, 1]
            
            return np.array(features_list), np.array(labels)
            
        except Exception as e:
            logger.error(f"Error collecting training data: {str(e)}")
            return None, None

    def _load_or_create_model(self):
        """Load existing model and scaler or create new ones."""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"Loaded existing model with scaler expecting {self.scaler.n_features_in_} features")
            else:
                logger.info("Creating and training new model...")
                X, y = self._get_training_data()
                
                # Initialize model and scaler
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
                self.scaler = StandardScaler()
                
                if X is not None and y is not None and len(X) > 0:
                    # Fit the scaler and transform the data
                    X_scaled = self.scaler.fit_transform(X)
                    # Train the model
                    self.model.fit(X_scaled, y)
                    # Save the model and scaler
                    joblib.dump(self.model, self.model_path)
                    joblib.dump(self.scaler, self.scaler_path)
                    logger.info("Successfully trained and saved new model")
                else:
                    # Create and fit with dummy data
                    logger.warning("No training data available. Initializing with dummy data")
                    dummy_X = np.array([[0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]])
                    dummy_y = np.array([0, 1])
                    X_scaled = self.scaler.fit_transform(dummy_X)
                    self.model.fit(X_scaled, dummy_y)
        except Exception as e:
            logger.error(f"Error loading/creating job prediction model: {e}")
            # Initialize with dummy data as fallback
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()
            dummy_X = np.array([[0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]])
            dummy_y = np.array([0, 1])
            X_scaled = self.scaler.fit_transform(dummy_X)
            self.model.fit(X_scaled, dummy_y)

    def _calculate_skills_match(self, applicant_skills, job_skills):
        """Calculate the skills match score between applicant and job."""
        try:
            if not applicant_skills or not job_skills:
                return 0.0
                
            # Convert skills to lowercase sets for case-insensitive comparison
            applicant_skills_set = set(str(skill).lower() for skill in applicant_skills)
            job_skills_set = set(str(skill).lower() for skill in job_skills)
            
            # Calculate intersection and match score
            matching_skills = applicant_skills_set.intersection(job_skills_set)
            
            # Return ratio of matching skills to required skills
            return len(matching_skills) / len(job_skills_set) if job_skills_set else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating skills match: {str(e)}")
            return 0.0

    def prepare_features(self, job_seeker_id, job_id, cover_letter=None):
        """Prepare features for prediction including cover letter analysis."""
        try:
            # Get job seeker data
            job_seeker = self.db.users.find_one({
                '_id': ObjectId(job_seeker_id),
                'user_type': 'job_seeker'
            })
            
            # Get job seeker profile
            job_seeker_profile = self.db.job_seeker_profiles.find_one({
                'user_id': ObjectId(job_seeker_id)
            })
            
            # Get job data
            job = self.db.jobs.find_one({'_id': ObjectId(job_id)})
            
            if not job_seeker or not job:
                raise ValueError("Job seeker or job not found")
                
            # Get application history
            previous_applications = list(self.db.applications.find({
                'job_seeker_id': ObjectId(job_seeker_id)
            }))
            
            # Calculate features matching the scaler's expectations
            features = {
                'total_applications': len(previous_applications),
                'experience_years': job_seeker_profile.get('experience_years', 0),
                'skills_count': len(job_seeker_profile.get('skills', [])),
                'average_job_salary': job.get('salary', 0),
                'accepted_applications': sum(1 for app in previous_applications if app.get('status') == 'accepted'),
                'success_rate': sum(1 for app in previous_applications if app.get('status') == 'accepted') / len(previous_applications) if previous_applications else 0
            }

            # Add cover letter analysis if provided
            cover_letter_score = 0.0
            if cover_letter:
                cover_letter_score = self.analyze_cover_letter(
                    cover_letter,
                    job.get('description', '')
                )

            # Convert features to array and scale
            feature_array = np.array([
                features['total_applications'],
                features['experience_years'],
                features['skills_count'],
                features['average_job_salary'],
                features['accepted_applications'],
                features['success_rate'],
                cover_letter_score
            ]).reshape(1, -1)

            if self.scaler is None:
                raise ValueError("Scaler is not initialized")
            
            logger.info(f"Feature array shape before scaling: {feature_array.shape}")
            logger.info(f"Number of features in scaler: {self.scaler.n_features_in_}")
            
            return self.scaler.transform(feature_array)
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise

    def analyze_cover_letter(self, cover_letter, job_description):
        """Analyze cover letter relevance to job description"""
        try:
            if not cover_letter or not job_description:
                return 0.0
                
            # Create TF-IDF vectors for cover letter and job description
            tfidf = TfidfVectorizer(stop_words='english')
            vectors = tfidf.fit_transform([cover_letter, job_description])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            
            return float(similarity)
        except Exception as e:
            logger.error(f"Error analyzing cover letter: {str(e)}")
            return 0.0

    def predict(self, job_seeker_id, job_id, cover_letter=None):
        """Predict success probability for a job application."""
        try:
            # Convert string IDs to ObjectId if needed
            if isinstance(job_seeker_id, str):
                job_seeker_id = ObjectId(job_seeker_id)
            if isinstance(job_id, str):
                job_id = ObjectId(job_id)
            
            # Get job seeker profile
            job_seeker_profile = self.db.job_seeker_profiles.find_one({
                'user_id': ObjectId(job_seeker_id)
            })
            
            # Get job data
            job = self.db.jobs.find_one({'_id': ObjectId(job_id)})
            
            if not job_seeker_profile or not job:
                logger.error(f"Job seeker profile or job not found: {job_seeker_id}, {job_id}")
                return {
                    'success_probability': 0.5,
                    'skills_match': 0.0,
                    'experience_relevance': 0.0,
                    'education_match': 0.0,
                    'industry_match': 0.0,
                    'cover_letter_match': 0.0
                }
            
            try:
                features = self.prepare_features(job_seeker_id, job_id, cover_letter)
                if features is None or self.model is None:
                    raise ValueError("Features or model not available")
                
                # Get prediction probabilities
                pred_probas = self.model.predict_proba(features)
                success_probability = float(pred_probas[0][1]) if pred_probas.shape[1] >= 2 else 0.5
                
                # Calculate match scores
                skills_match = min(1.0, self._calculate_skills_match(
                    job_seeker_profile.get('skills', []),
                    job.get('required_skills', [])
                ))
                
                experience_match = min(1.0, float(
                    job_seeker_profile.get('experience_years', 0) / 
                    max(1, job.get('experience_required', 1))
                ))
                
                # Map education levels to numeric values
                education_levels = {
                    'high_school': 1,
                    'associate': 2,
                    'bachelor': 3,
                    'master': 4,
                    'phd': 5
                }
                
                seeker_edu_level = education_levels.get(str(job_seeker_profile.get('education', '')).lower(), 1)
                required_edu_level = education_levels.get(str(job.get('education_required', '')).lower(), 1)
                education_match = min(1.0, float(seeker_edu_level) / max(1, float(required_edu_level)))
                
                # Calculate industry match using case-insensitive comparison
                seeker_industries = [str(ind).lower() for ind in job_seeker_profile.get('industry_preferences', [])]
                job_industry = str(job.get('industry', '')).lower()
                industry_match = 1.0 if job_industry in seeker_industries else 0.0
                
                # Calculate cover letter match if provided
                cover_letter_match = 0.0
                if cover_letter:
                    cover_letter_match = self.analyze_cover_letter(cover_letter, job.get('description', ''))
                
                # Ensure all values are valid floats between 0 and 1
                sanitized_response = {
                    'success_probability': max(0.0, min(1.0, float(success_probability))),
                    'skills_match': max(0.0, min(1.0, float(skills_match))),
                    'experience_relevance': max(0.0, min(1.0, float(experience_match))),
                    'education_match': max(0.0, min(1.0, float(education_match))),
                    'industry_match': max(0.0, min(1.0, float(industry_match))),
                    'cover_letter_match': max(0.0, min(1.0, float(cover_letter_match)))
                }
                
                # Log successful prediction
                logger.info(f"Successful prediction for job_seeker_id={job_seeker_id}, job_id={job_id}")
                return sanitized_response
                
            except Exception as calc_error:
                logger.error(f"Error in prediction calculation: {str(calc_error)}")
                return {
                    'success_probability': 0.5,
                    'skills_match': 0.0,
                    'experience_relevance': 0.0,
                    'education_match': 0.0,
                    'industry_match': 0.0,
                    'cover_letter_match': 0.0
                }
                
        except Exception as e:
            logger.error(f"Error in predict method: {str(e)}")
            return {
                'success_probability': 0.5,
                'skills_match': 0.0,
                'experience_relevance': 0.0,
                'education_match': 0.0,
                'industry_match': 0.0,
                'cover_letter_match': 0.0
            }
