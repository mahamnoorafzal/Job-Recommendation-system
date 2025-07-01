import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from bson import ObjectId
import joblib
import os
import logging
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from datetime import datetime0

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridJobPredictionModel:
    def __init__(self, db):
        """Initialize the hybrid job prediction model combining ML and NLP capabilities."""
        self.db = db
        self.ml_model = None
        self.nlp_model = None
        self.scaler = None
        self.skill_vectorizer = TfidfVectorizer(max_features=1000)
        self.text_vectorizer = TfidfVectorizer(max_features=1000)
        
        # Initialize paths
        self.models_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        self.ml_model_path = os.path.join(self.models_dir, 'hybrid_ml_model.pkl')
        self.nlp_model_path = os.path.join(self.models_dir, 'hybrid_nlp_model.pkl')
        self.scaler_path = os.path.join(self.models_dir, 'hybrid_scaler.pkl')
        
        # Initialize NLP components
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            self.sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
        except Exception as e:
            logger.error(f"Error initializing NLP components: {e}")
            self.sentiment_analyzer = None
        
        self._load_or_create_models()

    def _load_or_create_models(self):
        """Load existing models and scaler or create new ones."""
        try:
            if os.path.exists(self.ml_model_path):
                self.ml_model = joblib.load(self.ml_model_path)
                self.scaler = joblib.load(self.scaler_path)
                logger.info("Loaded existing models")
            else:
                logger.info("Creating new models")
                self.ml_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
                self.scaler = StandardScaler()
                self._train_models()
        except Exception as e:
            logger.error(f"Error loading/creating models: {e}")
            self.ml_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.scaler = StandardScaler()

    def _calculate_enhanced_skills_match(self, applicant_skills, job_skills):
        """Calculate enhanced skills match using both exact and semantic matching."""
        try:
            if not applicant_skills or not job_skills:
                return 0.0, []
            
            # Convert skills to lowercase sets for case-insensitive matching
            applicant_skills_set = set(str(skill).lower() for skill in applicant_skills)
            job_skills_set = set(str(skill).lower() for skill in job_skills)
            
            # Exact match score
            exact_matches = applicant_skills_set.intersection(job_skills_set)
            exact_match_score = len(exact_matches) / len(job_skills_set)
            
            # Semantic similarity score using TF-IDF
            try:
                applicant_skills_text = " ".join(applicant_skills_set)
                job_skills_text = " ".join(job_skills_set)
                
                # Calculate TF-IDF vectors
                tfidf_matrix = self.skill_vectorizer.fit_transform([applicant_skills_text, job_skills_text])
                semantic_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            except Exception as e:
                logger.warning(f"Error calculating semantic similarity: {e}")
                semantic_similarity = 0.0
            
            # Combined score with weights
            final_score = (0.7 * exact_match_score) + (0.3 * semantic_similarity)
            
            # Return matched skills for detailed feedback
            return final_score, list(exact_matches)
            
        except Exception as e:
            logger.error(f"Error calculating enhanced skills match: {e}")
            return 0.0, []

    def _calculate_experience_match(self, applicant_exp, required_exp):
        """Calculate experience match score with diminishing returns."""
        try:
            if not required_exp:
                return 1.0
            
            applicant_exp = float(applicant_exp or 0)
            required_exp = float(required_exp)
            
            if applicant_exp >= required_exp:
                # Add diminishing returns for excess experience
                base_score = 1.0
                excess_exp = applicant_exp - required_exp
                bonus = min(0.2, excess_exp * 0.02)  # Max 20% bonus
                return min(1.0, base_score + bonus)
            else:
                # Linear scaling for under-experienced candidates
                return applicant_exp / required_exp
                
        except Exception as e:
            logger.error(f"Error calculating experience match: {e}")
            return 0.0

    def _analyze_cover_letter(self, cover_letter, job_description):
        """Analyze cover letter using sentiment and content relevance."""
        try:
            if not cover_letter or not job_description:
                return 0.0, 0.0
            
            # Content relevance using TF-IDF and cosine similarity
            text_vectors = self.text_vectorizer.fit_transform([cover_letter, job_description])
            content_relevance = cosine_similarity(text_vectors[0:1], text_vectors[1:2])[0][0]
            
            # Sentiment analysis
            if self.sentiment_analyzer:
                sentiment_result = self.sentiment_analyzer(cover_letter[:512])[0]
                sentiment_score = sentiment_result['score'] if sentiment_result['label'] == 'POSITIVE' else 1 - sentiment_result['score']
            else:
                sentiment_score = 0.5
            
            return content_relevance, sentiment_score
            
        except Exception as e:
            logger.error(f"Error analyzing cover letter: {e}")
            return 0.0, 0.0

    def prepare_features(self, job_seeker_id, job_id, cover_letter=None):
        """Prepare comprehensive feature set for prediction."""
        try:
            # Get job seeker profile and job data
            job_seeker_profile = self.db.job_seeker_profiles.find_one({'user_id': ObjectId(job_seeker_id)})
            job = self.db.jobs.find_one({'_id': ObjectId(job_id)})
            
            if not job_seeker_profile or not job:
                raise ValueError("Job seeker profile or job not found")
            
            # Get application history
            previous_applications = list(self.db.applications.find({
                'job_seeker_id': ObjectId(job_seeker_id)
            }))
            
            # Calculate base features
            features = {
                'total_applications': len(previous_applications),
                'experience_years': job_seeker_profile.get('experience_years', 0),
                'skills_count': len(job_seeker_profile.get('skills', [])),
                'average_job_salary': job.get('salary', 0),
                'accepted_applications': sum(1 for app in previous_applications if app.get('status') == 'accepted'),
                'success_rate': 0.0
            }
            
            # Calculate success rate
            if features['total_applications'] > 0:
                features['success_rate'] = features['accepted_applications'] / features['total_applications']
            
            # Enhanced matching scores
            skills_match, matched_skills = self._calculate_enhanced_skills_match(
                job_seeker_profile.get('skills', []),
                job.get('required_skills', [])
            )
            features['skills_match'] = skills_match
            
            # Experience match with diminishing returns
            features['experience_match'] = self._calculate_experience_match(
                job_seeker_profile.get('experience_years', 0),
                job.get('experience_required', 0)
            )
            
            # Cover letter analysis
            if cover_letter:
                content_relevance, sentiment_score = self._analyze_cover_letter(
                    cover_letter,
                    job.get('description', '')
                )
                features['cover_letter_relevance'] = content_relevance
                features['cover_letter_sentiment'] = sentiment_score
            else:
                features['cover_letter_relevance'] = 0.0
                features['cover_letter_sentiment'] = 0.5
            
            # Convert features to array format
            feature_array = np.array([
                features['total_applications'],
                features['experience_years'],
                features['skills_count'],
                features['average_job_salary'],
                features['accepted_applications'],
                features['success_rate'],
                features['skills_match'],
                features['experience_match'],
                features['cover_letter_relevance'],
                features['cover_letter_sentiment']
            ]).reshape(1, -1)
            
            return self.scaler.transform(feature_array) if self.scaler else feature_array
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None

    def predict(self, job_seeker_id, job_id, cover_letter=None):
        """Make a comprehensive prediction using hybrid approach."""
        try:
            # Get required data
            job_seeker_profile = self.db.job_seeker_profiles.find_one({'user_id': ObjectId(job_seeker_id)})
            job = self.db.jobs.find_one({'_id': ObjectId(job_id)})
            
            if not job_seeker_profile or not job:
                logger.error(f"Job seeker profile or job not found: {job_seeker_id}, {job_id}")
                return self._get_default_response()
            
            # Prepare features and make prediction
            features = self.prepare_features(job_seeker_id, job_id, cover_letter)
            if features is None or self.ml_model is None:
                return self._get_default_response()
            
            # Get ML model prediction
            pred_probas = self.ml_model.predict_proba(features)
            success_probability = float(pred_probas[0][1]) if pred_probas.shape[1] >= 2 else 0.5
            
            # Calculate detailed match scores
            skills_match, matched_skills = self._calculate_enhanced_skills_match(
                job_seeker_profile.get('skills', []),
                job.get('required_skills', [])
            )
            
            experience_match = self._calculate_experience_match(
                job_seeker_profile.get('experience_years', 0),
                job.get('experience_required', 0)
            )
            
            # Education match calculation
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
            
            # Industry match
            seeker_industries = [str(ind).lower() for ind in job_seeker_profile.get('industry_preferences', [])]
            job_industry = str(job.get('industry', '')).lower()
            industry_match = 1.0 if job_industry in seeker_industries else 0.0
            
            # Cover letter analysis
            cover_letter_match = 0.0
            if cover_letter:
                content_relevance, sentiment_score = self._analyze_cover_letter(cover_letter, job.get('description', ''))
                cover_letter_match = (content_relevance + sentiment_score) / 2
            
            # Prepare response with comprehensive scores
            response = {
                'success_probability': max(0.0, min(1.0, float(success_probability))),
                'skills_match': max(0.0, min(1.0, float(skills_match))),
                'experience_relevance': max(0.0, min(1.0, float(experience_match))),
                'education_match': max(0.0, min(1.0, float(education_match))),
                'industry_match': max(0.0, min(1.0, float(industry_match))),
                'cover_letter_match': max(0.0, min(1.0, float(cover_letter_match))),
                'matched_skills': matched_skills,
                'missing_skills': list(set(job.get('required_skills', [])) - set(matched_skills)),
                'prediction_date': datetime.utcnow()
            }
            
            # Log successful prediction
            logger.info(f"Successful hybrid prediction for job_seeker_id={job_seeker_id}, job_id={job_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error in hybrid prediction: {str(e)}")
            return self._get_default_response()

    def _get_default_response(self):
        """Return default response structure with zero scores."""
        return {
            'success_probability': 0.5,
            'skills_match': 0.0,
            'experience_relevance': 0.0,
            'education_match': 0.0,
            'industry_match': 0.0,
            'cover_letter_match': 0.0,
            'matched_skills': [],
            'missing_skills': [],
            'prediction_date': datetime.utcnow()
        }

    def _train_models(self):
        """Train the hybrid prediction models using historical data."""
        try:
            # Get all completed applications
            applications = list(self.db.applications.find({
                'status': {'$in': ['accepted', 'rejected']}
            }))
            
            if not applications:
                logger.warning("No historical applications found for training")
                self._create_dummy_training_data()
                return
            
            features_list = []
            labels = []
            
            for app in applications:
                try:
                    # Get job seeker profile and job data
                    job_seeker_profile = self.db.job_seeker_profiles.find_one({
                        'user_id': app['job_seeker_id']
                    })
                    job = self.db.jobs.find_one({'_id': app['job_id']})
                    
                    if not job_seeker_profile or not job:
                        continue
                    
                    # Prepare features
                    features = self.prepare_features(
                        str(app['job_seeker_id']),
                        str(app['job_id']),
                        app.get('cover_letter')
                    )
                    
                    if features is not None:
                        features_list.append(features[0])
                        labels.append(1 if app['status'] == 'accepted' else 0)
                        
                except Exception as e:
                    logger.warning(f"Error processing application {app['_id']}: {e}")
                    continue
            
            if features_list:
                # Train models
                X = np.array(features_list)
                y = np.array(labels)
                self.ml_model.fit(X, y)
                
                # Save models
                joblib.dump(self.ml_model, self.ml_model_path)
                joblib.dump(self.scaler, self.scaler_path)
                
                logger.info("Successfully trained and saved models")
            else:
                logger.warning("No valid features extracted for training")
                self._create_dummy_training_data()
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            self._create_dummy_training_data()

    def _create_dummy_training_data(self):
        """Create and train with dummy data if no real training data is available."""
        try:
            # Create dummy feature vectors
            X = np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Rejected case
                [5, 3, 10, 75000, 3, 0.6, 0.8, 0.9, 0.7, 0.8]  # Accepted case
            ])
            y = np.array([0, 1])
            
            # Fit scaler and model
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            self.ml_model.fit(X_scaled, y)
            
            # Save models
            joblib.dump(self.ml_model, self.ml_model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
            logger.info("Created and trained with dummy data")
            
        except Exception as e:
            logger.error(f"Error creating dummy training data: {e}")
