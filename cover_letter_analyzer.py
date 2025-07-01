from spacy import load
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class CoverLetterAnalyzer:
    def __init__(self):
        # Load English language model
        self.nlp = load('en_core_web_sm')
        
        # Common skill keywords
        self.skill_patterns = [
            'python', 'java', 'javascript', 'sql', 'react', 'node.js', 'html', 'css',
            'machine learning', 'data analysis', 'project management', 'leadership',
            'communication', 'teamwork', 'problem solving', 'git', 'agile', 'scrum',
            'database', 'cloud', 'aws', 'azure', 'docker', 'kubernetes', 'ci/cd',
            'testing', 'debugging', 'api', 'rest', 'microservices', 'security',
            'mobile development', 'web development', 'devops', 'linux', 'windows'
        ]
        
    def extract_information(self, cover_letter):
        """Extract skills and experience from cover letter"""
        doc = self.nlp(cover_letter.lower())
        
        # Extract experience
        experience_pattern = re.compile(r'(\d+)\s*(?:year|yr|years|yrs?)(?:\s+of\s+experience)?')
        experience_matches = experience_pattern.findall(cover_letter.lower())
        years_experience = max([int(x) for x in experience_matches]) if experience_matches else 0
        
        # Extract skills
        extracted_skills = []
        text_lower = cover_letter.lower()
        
        # Direct skill matching
        for skill in self.skill_patterns:
            if skill in text_lower:
                extracted_skills.append(skill)
        
        # Extract skill phrases using NLP
        skill_phrases = ['proficient in', 'experience with', 'skilled in', 'knowledge of', 
                        'expertise in', 'familiar with', 'worked with', 'developed using']
        
        for sent in doc.sents:
            sent_lower = sent.text.lower()
            if any(phrase in sent_lower for phrase in skill_phrases):
                for token in sent:
                    # Check if token is a potential skill (noun or proper noun)
                    if token.pos_ in ['NOUN', 'PROPN']:
                        token_text = token.text.lower()
                        # Additional verification to avoid common non-skill nouns
                        if len(token_text) > 2 and not token_text in ['year', 'years', 'experience', 'knowledge']:
                            extracted_skills.append(token_text)
        
        return {
            'extracted_skills': list(set(extracted_skills)),
            'years_experience': years_experience
        }
