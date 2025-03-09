import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random
import string
from datetime import datetime, timedelta
import json

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_recruitment_dataset(n_samples=5000):
    """
    Generate a synthetic dataset for software engineering recruitment with:
    - Technical skills and experience
    - Education information
    - Protected attributes (gender, race, age)
    - Proxy variables
    - Historical hiring decisions (potentially biased)
    - True qualification metrics
    
    Parameters:
    -----------
    n_samples : int
        Number of candidate profiles to generate
        
    Returns:
    --------
    pandas.DataFrame
        Dataset containing candidate profiles
    """
    
    # Define distributions for protected attributes
    genders = ['Male', 'Female', 'Non-binary']
    gender_probs = [0.65, 0.30, 0.05]  # Intentionally imbalanced to reflect industry reality
    
    ethnicities = ['White', 'Asian', 'Black', 'Hispanic', 'Middle Eastern', 'Native American', 'Pacific Islander']
    ethnicity_probs = [0.55, 0.25, 0.07, 0.08, 0.03, 0.01, 0.01]  # Imbalanced
    
    # Names by gender and ethnicity (proxy variables)
    with open('names_data.json', 'w') as f:
        json.dump({
            'male_names': ['James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard', 'Joseph', 'Thomas', 'Wei', 'Chen', 'Mohammed', 'Ali', 'Juan', 'Carlos'],
            'female_names': ['Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan', 'Jessica', 'Sarah', 'Mei', 'Priya', 'Fatima', 'Maria', 'Sofia', 'Aisha'],
            'nb_names': ['Alex', 'Taylor', 'Jordan', 'Casey', 'Riley', 'Avery', 'Quinn', 'Skyler', 'Dakota', 'Hayden'],
            'white_last_names': ['Smith', 'Johnson', 'Williams', 'Jones', 'Brown', 'Miller', 'Davis', 'Anderson'],
            'asian_last_names': ['Wang', 'Li', 'Zhang', 'Liu', 'Chen', 'Patel', 'Kim', 'Singh', 'Nguyen'],
            'black_last_names': ['Washington', 'Jefferson', 'Jackson', 'Robinson', 'Harris', 'Coleman'],
            'hispanic_last_names': ['Garcia', 'Rodriguez', 'Martinez', 'Lopez', 'Gonzalez', 'Perez'],
            'middle_eastern_last_names': ['Khan', 'Ali', 'Ahmed', 'Hassan', 'Mohammad'],
            'native_american_last_names': ['Redbird', 'Blackhawk', 'Whitefeather', 'Running Bear'],
            'pacific_islander_last_names': ['Kama', 'Mahelona', 'Faletoi', 'Tuala']
        }, f)
    
    with open('names_data.json', 'r') as f:
        names_data = json.load(f)
    
    # Colleges by prestige level (proxy for socioeconomic status)
    college_tiers = {
        'elite': ['Stanford University', 'MIT', 'Harvard University', 'Carnegie Mellon', 'UC Berkeley', 'Princeton University'],
        'good': ['University of Michigan', 'Georgia Tech', 'UT Austin', 'University of Washington', 'UCLA', 'USC'],
        'average': ['Penn State', 'Ohio State', 'Arizona State', 'Michigan State', 'Purdue University', 'Texas A&M'],
        'below_average': ['Community colleges', 'Online universities', 'State colleges', 'Local universities'],
        'bootcamp': ['Coding Bootcamp A', 'Tech Academy B', 'Code School C', 'Programming Institute D']
    }
    
    # Technical skills
    programming_languages = ['Python', 'Java', 'JavaScript', 'C++', 'C#', 'Go', 'Ruby', 'Swift', 'Kotlin', 'PHP', 'TypeScript']
    frameworks = ['React', 'Angular', 'Vue.js', 'Django', 'Flask', 'Spring', 'TensorFlow', 'PyTorch', '.NET', 'Node.js']
    databases = ['MySQL', 'PostgreSQL', 'MongoDB', 'Oracle', 'Redis', 'SQLite', 'DynamoDB', 'Cassandra']
    cloud_platforms = ['AWS', 'Azure', 'Google Cloud', 'Heroku', 'DigitalOcean']
    
    # Companies by prestige (proxy for career privilege)
    company_tiers = {
        'FAANG+': ['Google', 'Meta', 'Amazon', 'Apple', 'Netflix', 'Microsoft', 'Tesla', 'Nvidia'],
        'tech_giants': ['Salesforce', 'Oracle', 'IBM', 'Intel', 'Adobe', 'Uber', 'Airbnb', 'Twitter'],
        'startups': ['Tech Startup A', 'AI Startup B', 'Fintech Startup C', 'Growth Startup D'],
        'traditional': ['Bank Corp', 'Insurance Inc', 'Retail Co', 'Manufacturing Ltd', 'Healthcare Services']
    }
    
    # Initialize data containers
    data = {
        'gender': [],
        'ethnicity': [],
        'age': [],
        'name': [],
        'email': [],
        'education_level': [],
        'college_tier': [],
        'college_name': [],
        'gpa': [],
        'years_experience': [],
        'company_tier': [],
        'current_company': [],
        'past_companies': [],
        'num_programming_languages': [],
        'programming_skills': [],
        'framework_knowledge': [],
        'database_experience': [],
        'cloud_platforms': [],
        'has_open_source': [],
        'github_contributions': [],
        'hackathon_participation': [],
        'coding_test_score': [],
        'problem_solving_score': [],
        'communication_score': [],
        'system_design_score': [],
        'location': [],
        'work_gaps': [],
        'hired': [],  # Potentially biased historical decision
        'truly_qualified': []  # Ground truth qualification (unbiased)
    }
    
    # Generate data
    for i in range(n_samples):
        # Protected attributes
        gender = np.random.choice(genders, p=gender_probs)
        ethnicity = np.random.choice(ethnicities, p=ethnicity_probs)
        age = int(np.random.normal(30, 5))
        age = max(21, min(55, age))  # Ensure reasonable age range
        
        # Proxy variables - Name based on gender and ethnicity
        if gender == 'Male':
            first_name = random.choice(names_data['male_names'])
        elif gender == 'Female':
            first_name = random.choice(names_data['female_names'])
        else:
            first_name = random.choice(names_data['nb_names'])
            
        if ethnicity == 'White':
            last_name = random.choice(names_data['white_last_names'])
        elif ethnicity == 'Asian':
            last_name = random.choice(names_data['asian_last_names'])
        elif ethnicity == 'Black':
            last_name = random.choice(names_data['black_last_names'])
        elif ethnicity == 'Hispanic':
            last_name = random.choice(names_data['hispanic_last_names'])
        elif ethnicity == 'Middle Eastern':
            last_name = random.choice(names_data['middle_eastern_last_names'])
        elif ethnicity == 'Native American':
            last_name = random.choice(names_data['native_american_last_names'])
        else:  # Pacific Islander
            last_name = random.choice(names_data['pacific_islander_last_names'])
            
        name = f"{first_name} {last_name}"
        
        # Email (another proxy)
        email_providers = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com', 'icloud.com']
        email = f"{first_name.lower()}.{last_name.lower()}@{random.choice(email_providers)}"
        
        # Education
        education_options = ['Bootcamp', 'Associates', 'Bachelors', 'Masters', 'PhD']
        education_probs = [0.1, 0.05, 0.6, 0.2, 0.05]
        education_level = np.random.choice(education_options, p=education_probs)
        
        # College tier and name (socioeconomic proxy)
        # Introducing correlation between ethnicity and college tier
        if ethnicity in ['White', 'Asian']:
            tier_probs = [0.2, 0.3, 0.3, 0.15, 0.05]  # Higher chance of elite/good
        else:
            tier_probs = [0.05, 0.15, 0.35, 0.35, 0.1]  # Lower chance of elite/good
            
        college_tier = np.random.choice(['elite', 'good', 'average', 'below_average', 'bootcamp'], p=tier_probs)
        college_name = random.choice(college_tiers[college_tier])
        
        # For bootcamp education, ensure college tier is bootcamp
        if education_level == 'Bootcamp':
            college_tier = 'bootcamp'
            college_name = random.choice(college_tiers['bootcamp'])
            
        # GPA - slightly correlated with college tier
        if college_tier == 'elite':
            gpa = max(2.5, min(4.0, np.random.normal(3.6, 0.3)))
        elif college_tier == 'good':
            gpa = max(2.3, min(4.0, np.random.normal(3.4, 0.4)))
        elif college_tier == 'average':
            gpa = max(2.0, min(4.0, np.random.normal(3.2, 0.5)))
        elif college_tier == 'below_average':
            gpa = max(2.0, min(4.0, np.random.normal(3.0, 0.6)))
        else:  # bootcamp
            gpa = np.nan  # No GPA for bootcamps
        
        # Experience - correlated with age
        years_experience = max(0, min(age - 21, int(np.random.normal((age - 21) * 0.8, 2))))
        
        # Company tier - correlated with education and experience
        if college_tier in ['elite', 'good'] and years_experience > 3:
            company_tier_probs = [0.3, 0.4, 0.2, 0.1]  # Higher chance of FAANG+
        elif education_level in ['Masters', 'PhD']:
            company_tier_probs = [0.2, 0.3, 0.3, 0.2]
        else:
            company_tier_probs = [0.05, 0.15, 0.4, 0.4]  # Lower chance of FAANG+
            
        company_tier = np.random.choice(['FAANG+', 'tech_giants', 'startups', 'traditional'], p=company_tier_probs)
        current_company = random.choice(company_tiers[company_tier])
        
        # Past companies
        num_past_companies = min(years_experience // 2, 5)  # Assume average 2 years per company
        past_companies_list = []
        
        for _ in range(num_past_companies):
            past_tier = np.random.choice(['FAANG+', 'tech_giants', 'startups', 'traditional'])
            past_company = random.choice(company_tiers[past_tier])
            if past_company != current_company and past_company not in past_companies_list:
                past_companies_list.append(past_company)
                
        past_companies = ', '.join(past_companies_list)
        
        # Technical skills - correlated with experience and education
        skill_factor = min(1.0, (years_experience / 10) + (0.2 if education_level in ['Masters', 'PhD'] else 0))
        
        # Number of programming languages
        max_languages = len(programming_languages)
        num_languages = max(1, min(max_languages, int(np.random.normal(max_languages * skill_factor, 2))))
        known_languages = random.sample(programming_languages, num_languages)
        
        # Frameworks, databases, cloud
        num_frameworks = max(0, min(len(frameworks), int(np.random.normal(len(frameworks) * skill_factor, 2))))
        num_databases = max(1, min(len(databases), int(np.random.normal(len(databases) * skill_factor * 0.8, 1))))
        num_cloud = max(0, min(len(cloud_platforms), int(np.random.normal(len(cloud_platforms) * skill_factor * 0.7, 1))))
        
        known_frameworks = random.sample(frameworks, num_frameworks)
        known_databases = random.sample(databases, num_databases)
        known_clouds = random.sample(cloud_platforms, num_cloud)
        
        # Open source contributions (correlated with skill and experience)
        os_probability = 0.2 + (skill_factor * 0.5)
        has_open_source = np.random.random() < os_probability
        
        # If has open source, generate contribution count
        if has_open_source:
            github_contributions = int(np.random.gamma(shape=5, scale=skill_factor * 50))
        else:
            github_contributions = 0
            
        # Hackathon participation
        hackathon_probability = 0.1 + (skill_factor * 0.4)
        hackathon_participation = int(np.random.binomial(5, hackathon_probability))
        
        # Test scores - truly reflective of ability
        base_skill = (
            (num_languages / max_languages) * 0.2 +
            (years_experience / 10) * 0.4 +
            has_open_source * 0.2 +
            (hackathon_participation / 5) * 0.1 +
            ((gpa if not np.isnan(gpa) else 3.0) / 4.0) * 0.1
        )
        
        # Add some random variation to skill
        true_skill = min(1.0, max(0.1, base_skill + np.random.normal(0, 0.1)))
        
        # Generate test scores based on true skill
        coding_test_score = min(100, max(0, int(np.random.normal(true_skill * 100, 10))))
        problem_solving_score = min(100, max(0, int(np.random.normal(true_skill * 100, 12))))
        communication_score = min(100, max(0, int(np.random.normal(70 + (true_skill * 30), 15))))  # Less variance
        system_design_score = min(100, max(0, int(np.random.normal(true_skill * 100, 10))))
        
        # Location - can be a proxy for socioeconomic status
        tech_hubs = ['San Francisco, CA', 'Seattle, WA', 'New York, NY', 'Austin, TX', 'Boston, MA']
        secondary_tech = ['Denver, CO', 'Atlanta, GA', 'Chicago, IL', 'Los Angeles, CA', 'Washington DC']
        other_locations = ['Phoenix, AZ', 'Miami, FL', 'Dallas, TX', 'Indianapolis, IN', 'Columbus, OH']
        
        # Location probabilities correlated with company tier
        if company_tier in ['FAANG+', 'tech_giants']:
            location_probs = [0.6, 0.3, 0.1]  # Higher chance of tech hub
        else:
            location_probs = [0.2, 0.3, 0.5]  # Higher chance of other locations
            
        location_type = np.random.choice(['tech_hub', 'secondary', 'other'], p=location_probs)
        
        if location_type == 'tech_hub':
            location = random.choice(tech_hubs)
        elif location_type == 'secondary':
            location = random.choice(secondary_tech)
        else:
            location = random.choice(other_locations)
        
        # Work gaps (can be biased against)
        work_gap_probability = 0.1
        # Higher for women (modeling potential family leave bias)
        if gender == 'Female':
            work_gap_probability = 0.25
            
        work_gaps = np.random.random() < work_gap_probability
        
        # True qualification (objective)
        # Based on coding_test_score, problem_solving_score, experience, etc.
        qualification_score = (
            coding_test_score * 0.3 +
            problem_solving_score * 0.3 +
            system_design_score * 0.2 +
            communication_score * 0.1 +
            (min(10, years_experience) / 10) * 100 * 0.1  # Max 10 years experience
        )
        
        truly_qualified = qualification_score >= 75  # Objective threshold
        
        # Biased historical hiring decision
        # Incorporates biases against certain demographics, work gaps, education paths
        bias_factors = 0
        
        # Gender bias
        if gender == 'Female':
            bias_factors -= 5
        elif gender == 'Non-binary':
            bias_factors -= 10
            
        # Ethnicity bias
        if ethnicity in ['Black', 'Hispanic', 'Native American']:
            bias_factors -= 7
        elif ethnicity in ['Middle Eastern', 'Pacific Islander']:
            bias_factors -= 5
            
        # Age bias (against older candidates)
        if age > 40:
            bias_factors -= (age - 40) * 0.5
            
        # Education bias
        if education_level == 'Bootcamp':
            bias_factors -= 10
        elif education_level == 'Associates':
            bias_factors -= 8
        elif education_level == 'PhD':  # Potentially perceived as "too academic"
            bias_factors -= 3
            
        # College bias
        if college_tier == 'elite':
            bias_factors += 10
        elif college_tier == 'good':
            bias_factors += 5
        elif college_tier == 'below_average':
            bias_factors -= 5
            
        # Work gap bias
        if work_gaps:
            bias_factors -= 8
            
        # Company bias
        if company_tier == 'FAANG+':
            bias_factors += 10
        elif company_tier == 'tech_giants':
            bias_factors += 5
        elif company_tier == 'traditional':
            bias_factors -= 3
            
        # Biased score
        biased_score = qualification_score + bias_factors
        
        # Historical hiring decision
        hired = biased_score >= 75
        
        # Add to data
        data['gender'].append(gender)
        data['ethnicity'].append(ethnicity)
        data['age'].append(age)
        data['name'].append(name)
        data['email'].append(email)
        data['education_level'].append(education_level)
        data['college_tier'].append(college_tier)
        data['college_name'].append(college_name)
        data['gpa'].append(gpa)
        data['years_experience'].append(years_experience)
        data['company_tier'].append(company_tier)
        data['current_company'].append(current_company)
        data['past_companies'].append(past_companies)
        data['num_programming_languages'].append(num_languages)
        data['programming_skills'].append(', '.join(known_languages))
        data['framework_knowledge'].append(', '.join(known_frameworks))
        data['database_experience'].append(', '.join(known_databases))
        data['cloud_platforms'].append(', '.join(known_clouds))
        data['has_open_source'].append(has_open_source)
        data['github_contributions'].append(github_contributions)
        data['hackathon_participation'].append(hackathon_participation)
        data['coding_test_score'].append(coding_test_score)
        data['problem_solving_score'].append(problem_solving_score)
        data['communication_score'].append(communication_score)
        data['system_design_score'].append(system_design_score)
        data['location'].append(location)
        data['work_gaps'].append(work_gaps)
        data['hired'].append(hired)
        data['truly_qualified'].append(truly_qualified)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add bias metrics
    df['biased_decision'] = ((df['hired'] != df['truly_qualified'])).astype(int)
    
    # Calculate bias rates by demographics
    print("Bias analysis:")
    print(f"Overall qualification rate: {df['truly_qualified'].mean():.2f}")
    print(f"Overall hiring rate: {df['hired'].mean():.2f}")
    
    for gender in genders:
        gender_qualified = df[df['gender'] == gender]['truly_qualified'].mean()
        gender_hired = df[df['gender'] == gender]['hired'].mean()
        print(f"Gender: {gender}, Qualified: {gender_qualified:.2f}, Hired: {gender_hired:.2f}")
    
    for ethnicity in ethnicities:
        eth_qualified = df[df['ethnicity'] == ethnicity]['truly_qualified'].mean()
        eth_hired = df[df['ethnicity'] == ethnicity]['hired'].mean()
        print(f"Ethnicity: {ethnicity}, Qualified: {eth_qualified:.2f}, Hired: {eth_hired:.2f}")
    
    return df

# Generate dataset
recruitment_data = generate_recruitment_dataset(5000)

# Save to CSV
recruitment_data.to_csv('recruitment_dataset.csv', index=False)

# Create train/test split with stratification on protected attributes
from sklearn.model_selection import train_test_split

# Features that should be used for modeling
model_features = [
    'years_experience', 'num_programming_languages',
    'has_open_source', 'github_contributions', 'hackathon_participation',
    'coding_test_score', 'problem_solving_score', 'communication_score', 'system_design_score',
    # Include some proxy variables
    'education_level', 'college_tier', 'gpa', 'company_tier', 'work_gaps'
]

# Optional: Include protected attributes for baseline comparison
protected_attributes = ['gender', 'ethnicity', 'age']

# Split data (for a fair test, use 'truly_qualified' as target)
X = recruitment_data[model_features]
y = recruitment_data['truly_qualified']

# Convert categorical variables to one-hot encoding
X_encoded = pd.get_dummies(X, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, 
    stratify=recruitment_data[['gender', 'ethnicity']].apply(lambda x: str(x['gender']) + '_' + str(x['ethnicity']), axis=1)
)

# Save splits
pd.concat([X_train, y_train], axis=1).to_csv('recruitment_train.csv', index=False)
pd.concat([X_test, y_test], axis=1).to_csv('recruitment_test.csv', index=False)

print(f"Dataset created with {len(recruitment_data)} entries")
print(f"Training set: {len(X_train)} entries")
print(f"Test set: {len(X_test)} entries")