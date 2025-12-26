import pandas as pd
import numpy as np
from faker import Faker
import random
from faker.providers import BaseProvider

# Initialize Faker and set seeds for reproducibility
fake = Faker('en_IN')
random.seed(42)
np.random.seed(42)

class EngineeringProvider(BaseProvider):
    def engineering_field(self):
        fields = ['Computer Science', 'Mechanical', 'Electrical', 'Civil', 'Electronics']
        weights = [0.45, 0.18, 0.15, 0.12, 0.10]  # More realistic distribution
        return random.choices(fields, weights=weights, k=1)[0]
    
    def college_tier(self, jee_rank, twelfth_percent):
        def logistic_prob(offset, scale, x):
            return 1 / (1 + np.exp(-(offset + scale * x)))
        
        # Calculate probabilities for each tier using logistic functions
        prob_top   = logistic_prob(offset=2.5, scale=-0.00004, x=jee_rank) * logistic_prob(offset=-8, scale=0.15, x=twelfth_percent)
        prob_mid   = logistic_prob(offset=1.8, scale=-0.00003, x=jee_rank) * logistic_prob(offset=-7, scale=0.13, x=twelfth_percent)
        prob_nit   = logistic_prob(offset=1.2, scale=-0.00002, x=jee_rank) * logistic_prob(offset=-6, scale=0.11, x=twelfth_percent)
        prob_state = logistic_prob(offset=0.8, scale=-0.00001, x=jee_rank) * logistic_prob(offset=-5, scale=0.09, x=twelfth_percent)
        
        tiers = ['Top IIT', 'Mid IIT', 'NIT', 'State College', 'Private College']
        probs = [prob_top, prob_mid, prob_nit, prob_state, 1 - sum([prob_top, prob_mid, prob_nit, prob_state])]
        return random.choices(tiers, weights=probs, k=1)[0]

fake.add_provider(EngineeringProvider)

# New options for extra categorical fields required for the new model
domain_options = ['Full Stack', 'Machine Learning', 'Android Development', 'Other']
referral_options = ['Yes', 'No']

# Number of records to generate
num_records = 5000
data = []

for _ in range(num_records):
    # Generate academic scores with correlation
    tenth_percent = np.random.beta(a=2, b=1.5) * 40 + 60
    tenth_percent = round(np.clip(tenth_percent, 60, 100), 1)
    
    twelfth_percent = np.clip(tenth_percent * 0.9 + np.random.normal(5, 7), 60, 100)
    twelfth_percent = round(twelfth_percent, 1)
    
    # Generate JEE rank with a realistic distribution
    jee_rank = int(np.exp(np.random.normal(8.5, 1.2)))
    jee_rank = max(1, min(jee_rank, 250000))
    
    # Determine college tier based on JEE rank and 12th percentage
    college_tier = fake.college_tier(jee_rank, twelfth_percent)
    
    # Personal details
    gender = random.choice(['Male', 'Female'])
    engineering_field = fake.engineering_field()
    
    # Overall work experience based on college tier
    exp_params = {
        'Top IIT': {'lambda': 0.8, 'max_exp': 8},
        'Mid IIT': {'lambda': 0.7, 'max_exp': 7},
        'NIT': {'lambda': 0.6, 'max_exp': 6},
        'State College': {'lambda': 0.5, 'max_exp': 5},
        'Private College': {'lambda': 0.4, 'max_exp': 4}
    }
    params = exp_params[college_tier]
    experience = min(np.random.poisson(params['lambda']) + np.random.binomial(1, 0.2), params['max_exp'])
    
    # Domain-specific experience (years in the selected field)
    experience_field = round(np.random.uniform(0, experience), 1) if experience > 0 else 0.0
    
    # Additional features
    num_projects = np.random.poisson(lam=(experience_field + 1))
    expertise_level = int(np.clip(round(np.random.normal(loc=1 + experience_field, scale=1.0)), 1, 5))
    internships_mean = {'Top IIT': 2, 'Mid IIT': 2, 'NIT': 1, 'State College': 1, 'Private College': 0.5}
    num_internships = np.random.poisson(lam=internships_mean[college_tier])
    soft_skill_rating = int(np.clip(round(np.random.normal(loc=3.5 if college_tier in ['Top IIT', 'Mid IIT'] else 3.0, scale=0.8)), 1, 5))
    aptitude_rating = int(np.clip(round(np.random.normal(loc=3.5 if college_tier in ['Top IIT', 'Mid IIT'] else 3.0, scale=0.8)), 1, 5))
    dsa_level = int(np.clip(round(np.random.normal(loc=3, scale=1.0)), 1, 5))
    hackathons_mean = {'Top IIT': 2, 'Mid IIT': 1.5, 'NIT': 1, 'State College': 0.8, 'Private College': 0.5}
    num_hackathons = np.random.poisson(lam=hackathons_mean[college_tier])
    competitive_coding_solved = int(np.clip(round(np.random.normal(loc=80 + 10 * experience_field, scale=30)), 1, 200))
    num_repos = np.random.poisson(lam=5)
    github_activities = np.random.poisson(lam=10)
    linkedin_posts = np.random.poisson(lam=3)
    num_certifications = np.random.poisson(lam=1.5)
    cgpa = round(np.random.beta(a=2, b=2) * 9 + 1, 2)
    
    # Extra categorical features for new model
    domain = random.choice(domain_options)
    referral = random.choice(referral_options)
    
    # Adjusted Salary calculation
    base_salaries = {
        'Top IIT': np.random.lognormal(mean=14.5, sigma=0.1),  # Increased mean for Top IIT
        'Mid IIT': np.random.lognormal(mean=13.8, sigma=0.12),
        'NIT': np.random.lognormal(mean=13.2, sigma=0.15),
        'State College': np.random.lognormal(mean=12.5, sigma=0.18),
        'Private College': np.random.lognormal(mean=12.0, sigma=0.2)
    }
    
    field_multipliers = {
        'Computer Science': 1.8 + 0.1 * (college_tier in ['Top IIT', 'Mid IIT']),
        'Electronics': 1.5,
        'Electrical': 1.3,
        'Mechanical': 1.1,
        'Civil': 1.0
    }
    
    # Adjust salary based on experience and internships
    salary = base_salaries[college_tier] * field_multipliers[engineering_field] * (1 + 0.1 * experience + 0.05 * num_internships)
    salary = int(np.clip(salary + np.random.normal(0, 50000), 300000, 8000000))  # Increased upper limit
    
    data.append([
        tenth_percent,
        twelfth_percent,
        jee_rank,
        gender,
        experience,
        experience_field,
        engineering_field,
        college_tier,
        num_projects,
        expertise_level,
        num_internships,
        soft_skill_rating,
        aptitude_rating,
        dsa_level,
        num_hackathons,
        competitive_coding_solved,
        num_repos,
        github_activities,
        linkedin_posts,
        num_certifications,
        cgpa,
        domain,
        referral,
        salary
    ])

# Define column names for the new dataset
columns = [
    '10th_percent', '12th_percent', 'jee_rank', 'gender', 'experience', 'experience_field',
    'engineering_field', 'college_tier', 'num_projects', 'expertise_level', 'num_internships',
    'soft_skill_rating', 'aptitude_rating', 'dsa_level', 'num_hackathons', 'competitive_coding_solved',
    'num_repos', 'github_activities', 'linkedin_posts', 'num_certifications', 'cgpa', 'domain', 'referral', 'salary'
]

# Create DataFrame and save to CSV
df = pd.DataFrame(data, columns=columns)
df.to_csv('data_csv_2.csv', index=False)

print(f"Generated {len(df)} records")
print(df.describe())