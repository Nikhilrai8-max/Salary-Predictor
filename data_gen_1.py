# data_generation.py
import pandas as pd
import numpy as np
from faker import Faker
import random
from faker.providers import BaseProvider

# Initialize Faker
fake = Faker('en_IN')
random.seed(42)
np.random.seed(42)

class EngineeringProvider(BaseProvider):
    def real_college_branch_tier(self, jee_rank):
        if jee_rank <= 2500:
            tier = 'Tier 1 - Top IIT'
            colleges = ['IIT Bombay', 'IIT Delhi', 'IIT Madras', 'IIT Kanpur', 'IIT Kharagpur']
            weights = [0.35, 0.30, 0.15, 0.12, 0.08]
        elif jee_rank <= 10000:
            tier = 'Tier 1 - Other IIT/Top NIT'
            colleges = ['IIT Roorkee', 'IIT Guwahati', 'IIT Hyderabad', 'NIT Trichy', 
                       'NIT Surathkal', 'NIT Warangal', 'IIIT Hyderabad']
            weights = [0.20, 0.20, 0.15, 0.15, 0.10, 0.10, 0.10]
        elif jee_rank <= 40000:
            tier = 'Tier 2 - Mid Colleges'
            colleges = ['NIT Calicut', 'NIT Rourkela', 'IIIT Bangalore', 'DTU Delhi',
                       'NSUT Delhi', 'Thapar University', 'BIT Mesra']
            weights = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.10]
        elif jee_rank <= 100000:
            tier = 'Tier 3 - Private/State'
            colleges = ['VIT Vellore', 'SRM Chennai', 'KIIT Bhubaneswar', 
                        'Manipal Institute', 'PES University', 'Jadavpur University']
            weights = [0.25, 0.20, 0.20, 0.15, 0.10, 0.10]
        else:
            tier = 'Tier 4 - Other'
            colleges = ['Local State College', 'Private Engineering College',
                       'Regional Institute', 'Community College']
            weights = [0.40, 0.30, 0.20, 0.10]

        college = random.choices(colleges, weights=weights, k=1)[0]

        branch_dist = {
            'Tier 1 - Top IIT': {'CSE':0.40, 'ECE':0.25, 'EE':0.15, 'ME':0.10, 'Other':0.10},
            'Tier 1 - Other IIT/Top NIT': {'CSE':0.35, 'ECE':0.25, 'EE':0.15, 'ME':0.15, 'Other':0.10},
            'Tier 2 - Mid Colleges': {'CSE':0.30, 'ECE':0.20, 'EE':0.15, 'ME':0.20, 'Other':0.15},
            'Tier 3 - Private/State': {'CSE':0.25, 'ECE':0.20, 'EE':0.15, 'ME':0.25, 'Other':0.15},
            'Tier 4 - Other': {'CSE':0.20, 'ECE':0.15, 'EE':0.15, 'ME':0.30, 'Other':0.20}
        }
        
        branch = random.choices(
            list(branch_dist[tier].keys()),
            weights=list(branch_dist[tier].values()),
            k=1
        )[0]

        return tier, college, branch

fake.add_provider(EngineeringProvider)

def get_domain(branch):
    domain_map = {
        'CSE': ['Machine Learning', 'Full Stack', 'Data Science', 'Cyber Security'],
        'ECE': ['VLSI Design', 'Embedded Systems', 'IoT Development'],
        'EE': ['Power Systems', 'Renewable Energy', 'Smart Grids'],
        'ME': ['Robotics', 'Automotive', 'Thermal Engineering'],
        'Other': ['Core Engineering', 'Management', 'Research']
    }
    return random.choice(domain_map.get(branch, ['General Engineering']))

def generate_data(num_records):
    data = []
    for _ in range(num_records):
        # Academic Background
        jee_rank = int(np.clip(np.random.lognormal(10, 1.2), 1, 250000))
        tenth = round(np.random.normal(85, 5), 1)
        twelfth = round(np.random.normal(80, 7), 1)
        cgpa = round(np.random.normal(8.5, 0.8), 2)
        
        # College Details
        tier, college, branch = fake.real_college_branch_tier(jee_rank)
        
        # Professional Skills
        experience = min(np.random.poisson(1.5), 8)
        experience_field = round(min(experience * np.random.beta(2, 1), experience), 1)
        num_internships = min(np.random.poisson(1 + experience*0.5), 5)
        num_projects = min(np.random.poisson(3 + experience*0.7), 20)
        competitive_coding = min(np.random.poisson(100 + 50*experience), 1000)
        num_certifications = min(np.random.poisson(2 + experience*0.3), 12)
        soft_skills = int(np.clip(np.random.normal(3.5, 0.8), 1, 5))
        dsa_level = int(np.clip(np.random.normal(3, 1), 1, 5))
        
        # Social Activities
        num_repos = min(np.random.poisson(5 + experience*0.5), 30)
        github_activities = min(np.random.poisson(10 + experience), 50)
        linkedin_posts = min(np.random.poisson(2 + experience*0.3), 20)
        
        # Domain and Referral
        domain = get_domain(branch)
        referral = 'Yes' if np.random.rand() < 0.15 else 'No'
        gender = random.choice(['Male', 'Female', 'Other'])
        
        # Salary Calculation
        base_salary = {
            'Tier 1 - Top IIT': 2400000,
            'Tier 1 - Other IIT/Top NIT': 1600000,
            'Tier 2 - Mid Colleges': 900000,
            'Tier 3 - Private/State': 600000,
            'Tier 4 - Other': 400000
        }[tier]
        
        salary = base_salary * (1 + 0.1 * dsa_level + 0.05 * soft_skills + 0.15 * (num_internships/5))
        salary *= np.random.uniform(0.95, 1.05)
        salary = int(np.clip(salary, 300000, 4000000))
        
        data.append({
            '10th_percent': tenth,
            '12th_percent': twelfth,
            'jee_rank': jee_rank,
            'gender': gender,
            'cgpa': cgpa,
            'college_tier': tier,
            'college_name': college,
            'branch': branch,
            'experience': experience,
            'experience_field': experience_field,
            'num_projects': num_projects,
            'expertise_level': int(np.clip(experience_field + num_projects/5, 1, 5)),
            'num_internships': num_internships,
            'soft_skill_rating': soft_skills,
            'aptitude_rating': int(np.clip(np.random.normal(3.5, 0.8), 1, 5)),
            'dsa_level': dsa_level,
            'num_hackathons': min(np.random.poisson(1 + experience*0.3), 10),
            'competitive_coding_solved': competitive_coding,
            'num_repos': num_repos,
            'github_activities': github_activities,
            'linkedin_posts': linkedin_posts,
            'num_certifications': num_certifications,
            'domain': domain,
            'referral': referral,
            'salary': salary
        })
    
    return pd.DataFrame(data)

# Main execution
if __name__ == "__main__":
    df = generate_data(5000)
    df.to_csv('data_csv_1.csv', index=False)
    print("Data generation complete. Shape:", df.shape)



# import pandas as pd
# import numpy as np
# from faker import Faker
# import random
# from faker.providers import BaseProvider

# # ─── Init ─────────────────────────────────────────────────────────────
# fake = Faker('en_IN')
# random.seed(42)
# np.random.seed(42)

# class EngineeringProvider(BaseProvider):
#     def real_college_branch_tier(self, jee_rank):
#         # Determine tier & candidate colleges by JEE rank
#         if jee_rank <= 2000:
#             tier     = 'Top IIT'
#             colleges = ['IIT Bombay','IIT Delhi','IIT Madras','IIT Kanpur','IIT Kharagpur']
#             weights  = [0.25,0.25,0.20,0.15,0.15]
#         elif jee_rank <= 5000:
#             tier     = 'Other IIT'
#             colleges = ['IIT Guwahati','IIT Roorkee','IIT BHU','IIT Hyderabad','IIT Dhanbad','IIT Ropar','IIT Gandhinagar']
#             weights  = [0.20,0.20,0.15,0.15,0.10,0.10,0.10]
#         elif jee_rank <= 15000:
#             tier     = 'NIT'
#             colleges = ['NIT Trichy','NIT Surathkal','NIT Warangal','NIT Calicut','NIT Rourkela','NIT Allahabad']
#             weights  = [0.20,0.20,0.20,0.15,0.15,0.10]
#         elif jee_rank <= 30000:
#             tier     = 'IIIT/State College'
#             colleges = ['IIIT Hyderabad','IIIT Delhi','DTU Delhi','PEC Chandigarh','MAIT Delhi']
#             weights  = [0.25,0.20,0.20,0.20,0.15]
#         else:
#             tier     = 'Private College'
#             colleges = ['BITS Pilani','VIT Vellore','SRM Chennai','Amity Noida','Lovely Professional University']
#             weights  = [0.20,0.20,0.20,0.20,0.20]

#         college = random.choices(colleges, weights=weights, k=1)[0]

#         # Pick a branch with realistic popularity by tier
#         branch_weights = {
#             'Top IIT':            {'CSE':0.35,'ECE':0.25,'EE':0.15,'ME':0.15,'CE':0.10},
#             'Other IIT':          {'CSE':0.30,'ECE':0.25,'EE':0.20,'ME':0.15,'CE':0.10},
#             'NIT':                {'CSE':0.30,'ECE':0.25,'ME':0.20,'EE':0.15,'CE':0.10},
#             'IIIT/State College': {'CSE':0.25,'ECE':0.25,'ME':0.20,'EE':0.15,'CE':0.15},
#             'Private College':    {'CSE':0.25,'ECE':0.20,'ME':0.20,'EE':0.15,'CE':0.20}
#         }
#         branches = list(branch_weights[tier].keys())
#         b_weights = list(branch_weights[tier].values())
#         branch   = random.choices(branches, weights=b_weights, k=1)[0]

#         return tier, college, branch

# fake.add_provider(EngineeringProvider)

# # ─── Static options ─────────────────────────────────────────────────
# domain_options   = ['Full Stack', 'Machine Learning', 'Android Development', 'Other']
# referral_options = ['Yes', 'No']

# # ─── Data generation ─────────────────────────────────────────────────
# num_records = 5000
# data = []

# for _ in range(num_records):
#     # Academics
#     tenth_percent   = round(np.clip(np.random.beta(2, 1.5) * 40 + 60, 60, 100), 1)
#     twelfth_percent = round(np.clip(tenth_percent * 0.9 + np.random.normal(5, 7), 60, 100), 1)
#     jee_rank        = int(np.clip(np.exp(np.random.normal(8.5, 1.2)), 1, 250000))

#     # Realistic college + branch
#     college_tier, college_name, branch = fake.real_college_branch_tier(jee_rank)

#     # Demographics & Experience
#     gender = random.choice(['Male', 'Female'])
#     exp_params = {
#         'Top IIT': {'lam': 0.8, 'max': 8},
#         'Other IIT': {'lam': 0.7, 'max': 7},
#         'NIT': {'lam': 0.6, 'max': 6},
#         'IIIT/State College': {'lam': 0.5, 'max': 5},
#         'Private College': {'lam': 0.4, 'max': 4}
#     }[college_tier]
#     experience = min(np.random.poisson(exp_params['lam']) + np.random.binomial(1, 0.2), exp_params['max'])
#     experience_field = round(np.random.uniform(0, experience), 1) if experience > 0 else 0.0

#     # Skills & Activities
#     num_projects              = np.random.poisson(lam=(experience_field + 1))
#     expertise_level           = int(np.clip(round(np.random.normal(1 + experience_field, 1)), 1, 5))
#     internships_mean          = {'Top IIT':2,'Other IIT':2,'NIT':1,'IIIT/State College':1,'Private College':0.5}
#     num_internships           = np.random.poisson(lam=internships_mean[college_tier])
#     soft_skill_rating         = int(np.clip(round(np.random.normal(3.5 if college_tier in ['Top IIT','Other IIT'] else 3.0, 0.8)), 1, 5))
#     aptitude_rating           = int(np.clip(round(np.random.normal(3.5 if college_tier in ['Top IIT','Other IIT'] else 3.0, 0.8)), 1, 5))
#     dsa_level                 = int(np.clip(round(np.random.normal(3, 1)), 1, 5))
#     hackathons_mean           = {'Top IIT':2,'Other IIT':1.5,'NIT':1,'IIIT/State College':0.8,'Private College':0.5}
#     num_hackathons            = np.random.poisson(lam=hackathons_mean[college_tier])
#     competitive_coding_solved = int(np.clip(round(np.random.normal(80 + 10 * experience_field, 30)), 1, 200))
#     num_repos                 = np.random.poisson(lam=5)
#     github_activities         = np.random.poisson(lam=10)
#     linkedin_posts            = np.random.poisson(lam=3)
#     num_certifications        = np.random.poisson(lam=1.5)
#     cgpa                      = round(np.random.beta(2, 2) * 9 + 1, 2)

#     # Extra categoricals
#     domain   = random.choice(domain_options)
#     referral = random.choice(referral_options)

#     # Salary (with branch multipliers)
#     base_salaries = {
#         'Top IIT':    np.random.lognormal(mean=13.8, sigma=0.12),
#         'Other IIT':  np.random.lognormal(mean=13.2, sigma=0.15),
#         'NIT':        np.random.lognormal(mean=12.8, sigma=0.18),
#         'IIIT/State College': np.random.lognormal(mean=12.2, sigma=0.2),
#         'Private College':     np.random.lognormal(mean=11.8, sigma=0.25)
#     }
#     branch_multipliers = {'CSE':1.7,'ECE':1.5,'EE':1.4,'ME':1.2,'CE':1.1}

#     salary = base_salaries[college_tier] \
#            * branch_multipliers[branch] \
#            * (1 + 0.08 * experience)
#     salary = int(np.clip(salary + np.random.normal(0, 50000), 300000, 2500000))

#     data.append([
#         tenth_percent, twelfth_percent, jee_rank, gender,
#         experience, experience_field,
#         college_tier, college_name, branch,
#         num_projects, expertise_level, num_internships,
#         soft_skill_rating, aptitude_rating, dsa_level, num_hackathons,
#         competitive_coding_solved, num_repos, github_activities, linkedin_posts,
#         num_certifications, cgpa, domain, referral, salary
#     ])

# # ─── Save ─────────────────────────────────────────────────────────────
# columns = [
#     '10th_percent','12th_percent','jee_rank','gender',
#     'experience','experience_field',
#     'college_tier','college_name','branch',
#     'num_projects','expertise_level','num_internships',
#     'soft_skill_rating','aptitude_rating','dsa_level','num_hackathons',
#     'competitive_coding_solved','num_repos','github_activities','linkedin_posts',
#     'num_certifications','cgpa','domain','referral','salary'
# ]

# df = pd.DataFrame(data, columns=columns)
# df.to_csv('data_csv_1.csv', index=False)

# print(f"Generated {len(df)} records")
# print(df.describe())


