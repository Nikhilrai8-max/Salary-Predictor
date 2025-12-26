
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved models, encoders, and preprocessor
@st.cache_resource
def load_models():
    tier_model = joblib.load('tier_model.pkl')
    name_model = joblib.load('name_model.pkl')
    branch_model = joblib.load('branch_model.pkl')
    salary_model = joblib.load('salary_model.pkl')
    name_encoder = joblib.load('name_encoder.pkl')
    branch_encoder = joblib.load('branch_encoder.pkl')
    college_encoder = joblib.load('college_encoder.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    return (tier_model, name_model, branch_model, salary_model, 
            name_encoder, branch_encoder, college_encoder, preprocessor)

(tier_model, name_model, branch_model, salary_model, 
 name_encoder, branch_encoder, college_encoder, preprocessor) = load_models()

# Define college hierarchy (same as in your notebook)
college_hierarchy = [
    'Tier 4 - Other',
    'Tier 3 - Private/State', 
    'Tier 2 - Mid Colleges',
    'Tier 1 - Other IIT/Top NIT',
    'Tier 1 - Top IIT'
]

# Streamlit app
def main():
    st.title("College Admission & Salary Predictor")
    st.write("Enter your details below to predict college tier, name, branch, and expected salary.")

    # Sidebar for input fields
    with st.sidebar:
        st.header("Student Details")
        
        # Numerical inputs
        tenth = st.number_input("10th Percentage", min_value=0.0, max_value=100.0, value=80.0, step=0.1)
        twelfth = st.number_input("12th Percentage", min_value=0.0, max_value=100.0, value=80.0, step=0.1)
        jee = st.number_input("JEE Rank", min_value=1, value=1000, step=1)
        workexp = st.number_input("Work Experience (years)", min_value=0, value=2, step=1)
        fexp = st.slider("Field Experience (years)", min_value=0, max_value=20, value=2, step=1)
        proj = st.slider("Number of Projects", min_value=0, max_value=20, value=1, step=1)
        exp_lev = st.selectbox("Expertise Level", options=[1, 2, 3, 4, 5], index=2)
        intern = st.slider("Number of Internships", min_value=0, max_value=10, value=0, step=1)
        soft = st.selectbox("Soft Skills Rating", options=[1, 2, 3, 4, 5], index=2)
        apt = st.selectbox("Aptitude Rating", options=[1, 2, 3, 4, 5], index=2)
        dsa = st.selectbox("DSA Level", options=[1, 2, 3, 4, 5], index=2)
        hack = st.slider("Number of Hackathons", min_value=0, max_value=10, value=0, step=1)
        codeqs = st.slider("Competitive Coding Questions Solved", min_value=0, max_value=200, value=50, step=10)
        repos = st.slider("Number of Repositories", min_value=0, max_value=50, value=5, step=1)
        ghacts = st.slider("GitHub Activities", min_value=0, max_value=50, value=10, step=1)
        li = st.slider("LinkedIn Posts", min_value=0, max_value=50, value=3, step=1)
        certs = st.slider("Number of Certifications", min_value=0, max_value=20, value=0, step=1)
        cgpa = st.slider("CGPA", min_value=1.0, max_value=10.0, value=7.0, step=0.1)

        # Categorical inputs
        gender = st.selectbox("Gender", options=['Male', 'Female', 'Other'], index=0)
        domain = st.selectbox("Domain", options=['Full Stack', 'Machine Learning', 'Android Development', 'Other'], index=0)
        ref = st.selectbox("Referral", options=['Yes', 'No'], index=1)

        # Predict button
        predict_button = st.button("Predict")

    # Prediction logic
    if predict_button:
        # Create input dataframe
        input_data = {
            '10th_percent': tenth,
            '12th_percent': twelfth,
            'jee_rank': jee,
            'experience': workexp,
            'experience_field': fexp,
            'num_projects': proj,
            'expertise_level': exp_lev,
            'num_internships': intern,
            'soft_skill_rating': soft,
            'aptitude_rating': apt,
            'dsa_level': dsa,
            'num_hackathons': hack,
            'competitive_coding_solved': codeqs,
            'num_repos': repos,
            'github_activities': ghacts,
            'linkedin_posts': li,
            'num_certifications': certs,
            'cgpa': cgpa,
            'gender': gender,
            'domain': domain,
            'referral': ref
        }
        inp_df = pd.DataFrame([input_data])

        # Make predictions
        t_code = tier_model.predict(inp_df)[0]
        n_code = name_model.predict(inp_df)[0]
        b_code = branch_model.predict(inp_df)[0]
        sal = salary_model.predict(inp_df)[0]

        # Decode predictions
        tier_pred = college_hierarchy[int(t_code)]
        name_pred = name_encoder.inverse_transform([[n_code]])[0][0]
        branch_pred = branch_encoder.inverse_transform([[b_code]])[0][0]
        salary_pred = sal

        # Display results
        st.subheader("Prediction Results")
        st.write(f"**College Tier**: {tier_pred}")
        st.write(f"**College Name**: {name_pred}")
        st.write(f"**Branch**: {branch_pred}")
        st.write(f"**Expected Salary**: ₹{salary_pred:,.2f}")

if __name__ == "__main__":
    main()





    
# import streamlit as st
# import joblib
# import pandas as pd

# # Load trained models
# college_model = joblib.load("college_predictor.pkl")
# salary_model = joblib.load("salary_predictor.pkl")

# st.title("🎓 Career Predictor App")
# st.markdown("Predict your **college** and **expected salary** based on your profile.")

# # Input fields
# tenth = st.number_input("10th %", min_value=0.0, max_value=100.0, step=0.1)
# twelfth = st.number_input("12th %", min_value=0.0, max_value=100.0, step=0.1)
# jee_rank = st.number_input("JEE Rank", min_value=1, step=1)
# work_exp = st.number_input("Work Experience (years)", min_value=0, step=1)
# field_exp = st.number_input("Field Experience (years)", min_value=0, step=1)
# projects = st.number_input("Number of Projects", min_value=0, step=1)
# expertise = st.slider("Expertise Level (1-5)", 1, 5)
# internships = st.number_input("Number of Internships", min_value=0, step=1)
# soft_skills = st.slider("Soft Skill Rating (1-5)", 1, 5)
# aptitude = st.slider("Aptitude Rating (1-5)", 1, 5)
# dsa_level = st.slider("DSA Level (1-5)", 1, 5)
# hackathons = st.number_input("Hackathons Participated", min_value=0, step=1)
# coding_qs = st.number_input("Competitive Coding Questions Solved", min_value=0, step=1)
# repos = st.number_input("Number of GitHub Repositories", min_value=0, step=1)
# github_acts = st.number_input("GitHub Contributions", min_value=0, step=1)
# linkedin_posts = st.number_input("LinkedIn Posts", min_value=0, step=1)
# certifications = st.number_input("Number of Certifications", min_value=0, step=1)
# cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.01)

# # Keep as strings — do not encode manually
# gender = st.selectbox("Gender", ["Male", "Female"])
# domain = st.selectbox("Preferred Domain", ["Full Stack", "Data Science", "AI", "Cybersecurity", "Other"])
# referral = st.selectbox("Got Referral?", ["Yes", "No"])

# # Define DataFrame with original column names used during training
# features_df = pd.DataFrame([{
#     '10th_percent': tenth,
#     '12th_percent': twelfth,
#     'jee_rank': jee_rank,
#     'experience': work_exp,
#     'experience_field': field_exp,
#     'num_projects': projects,
#     'expertise_level': expertise,
#     'num_internships': internships,
#     'soft_skill_rating': soft_skills,
#     'aptitude_rating': aptitude,
#     'dsa_level': dsa_level,
#     'num_hackathons': hackathons,
#     'competitive_coding_solved': coding_qs,
#     'num_repos': repos,
#     'github_activities': github_acts,
#     'linkedin_posts': linkedin_posts,
#     'num_certifications': certifications,
#     'cgpa': cgpa,
#     'gender': gender,
#     'domain': domain,
#     'referral': referral
# }])

# # Predict on button click
# if st.button("Predict"):
#     try:
#         college = college_model.predict(features_df)[0]
#         salary = salary_model.predict(features_df)[0]

#         st.success(f"🎓 **Predicted College Tier:** {college}")
#         st.success(f"💰 **Expected Salary:** ₹{salary:,.2f}")
#     except Exception as e:
#         st.error(f"⚠️ Prediction failed: {e}")




# import streamlit as st
# import joblib
# import pandas as pd

# # Load trained models
# college_model_dict = joblib.load("college_prediction_system.pkl")
# salary_model_dict = joblib.load("salary_prediction_system.pkl")

# # Extract actual models
# college_model = college_model_dict['model']  # or the correct key in your dict
# salary_model = salary_model_dict['model']

# # Custom CSS for styling
# st.markdown("""
#     <style>
#     .main-title {
#         font-size: 200px;  /* Larger heading */
#         color: #2c3e50;
#         text-align: center;
#         font-weight: bold;
#     }
#     .subtitle {
#         font-size: 20px;
#         color: #7f8c8d;
#         text-align: center;
#         margin-bottom: 30px;
#     }
#     .section-header {
#         font-size: 28px;
#         color: #2980b9;
#         margin-top: 20px;
#         font-weight: bold;
#     }
#     .stButton>button {
#         background-color: #27ae60;
#         color: white;
#         font-size: 18px;
#         padding: 10px 20px;
#         border-radius: 10px;
#     }
#     .stSuccess {
#         font-size: 20px;
#         font-weight: bold;
#     }
#     .sidebar .sidebar-content {
#         background-color: #2c3e50;
#         color: white;
#     }
#     .sidebar .sidebar-content a {
#         color: #e74c3c;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # App Header
# st.markdown('<p class="main-title">🎓 Career Predictor Pro</p>', unsafe_allow_html=True)
# st.markdown('<p class="subtitle">Unlock Your Future: Predict Your College Tier & Expected Salary</p>', unsafe_allow_html=True)

# # Sidebar with Predictor, About, and Help
# with st.sidebar:
#     st.markdown('<h2 style="color: white;">Navigation</h2>', unsafe_allow_html=True)
#     st.image("https://via.placeholder.com/150", caption="Your Career Journey", use_container_width=True)
    
#     # Predictor Section
#     st.markdown("### 📊 Predictor")
#     st.write("Enter your academic and professional details to get personalized predictions!")
    
#     # About Section
#     st.markdown("### ℹ️ About")
#     st.write("Learn more about Career Predictor Pro and how it can guide your career path.")
    
#     # Help Section
#     st.markdown("### ❓ Help")
#     st.write("Need assistance? Find FAQs or contact support for guidance.")
    
#     st.markdown("Built with ❤️ by [Your Name]")

# # Tabs for better organization
# tab1, tab2, tab3 = st.tabs(["📚 Academic Details", "💼 Professional Details", "🔍 Predictions"])

# # Academic Details Tab
# with tab1:
#     st.markdown('<p class="section-header">📚 Academic Background</p>', unsafe_allow_html=True)
#     col1, col2 = st.columns(2)
    
#     with col1:
#         tenth = st.number_input("10th % 📖", min_value=0.0, max_value=100.0, step=0.1, help="Your 10th grade percentage")
#         twelfth = st.number_input("12th % 📘", min_value=0.0, max_value=100.0, step=0.1, help="Your 12th grade percentage")
#         jee_rank = st.number_input("JEE Rank 🏆", min_value=1, step=1, help="Your JEE rank")
    
#     with col2:
#         cgpa = st.number_input("CGPA 🎓", min_value=0.0, max_value=10.0, step=0.01, help="Your college CGPA")
#         certifications = st.number_input("Certifications 🏅", min_value=0, step=1, help="Number of certifications earned")
    
#     st.progress(min(tenth / 100, 1.0), text="10th % Progress")
#     st.progress(min(twelfth / 100, 1.0), text="12th % Progress")

# # Professional Details Tab
# with tab2:
#     st.markdown('<p class="section-header">💼 Professional Experience</p>', unsafe_allow_html=True)
#     col1, col2 = st.columns(2)
    
#     with col1:
#         work_exp = st.number_input("Work Experience (years) 🕒", min_value=0, step=1, help="Years of work experience")
#         field_exp = st.number_input("Field Experience (years) 🌐", min_value=0, step=1, help="Years in your domain")
#         projects = st.number_input("Projects Completed 🚀", min_value=0, step=1, help="Number of projects")
#         internships = st.number_input("Internships 🎯", min_value=0, step=1, help="Number of internships")
#         hackathons = st.number_input("Hackathons 🖥️", min_value=0, step=1, help="Hackathons participated")
    
#     with col2:
#         expertise = st.slider("Expertise Level (1-5) 🌟", 1, 5, help="Rate your expertise")
#         soft_skills = st.slider("Soft Skills (1-5) 🤝", 1, 5, help="Rate your soft skills")
#         aptitude = st.slider("Aptitude (1-5) 🧠", 1, 5, help="Rate your aptitude")
#         dsa_level = st.slider("DSA Level (1-5) 💻", 1, 5, help="Data Structures & Algorithms level")
#         coding_qs = st.number_input("Coding Questions Solved 🧩", min_value=0, step=1, help="Competitive coding questions")
    
#     st.markdown("### Online Presence")
#     repos = st.number_input("GitHub Repos 📂", min_value=0, step=1, help="Number of GitHub repositories")
#     github_acts = st.number_input("GitHub Contributions 🌍", min_value=0, step=1, help="Total GitHub contributions")
#     linkedin_posts = st.number_input("LinkedIn Posts 📝", min_value=0, step=1, help="Number of LinkedIn posts")

# # Predictions Tab
# with tab3:
#     st.markdown('<p class="section-header">🔍 Your Profile</p>', unsafe_allow_html=True)
#     col1, col2 = st.columns(2)
    
#     with col1:
#         gender = st.selectbox("Gender 👤", ["Male", "Female"], help="Select your gender")
#         domain = st.selectbox("Preferred Domain 🌐", ["Full Stack", "Data Science", "AI", "Cybersecurity", "Other"], help="Your career domain")
    
#     with col2:
#         referral = st.selectbox("Got Referral? 🤝", ["Yes", "No"], help="Do you have a referral?")
    
#     # DataFrame for prediction
#     features_df = pd.DataFrame([{
#         '10th_percent': tenth,
#         '12th_percent': twelfth,
#         'jee_rank': jee_rank,
#         'experience': work_exp,
#         'experience_field': field_exp,
#         'num_projects': projects,
#         'expertise_level': expertise,
#         'num_internships': internships,
#         'soft_skill_rating': soft_skills,
#         'aptitude_rating': aptitude,
#         'dsa_level': dsa_level,
#         'num_hackathons': hackathons,
#         'competitive_coding_solved': coding_qs,
#         'num_repos': repos,
#         'github_activities': github_acts,
#         'linkedin_posts': linkedin_posts,
#         'num_certifications': certifications,
#         'cgpa': cgpa,
#         'gender': gender,
#         'domain': domain,
#         'referral': referral
#     }])
    
#     # Predict Button
#     if st.button("🚀 Predict My Future"):
#         with st.spinner("Analyzing your profile..."):
#             try:
#                 college = college_model.predict(features_df)[0]
#                 salary = salary_model.predict(features_df)[0]
                
#                 st.markdown("### 🎉 Prediction Results")
#                 st.success(f"🎓 **Predicted College Tier:** {college}")
#                 st.success(f"💰 **Expected Salary:** ₹{salary:,.2f}")
                
#                 # Visualization
#                 st.bar_chart({"College Tier": [college], "Salary (Lakhs)": [salary / 100000]})
#             except Exception as e:
#                 st.error(f"⚠️ Oops! Something went wrong: {e}")

# # Footer
# st.markdown("---")
# st.write("© 2025 Career Predictor Pro | Powered by Streamlit")





# import streamlit as st
# import joblib
# import pandas as pd

# # Load individual models
# tier_model = joblib.load("tier_model.pkl")
# name_model = joblib.load("name_model.pkl")
# branch_model = joblib.load("branch_model.pkl")
# salary_model = joblib.load("salary_model.pkl")

# # Load encoders
# name_encoder = joblib.load("name_encoder.pkl")
# branch_encoder = joblib.load("branch_encoder.pkl")
# college_encoder = joblib.load("college_encoder.pkl")

# # Load preprocessor
# preprocessor = joblib.load("preprocessor.pkl")

# # Custom CSS for styling
# st.markdown("""
#     <style>
#     .main-title {
#         font-size: 60px;
#         color: #2c3e50;
#         text-align: center;
#         font-weight: bold;
#     }
#     .subtitle {
#         font-size: 20px;
#         color: #7f8c8d;
#         text-align: center;
#         margin-bottom: 30px;
#     }
#     .section-header {
#         font-size: 28px;
#         color: #2980b9;
#         margin-top: 20px;
#         font-weight: bold;
#     }
#     .stButton>button {
#         background-color: #27ae60;
#         color: white;
#         font-size: 18px;
#         padding: 10px 20px;
#         border-radius: 10px;
#     }
#     .stSuccess {
#         font-size: 20px;
#         font-weight: bold;
#     }
#     .sidebar .sidebar-content {
#         background-color: #2c3e50;
#         color: white;
#     }
#     .sidebar .sidebar-content a {
#         color: #e74c3c;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # App Header
# st.markdown('<p class="main-title">🎓 Career Predictor Pro</p>', unsafe_allow_html=True)
# st.markdown('<p class="subtitle">Unlock Your Future: Predict Your College Tier, Name, Branch & Expected Salary</p>', unsafe_allow_html=True)

# # Sidebar Navigation
# with st.sidebar:
#     st.markdown('<h2 style="color: white;">Navigation</h2>', unsafe_allow_html=True)
#     st.image("https://via.placeholder.com/150", caption="Your Career Journey", use_container_width=True)
#     st.markdown("### 📊 Predictor")
#     st.write("Enter your academic and professional details to get personalized predictions!")
#     st.markdown("### ℹ️ About")
#     st.write("Career Predictor Pro uses machine learning to guide your college and career path.")
#     st.markdown("### ❓ Help")
#     st.write("Find FAQs or contact support for assistance.")
#     st.markdown("Built with ❤️ by Samya Vig")

# # Tabs
# tab1, tab2, tab3 = st.tabs(["📚 Academic Details", "💼 Professional Details", "🔍 Predictions"])

# # Tab 1: Academic Details
# with tab1:
#     st.markdown('<p class="section-header">📚 Academic Background</p>', unsafe_allow_html=True)
#     col1, col2 = st.columns(2)

#     with col1:
#         tenth = st.number_input("10th % 📖", min_value=0.0, max_value=100.0, step=0.1)
#         twelfth = st.number_input("12th % 📘", min_value=0.0, max_value=100.0, step=0.1)
#         jee_rank = st.number_input("JEE Rank 🏆", min_value=1, step=1)

#     with col2:
#         cgpa = st.number_input("CGPA 🎓", min_value=0.0, max_value=10.0, step=0.01)
#         certifications = st.number_input("Certifications 🏅", min_value=0, step=1)

#     st.progress(min(tenth / 100, 1.0), text="10th % Progress")
#     st.progress(min(twelfth / 100, 1.0), text="12th % Progress")

# # Tab 2: Professional Details
# with tab2:
#     st.markdown('<p class="section-header">💼 Professional Experience</p>', unsafe_allow_html=True)
#     col1, col2 = st.columns(2)

#     with col1:
#         work_exp = st.number_input("Work Experience (years) 🕒", min_value=0, step=1)
#         field_exp = st.number_input("Field Experience (years) 🌐", min_value=0, step=1)
#         projects = st.number_input("Projects Completed 🚀", min_value=0, step=1)
#         internships = st.number_input("Internships 🎯", min_value=0, step=1)
#         hackathons = st.number_input("Hackathons 🖥️", min_value=0, step=1)

#     with col2:
#         expertise = st.slider("Expertise Level (1-5) 🌟", 1, 5)
#         soft_skills = st.slider("Soft Skills (1-5) 🤝", 1, 5)
#         aptitude = st.slider("Aptitude (1-5) 🧠", 1, 5)
#         dsa_level = st.slider("DSA Level (1-5) 💻", 1, 5)
#         coding_qs = st.number_input("Coding Questions Solved 🧩", min_value=0, step=1)

#     st.markdown("### Online Presence")
#     repos = st.number_input("GitHub Repos 📂", min_value=0, step=1)
#     github_acts = st.number_input("GitHub Contributions 🌍", min_value=0, step=1)
#     linkedin_posts = st.number_input("LinkedIn Posts 📝", min_value=0, step=1)

# # Tab 3: Predictions
# with tab3:
#     st.markdown('<p class="section-header">🔍 Your Profile</p>', unsafe_allow_html=True)
#     col1, col2 = st.columns(2)

#     with col1:
#         gender = st.selectbox("Gender 👤", ["Male", "Female"])
#         domain = st.selectbox("Preferred Domain 🌐", ["Full Stack", "Data Science", "AI", "Cybersecurity", "Other"])

#     with col2:
#         referral = st.selectbox("Got Referral? 🤝", ["Yes", "No"])

#     # Collect features into a DataFrame
#     features_df = pd.DataFrame([{
#         '10th_percent': tenth,
#         '12th_percent': twelfth,
#         'jee_rank': jee_rank,
#         'experience': work_exp,
#         'experience_field': field_exp,
#         'num_projects': projects,
#         'expertise_level': expertise,
#         'num_internships': internships,
#         'soft_skill_rating': soft_skills,
#         'aptitude_rating': aptitude,
#         'dsa_level': dsa_level,
#         'num_hackathons': hackathons,
#         'competitive_coding_solved': coding_qs,
#         'num_repos': repos,
#         'github_activities': github_acts,
#         'linkedin_posts': linkedin_posts,
#         'num_certifications': certifications,
#         'cgpa': cgpa,
#         'gender': gender,
#         'domain': domain,
#         'referral': referral
#     }])

#     # Prediction
#     if st.button("🚀 Predict My Future"):
#         with st.spinner("Analyzing your profile..."):
#             try:
#                 # Transform data
#                 transformed_input = preprocessor.transform(features_df)

#                 # Make predictions
#                 tier_pred = tier_model.predict(transformed_input)[0]
#                 name_pred = name_model.predict(transformed_input)[0]
#                 branch_pred = branch_model.predict(transformed_input)[0]
#                 salary_pred = salary_model.predict(transformed_input)[0]

#                 # Decode results
#                 decoded_tier = college_encoder.inverse_transform([tier_pred])[0]
#                 decoded_name = name_encoder.inverse_transform([name_pred])[0]
#                 decoded_branch = branch_encoder.inverse_transform([branch_pred])[0]

#                 # Display
#                 st.markdown("### 🎉 Prediction Results")
#                 st.success(f"🏫 **College Tier:** {decoded_tier}")
#                 st.success(f"🏛️ **College Name:** {decoded_name}")
#                 st.success(f"📚 **Branch:** {decoded_branch}")
#                 st.success(f"💰 **Expected Salary:** ₹{salary_pred:,.2f}")

#                 st.bar_chart({
#                     "Tier (Encoded)": [tier_pred],
#                     "Salary (in Lakhs)": [salary_pred / 100000]
#                 })

#             except Exception as e:
#                 st.error(f"⚠️ Error during prediction: {e}")

# # Footer
# st.markdown("---")
# st.write("© 2025 Career Predictor Pro | Powered by Streamlit")


