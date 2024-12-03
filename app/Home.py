import streamlit as st
import pandas as pd
import numpy as np
from streamlit_folium import st_folium, folium_static
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from wordcloud import STOPWORDS
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer


def read_file():
    df = pd.read_pickle('data/df.pkl')
    return df

def filter_count(df, column, values):
    copy = df.copy()
    count_df = copy[copy[column].isin(values)]
    return count_df

def filterWordCloud(df, column, fraud):
    val = 'f' if fraud else 't'
    
    word_filter_df = df[df['fraudulent'] == val]

    data = word_filter_df[column]
    text = " ".join(data.tolist())
    return text


def makeWordCloud(text):
    fig, ax = plt.subplots()
    wordcloud = WordCloud(stopwords=set(STOPWORDS), background_color="white", random_state=42).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    st.pyplot(fig)

def makeCountPlot(df, x):
    fig, ax = plt.subplots()
    sns.countplot(data=df, x=x, hue='fraudulent', palette='coolwarm', ax=ax)
    ax.set_title(f"Count Plot for {x}")
    plt.xticks(rotation=45)
    st.pyplot(fig)

def runModel(text, model):
    print(text)
    if not text.strip():
        st.session_state["model_output"] = 'No text found. Please try again'
    else:
        vectorizer_full = load('data/vectorizer_fit.joblib')
        print("TYPE", type(vectorizer_full)) 
        text_df = pd.DataFrame({'text':[text]})

        X_test = vectorizer_full.transform(text_df['text'])
        y_pred = model.predict(X_test)
        st.session_state["model_output"] = y_pred
        
        

def main():
    if 'job_description' not in st.session_state:
        st.session_state['job_description'] = ''
    
    VAR_MAP = {'Experience': ['Internship', 'Not Applicable', 'Unspecified', 'Mid-senior level', 'Associate', 'Entry level', 'Executive', 'Director'], 
           'Employment Type': ['Full-time', 'Unspecified', 'Contract', 'Part-time', 'Temporary', 'Other'],
           'Education': ["Unspecified", "Bachelor's Degree", "High School or equivalent", "Master's Degree", "Associate Degree", "Certification", 
                         "Some College Coursework Completed", "Professional", "Vocational", "Some High School Coursework", "Doctorate", "Vocational - HS Diploma",
                           "Vocational - Degree"],
            'Function': ["Unspecified", "Information Technology", "Sales", "Engineering", "Customer Service", "Marketing", "Administrative", "Design",
                        "Health Care Provider", "Education", "Other", "Management", "Business Development", "Accounting/Auditing", "Human Resources", "Project Management",
                        "Finance", "Consulting", "Writing/Editing", "Art/Creative", "Production", "Product Management", "Quality Assurance", "Advertising", "Business Analyst",
                        "Data Analyst", "Public Relations", "Manufacturing", "General Business", "Research", "Legal", "Strategy/Planning", "Training", "Supply Chain", 
                        "Financial Analyst", "Distribution", "Purchasing", "Science"]}
    prompt = ''
    
    df = read_file()
    model = load('data/log_model_fit.joblib')
    vectorizer_full = load('data/vectorizer_fit.joblib')
    st.set_page_config(page_title="Fraudulent Job Posting Detection App", initial_sidebar_state="collapsed", layout="wide")
    with st.container():
        print(type(vectorizer_full))
        st.title("Fraudulent Job Posting Detection App")
        st.header("About")
        st.write(
            "Welcome to the fraudulent job posting detection app! This web application showcases key insights found regarding"
            "job postings across the internet and a working model used to detect fraudulent postings.")
        st.divider()
        st.header("Key Insights")
        left, right = st.columns([1,1])
        with left:
            st.subheader("Count Plot")
            x = st.select_slider(label="Check distribution of postings based on a data column", options=VAR_MAP.keys(), value="Experience")
            options = st.multiselect(
                options=VAR_MAP[x], label="Select one or many categories"
            )
            filter_df = filter_count(df, x, options)
            makeCountPlot(filter_df, x)
        with right:
            st.subheader("Word Cloud")
            st.write("")
            st.write("Look at key words based on the text category shown")
            feat = st.toggle("Fraudulent or Not", value=True)
            word_option = st.selectbox(
                "Select a text category",
                ("description", "requirements", "benefits", "company_profile"))
            text = filterWordCloud(df, word_option, feat)
            makeWordCloud(text)
        
        st.divider()
        st.header("Model")
        st.write("This model uses sentiment analysis and a logistic regression classifier to predict fraudulent job postings.")
        st.subheader("Model Metrics")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Recall", "86%")
        col2.metric("Precision", "82%")
        col3.metric("F1 Score", "84%")

        st.subheader("Predict Job Posting")
        prompt = st.text_area(label="Insert a job description")
        st.button("Run Model", on_click=lambda: runModel(prompt, model))
        if 'model_output' in st.session_state:
            if st.session_state['model_output'][0] == 0:
                st.write("Not Fraud")
            else:
                st.write("Fraud")

main()