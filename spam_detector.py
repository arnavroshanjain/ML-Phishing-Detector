import streamlit as st
import pickle
import string
import plotly.graph_objects as go
import plotly
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

#nltk.download('all')

# Ensure necessary NLTK resources are downloaded
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Initializing lemmatizer
lemma = WordNetLemmatizer()

# Processing the input text
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize
    
    # Removing non-alphanumeric characters
    text = [i for i in text if i.isalnum()]
    
    # Removing stopwords, punctuation
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    
    # Lemmatization
    text = [lemma.lemmatize(i) for i in text]
    
    return " ".join(text)

# Load the trained TF-IDF vectorizer.pkl and model.pkl
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"Error: {e}")
    st.stop()  # Stop execution if files are missing

st.markdown("""
    <style>
        .stButton + div {
            margin-top: 0px !important;  /* No space between button and next element */
        }

        .stButton + div + div {
            margin-top: 0px !important;  /* No space between result text/markdown and plotly chart */
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
""", unsafe_allow_html=True)

st.title("ML Driven Phishing Detector")
st.sidebar.subheader("About")


st.sidebar.info("This tool uses machine learning to detect phishing attempts in messages. Enter the message below and press 'Predict' to see the results.")


st.sidebar.markdown("""
## Tech Stack
- <i class="fas fa-code"></i> Python - Main programming language
- <i class="fab fa-python"></i> NLTK - Natural Language Toolkit for Python
- <i class="fas fa-database"></i> Scikit-Learn - Machine learning library
- <i class="fas fa-chart-line"></i> Plotly - Visualization library
- <i class="fas fa-laptop-code"></i> Streamlit - App framework
""", unsafe_allow_html=True)

st.sidebar.markdown("""
## Created by Arnav J
- <i class="fab fa-github"></i> [GitHub](https://github.com/arnavroshanjain)
- <i class="fab fa-linkedin"></i> [LinkedIn](https://linkedin.com/in/arnavrjain)
""", unsafe_allow_html=True)


# st.sidebar.markdown("## Additional Settings")


# Main content area
input_sms = st.text_area("Enter the message", height=150)

# Button for prediction
if st.button('Predict'):
    # 1. Preprocess the input text
    transformed_sms = transform_text(input_sms)
    
    # 2. Vectorize the text using the loaded TF-IDF vectorizer
    try:
        vector_input = tfidf.transform([transformed_sms])
    except Exception as e:
        st.error(f"Vectorization Error: {e}")
        st.stop()

    # 3. Predict using the loaded model
    try:
        result = model.predict(vector_input)[0]
        confidence = model.predict_proba(vector_input)[0]
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.stop()

    # # 4. Display the result and confidence
    # if result == 1:
    #     st.header("Spam")
    #     confidence_percentage = confidence[1] * 100
    #     st.subheader(f"Confidence: {confidence[1]*100:.2f}%")
    # else:
    #     st.header("Not Spam")
    #     confidence_percentage = confidence[0] * 100
    #     st.subheader(f"Confidence: {confidence[0]*100:.2f}%")

    #5. Create a gauge chart using Plotly
    if result == 1:
        confidence_percentage = confidence[1] * 100
        result_text = "Spam"
    else:
        confidence_percentage = confidence[0] * 100
        result_text = "Not Spam"

    col1, col2 = st.columns(2)


    # Create and style the gauge chart
    with col1:
        st.markdown("#")
        st.header(result_text)
        st.subheader(f"Confidence: {confidence_percentage:.2f}%")

    # gauge diagram
    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence_percentage,
            title={'text': "Confidence"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "red" if result == 1 else "green"},
                   'steps': [
                       {'range': [0, 50], 'color': "lightgray"},
                       {'range': [50, 100], 'color': "gray"}],
                   }
        ))

        # size of the gauge diagram
        fig.update_layout(autosize=False, width=400, height=300)
        st.plotly_chart(fig, use_container_width=True)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("styles.css")