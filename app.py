import streamlit as st
import torch
from sentence_transformers import util
import pickle
import numpy as np
from tensorflow import keras

# Loading saved recommendation models
embeddings = pickle.load(open('models/embeddings.pkl', 'rb'))
sentences = pickle.load(open('models/sentences.pkl', 'rb'))
rec_model = pickle.load(open('models/rec_model.pkl', 'rb'))

# Custom functions
def recommendation(input_paper):
    if not input_paper:
        return ["Please enter some keywords or title."]
    
    try:
        # Calculating cosine similarity scores
        cosine_scores = util.cos_sim(embeddings, rec_model.encode(input_paper))

        # Get the indices of the top-k most similar papers
        top_similar_papers = torch.topk(cosine_scores, dim=0, k=10, sorted=True)

        # Retrieving the titles of the top similar papers
        papers_list = [sentences[i.item()] for i in top_similar_papers.indices]

        return papers_list

    except Exception as e:
        return [f"An error occurred: {str(e)}"]

# Creating app
st.set_page_config(
    page_title="Mental-Health Research Paper Recommendation System",
    page_icon=":mag_right:",
    layout="wide"
)

# App title and description
st.title('PsychoSphere: Mental-Health Research Paper Recommendation System')
st.markdown("""
    **Welcome to your personal mental-health research paper recommendation system!**

    This tool helps you find relevant research papers based on your input. Simply enter keywords or the title of a research paper, and get recommendations for related papers in the field of mental health.

    **How it works:**
    - **Enter Keywords or Title:** Provide the title or relevant keywords of a paper you are interested in.
    - **Click "Recommend":** Our model will analyze your input and find similar research papers.
    - **View Recommendations:** Explore a list of papers that are most relevant to your input.
""")

# Sidebar for additional controls
st.sidebar.header('Additional Information')
st.sidebar.markdown("""
    Use this app to:
    - Find top 10 relevant research papers based on your input.
    - Get recommendations for related papers in the field of mental health.

    **Tips:**
    - Use specific keywords or titles for more accurate recommendations.
    - If you don’t see relevant results, try adjusting your input terms.
""")

# Input fields and processing
with st.form(key='input_form'):
    input_paper = st.text_input("Enter Keywords or Title:", placeholder="e.g., Depression and Anxiety in Adolescents")
    submit_button = st.form_submit_button(label='Recommend')

    # Showing spinner while processing
    if submit_button:
        with st.spinner("Processing..."):
            recommend_papers = recommendation(input_paper)
            if recommend_papers:
                st.success("Recommendations generated!")
            else:
                st.error("No recommendations found. Please try a different input.")

# Displaying results
if submit_button:
    st.subheader("Recommended Papers:")
    if recommend_papers:
        for i, paper in enumerate(recommend_papers, start=1):
            st.write(f"{i}. {paper}")
    else:
        st.write("No recommendations found.")

# Footer
st.markdown("""
    ---
    Made with ❤️ by Ashay Kushwaha
""")
