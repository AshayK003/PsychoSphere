# PsychoSphere: Mental-Health Research Paper Recommendation System

## Project Overview

**PsychoSphere** is a recommendation system designed to assist researchers in the field of mental health by suggesting relevant research papers. Leveraging advanced machine learning techniques, this system provides recommendations based on the input title or keywords of a research paper.

### Key Features
- **Semantic Search:** Finds relevant research papers based on cosine similarity between embeddings of input and existing papers.
- **User-Friendly Interface:** Streamlit-based web app for easy interaction.
- **Pre-trained Models:** Uses pre-trained models for generating embeddings and making recommendations.

## Installation

To get started with the PsychoSphere recommendation system, follow these steps:

### Prerequisites

Ensure you have Python 3.8 or higher installed. You will also need to set up a virtual environment for managing dependencies.

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/psychosphere.git
   cd psychosphere
   ```

2. **Create and Activate a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   Install the required packages using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

   Create a `requirements.txt` file with the following content:

   ```plaintext
   streamlit
   torch
   sentence-transformers
   tensorflow
   numpy
   pandas
   scikit-learn
   ```

## Usage

### Running the Streamlit App

1. **Prepare the Model Files**

   Ensure that the following `.pkl` files are present in the `models` directory:
   - `embeddings.pkl`
   - `sentences.pkl`
   - `rec_model.pkl`

2. **Start the Streamlit Application**

   Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

3. **Interact with the App**

   - Open your web browser and navigate to `http://localhost:8501`.
   - Enter the title or keywords of a research paper.
   - Click "Recommend" to see the list of similar papers.

## Models and Data

### Models

- **Embedding Model:** Sentence Transformer model `all-MiniLM-L6-v2` for generating sentence embeddings.
- **Recommendation Model:** Pre-trained model to encode paper titles.

### Data

The dataset used in this project is sourced from Kaggle:

- **Dataset:** [Mental Health Research Paper Abstracts](https://www.kaggle.com/datasets/xingao89/mental-health-research-paper-abstracts)
- **Preprocessing:** The dataset was preprocessed to retain relevant columns such as `title` and `keywords`, and to handle missing or empty keyword lists.

The preprocessing steps include:
- Filtering columns to keep only `title` and `keywords`.
- Removing rows with missing or empty keyword lists.
- Flattening and extracting unique keywords for further use.

## Project Structure

```
psychosphere/
│
├── app.py                 # Streamlit application code
├── models/                 # Directory containing .pkl files
│   ├── embeddings.pkl
│   ├── sentences.pkl
│   └── rec_model.pkl
├── requirements.txt        # List of dependencies
├── papers_mental_arxiv_pubmed_biorxiv_medrxiv_acm.csv   #dataset

```

## Dependencies

- **Streamlit:** For building the web interface.
- **Torch:** For tensor operations and cosine similarity.
- **Sentence Transformers:** For sentence embeddings.
- **TensorFlow:** For loading and using the recommendation model.
- **NumPy:** For numerical operations.
- **Pandas:** For data manipulation and processing.
- **Scikit-Learn:** For model evaluation and other utilities.

## Troubleshooting

- **File Not Found Errors:** Ensure that the `models` directory contains all required `.pkl` files.
- **Version Conflicts:** Verify that the installed package versions match those specified in `requirements.txt`.
- **Errors During Model Loading:** Ensure compatibility between saved models and the current environment.

## License

This project is licensed under the MIT License.

## Acknowledgements

- **Sentence Transformers:** [Hugging Face](https://huggingface.co/sentence-transformers)
- **Streamlit:** [Streamlit Documentation](https://docs.streamlit.io)
- **Kaggle Dataset:** [Mental Health Research Paper Abstracts](https://www.kaggle.com/datasets/xingao89/mental-health-research-paper-abstracts)
