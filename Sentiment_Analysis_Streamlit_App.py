import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc, mean_squared_error
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Read social media posts from CSV file and generate true labels
def read_posts(file, text_column, pos_threshold, neg_threshold):
    try:
        df = pd.read_csv(file)

        # Check if text column exists
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in the dataset.")

        # Display the first five rows
        st.subheader("First Five Rows of Loaded CSV")
        st.dataframe(df.head())

        # Remove duplicate records
        df = df.drop_duplicates(subset=text_column)

        # Drop missing (null) values
        df = df[[text_column]].dropna()

        # Convert to string format and generate true labels using VADER
        posts = df[text_column].astype(str).tolist()
        true_labels = []
        for post in posts:
            score = analyzer.polarity_scores(post)['compound']
            true_labels.append(decode_emotion(score, pos_threshold, neg_threshold))

        st.success(f"Total posts loaded: {len(posts)}")
        return posts, true_labels, df
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None, None, None

# Decode sentiment score to emotion
def decode_emotion(score, pos_threshold, neg_threshold):
    if score >= pos_threshold:
        return "Positive"
    elif score <= neg_threshold:
        return "Negative"
    else:
        return "Neutral"

# Analyze sentiments and compute scores
def analyze_posts(posts, pos_threshold, neg_threshold):
    predicted_labels = []
    compound_scores = []
    polarity_scores = {'compound': [], 'pos': [], 'neu': [], 'neg': []}
    results_text = ""
    for post in posts:
        scores = analyzer.polarity_scores(post)
        score = scores['compound']
        emotion = decode_emotion(score, pos_threshold, neg_threshold)
        predicted_labels.append(emotion)
        compound_scores.append(score)
        polarity_scores['compound'].append(scores['compound'])
        polarity_scores['pos'].append(scores['pos'])
        polarity_scores['neu'].append(scores['neu'])
        polarity_scores['neg'].append(scores['neg'])
        results_text += f"{post[:50]}...\n-> {emotion} (Score: {score:.3f})\n\n"
    return predicted_labels, compound_scores, polarity_scores, results_text

# Compute evaluation metrics
def compute_metrics(true_labels, predicted_labels, compound_scores):
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    label_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    true_numeric = [label_mapping[label] for label in true_labels]
    predicted_numeric = [label_mapping[label] for label in predicted_labels]
    rmse = np.sqrt(mean_squared_error(true_numeric, predicted_numeric))
    cm = confusion_matrix(true_labels, predicted_labels, labels=['Positive', 'Neutral', 'Negative'])
    binary_true = [1 if label == 'Positive' else 0 for label in true_labels]
    fpr, tpr, _ = roc_curve(binary_true, compound_scores)
    roc_auc = auc(fpr, tpr)
    return accuracy, f1, rmse, cm, fpr, tpr, roc_auc

# Display pie chart
def create_pie_chart(results):
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = results.keys()
    sizes = results.values()
    colors = ['green', 'gray', 'red']
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.set_title("Emotion Distribution in Social Media Posts")
    ax.axis('equal')
    return fig

# Display bar graph
def create_bar_graph(results):
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = list(results.keys())
    counts = list(results.values())
    colors = ['green', 'gray', 'red']
    ax.bar(labels, counts, color=colors)
    ax.set_title('Sentiment Distribution in Social Media Posts')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Number of Posts')
    return fig

# Display confusion matrix
def create_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Positive', 'Neutral', 'Negative'],
                yticklabels=['Positive', 'Neutral', 'Negative'], ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('True Label (Auto-Generated)')
    ax.set_xlabel('Predicted Label')
    return fig

# Display ROC curve
def create_roc_curve(fpr, tpr, roc_auc):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve (Positive vs Non-Positive)')
    ax.legend(loc="lower right")
    return fig

# Display correlation matrix
def create_correlation_matrix(polarity_scores):
    df_scores = pd.DataFrame(polarity_scores, columns=['compound', 'pos', 'neu', 'neg'])
    corr_matrix = df_scores.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=ax)
    ax.set_title('Correlation Matrix of Sentiment Scores')
    return fig

# Main app
def main():
    st.title("📊 Sentiment Analysis App")
    st.markdown("""
    Upload a CSV file containing social media posts, specify the text column, and adjust sentiment thresholds.
    The app will analyze sentiments using VADER and display metrics and visualizations.
    """)

    # Sidebar for inputs
    with st.sidebar:
        st.header("Input Parameters")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        text_column = st.text_input("Text Column Name", value="text")
        pos_threshold = st.slider("Positive Sentiment Threshold", 0.0, 1.0, 0.5, 0.05)
        neg_threshold = st.slider("Negative Sentiment Threshold", -1.0, 0.0, -0.5, 0.05)
        analyze_button = st.button("Analyze")

    # Main content
    if analyze_button and uploaded_file is not None:
        with st.spinner("Analyzing posts..."):
            posts, true_labels, df = read_posts(uploaded_file, text_column, pos_threshold, neg_threshold)
            
            if posts is not None and true_labels is not None:
                # Analyze posts
                predicted_labels, compound_scores, polarity_scores, results_text = analyze_posts(
                    posts, pos_threshold, neg_threshold
                )

                # Compute results for visualizations
                results = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
                for label in predicted_labels:
                    results[label] += 1

                # Compute metrics
                accuracy, f1, rmse, cm, fpr, tpr, roc_auc = compute_metrics(
                    true_labels, predicted_labels, compound_scores
                )

                # Display results
                st.subheader("Analysis Results")
                
                # Sentiment analysis output
                with st.expander("Sentiment Analysis Output", expanded=False):
                    st.text_area("Post Sentiments", results_text, height=300)

                # Metrics
                st.subheader("Evaluation Metrics (Auto-Generated Labels)")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy", f"{accuracy:.3f}")
                    st.metric("F1 Score (Weighted)", f"{f1:.3f}")
                with col2:
                    st.metric("RMSE", f"{rmse:.3f}")
                    st.metric("ROC AUC", f"{roc_auc:.3f}")
                
                st.write("**Confusion Matrix (Rows: True, Columns: Predicted)**")
                cm_df = pd.DataFrame(
                    cm, 
                    index=['Positive', 'Neutral', 'Negative'],
                    columns=['Positive', 'Neutral', 'Negative']
                )
                st.dataframe(cm_df)

                # Visualizations
                st.subheader("Visualizations")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Emotion Distribution Pie Chart**")
                    st.pyplot(create_pie_chart(results))
                
                with col2:
                    st.write("**Sentiment Distribution Bar Graph**")
                    st.pyplot(create_bar_graph(results))

                st.write("**Confusion Matrix**")
                st.pyplot(create_confusion_matrix(cm))

                st.write("**ROC Curve**")
                st.pyplot(create_roc_curve(fpr, tpr, roc_auc))

                st.write("**Correlation Matrix of Sentiment Scores**")
                st.pyplot(create_correlation_matrix(polarity_scores))

    elif analyze_button and uploaded_file is None:
        st.error("Please upload a CSV file.")

if __name__ == "__main__":
    main()