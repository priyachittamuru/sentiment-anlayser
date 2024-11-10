import spacy
import re
import streamlit as st
from openai import OPENAI

# Initialize OpenAI client using Streamlit's secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"]

def analyze_sentiment_with_words(review, category):
    """
    Analyzes sentiment of the given review, providing word-level sentiment contributions in percentage.
    Also determines whether the analysis is positive or negative and the overall sentiment percentage.

    Parameters:
    review (str): The review text to analyze.
    category (str): The category of the review (e.g., 'Food', 'Product', 'Place').

    Returns:
    tuple: The response from the GPT-4 API with word-level sentiment contributions and an overall sentiment summary.
    """
    # Prepare the prompt for GPT-4
    prompt = (
        f"Analyze the following {category} review for sentiment. "
        "Provide the sentiment contribution of each significant word as a percentage and indicate whether the overall sentiment is positive or negative:\n"
        f"Review: {review}"
    )

    # Call the OpenAI GPT-4 API
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in sentiment analysis."},
                {"role": "user", "content": prompt}
            ]
        )
        content = response['choices'][0]['message']['content']
        
        # Extract overall sentiment summary using regex
        match = re.search(r'Overall sentiment: (positive|negative).*?Percentage: (\d+)%', content, re.IGNORECASE)
        if match:
            overall_sentiment = match.group(1)
            overall_percentage = match.group(2)
            return content, overall_sentiment, overall_percentage
        else:
            # Fallback if regex fails to find the pattern
            overall_sentiment = "Positive" if "positive" in content.lower() else "Negative"
            overall_percentage = "50"  # Default percentage if not found
            return content, overall_sentiment, overall_percentage

    except openai.error.OpenAIError as e:
        return f"Error with OpenAI API: {e}", "Error", "Error"

def extract_entities(review):
    """
    Extracts named entities from the given review using the SpaCy model.

    Parameters:
    review (str): The review text to extract entities from.

    Returns:
    list: A list of tuples containing entities and their labels.
    """
    try:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(review)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities
    except OSError as e:
        return f"Error loading SpaCy model: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"

def main():
    """
    Main function to get user input, analyze sentiment, and extract named entities using Streamlit.
    """
    st.title("Sentiment Analyzer")
    st.write("Enter a review and select its category to analyze sentiment and extract named entities.")

    # Get user input for category and review
    category = st.selectbox("please Select the review category", ["Food", "Product", "Place", "Other"])
    review = st.text_area("please Enter the review content")

    if st.button("Analyze Review"):
        if review:
            # Analyze sentiment with word-level contributions
            sentiment_output, overall_sentiment, overall_percentage = analyze_sentiment_with_words(review, category)
            st.subheader("Sentiment Analysis with Word-Level Contributions:")
            st.text(sentiment_output)
            st.write(f"**Overall Sentiment:** {overall_sentiment}")
            st.write(f"**Overall Sentiment Percentage:** {overall_percentage}%")

            # Extract named entities
            entities = extract_entities(review)
            if isinstance(entities, str):  # Check if an error string is returned
                st.error(entities)
            else:
                st.subheader("Named Entities:")
                if entities:
                    for entity, label in entities:
                        st.write(f"{entity} ({label})")
                else:
                    st.write("No named entities found.")
        else:
            st.warning("Please enter a review to analyze.")

if __name__ == "__main__":
    main()

