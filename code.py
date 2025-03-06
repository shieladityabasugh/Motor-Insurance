import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
#from google.colab import files
from sklearn.metrics import jaccard_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import ratio
import numpy as np
import streamlit as st
st.set_page_config(page_title="Damage Assessment System", layout="wide")
#!pip install python-Levenshtein

import nltk

# Step 1: Download required NLTK tokenizer
nltk.download('punkt')

from nltk import punkt
# Step 2: Load data
primary_parts = pd.read_excel("Primary_Parts_Code.xlsx")
garage_data = pd.read_csv("preview_garage_data.csv")
surveyor_data = pd.read_csv("preview_surveyor_data.csv")


# Abbreviation Mapping
abbreviation_map = {
    r"\br\b": "right",
    r"\bl\b": "left",
    r"\brr\b": "rear",
    r"\bassy\b": "assembly",
    r"\bfr\b": "front",
    r"\brh\b": "right hand",
    r"\blh\b": "left hand",
    r"\bfrt\b": "front",
    r"\bqtr\b": "quarter"
}

# Text Cleaning Function
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # Remove special characters
    for abbr, full_form in abbreviation_map.items():
        text = re.sub(abbr, full_form, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Apply Cleaning
garage_data["CLEAN_PARTDESCRIPTION"] = garage_data["PARTDESCRIPTION"].apply(clean_text)
surveyor_data["CLEAN_TXT_PARTS_NAME"] = surveyor_data["TXT_PARTS_NAME"].apply(clean_text)

#Vectorizing for TF-IDF Cosine Similarity
vectorizer = TfidfVectorizer()
all_texts = list(surveyor_data["CLEAN_TXT_PARTS_NAME"]) + list(garage_data["CLEAN_PARTDESCRIPTION"])
tfidf_matrix = vectorizer.fit_transform(all_texts)

# Split matrices
survey_tfidf = tfidf_matrix[:len(surveyor_data)]
garage_tfidf = tfidf_matrix[len(surveyor_data):]

# Compute Similarity in One Step
similarity_matrix = cosine_similarity(survey_tfidf, garage_tfidf)

# Get Best Matches
best_matches = similarity_matrix.argmax(axis=1)
best_scores = similarity_matrix.max(axis=1)

# Apply Threshold
from fuzzywuzzy import fuzz

threshold = 0.5  # Set minimum similarity score for matching
matching_results = []
for i, idx in enumerate(best_matches):
    tfidf_score = best_scores[i]
    fuzzy_score = fuzz.token_sort_ratio(surveyor_data.iloc[i]["CLEAN_TXT_PARTS_NAME"], garage_data.iloc[idx]["CLEAN_PARTDESCRIPTION"]) / 100.0
    final_score = max(tfidf_score, fuzzy_score)
    method_used = "TF-IDF" if final_score == tfidf_score else "Fuzzy Matching"

    if final_score >= threshold:
        matching_results.append((
            surveyor_data.iloc[i]["CLEAN_TXT_PARTS_NAME"],
            garage_data.iloc[idx]["CLEAN_PARTDESCRIPTION"],
            final_score,
            method_used
        ))

# Convert results to DataFrame
matching_df = pd.DataFrame(matching_results, columns=["Surveyor Part", "Best Matched Garage Part", "Similarity Score", "Method Used"])
matching_df.to_csv("part_matching_results.csv", index=False)

# Save cleaned data
garage_data.to_csv("cleaned_garage_data.csv", index=False)
surveyor_data.to_csv("cleaned_surveyor_data.csv", index=False)



# Most Commonly Damaged Parts
common_parts = surveyor_data["CLEAN_TXT_PARTS_NAME"].value_counts().reset_index()
common_parts.columns = ["Part Name", "Count"]
common_parts["Percentage"] = (common_parts["Count"] / common_parts["Count"].sum()) * 100
common_parts.to_csv("common_damaged_parts.csv", index=False)
#files.download("common_damaged_parts.csv")

# Apply Cleaning
Primary_Parts_Code=pd.read_excel(r"C:\Users\Himani Grover\Downloads\Primary_Parts_Code.xlsx")
part_matching_results=matching_df
Primary_Parts_Code["CLEAN_SURVEYOR_PART_NAME"] = Primary_Parts_Code["Surveyor Part Name"].astype(str)
part_matching_results["CLEAN_SURVEYOR_PART"] = part_matching_results["Surveyor Part"].astype(str)

# Vectorizing for TF-IDF Cosine Similarity
vectorizer = TfidfVectorizer()
all_texts = list(part_matching_results["CLEAN_SURVEYOR_PART"]) + list(Primary_Parts_Code["CLEAN_SURVEYOR_PART_NAME"])
tfidf_matrix = vectorizer.fit_transform(all_texts)

# Split matrices
matching_tfidf = tfidf_matrix[:len(part_matching_results)]
primary_tfidf = tfidf_matrix[len(Primary_Parts_Code):]

# Compute Similarity
similarity_matrix = cosine_similarity(matching_tfidf, primary_tfidf)

# Get Best Matches Using TF-IDF and Fuzzy Matching
best_match_results = []
thresh = 0.5  # Similarity threshold
for i, row in part_matching_results.iterrows():
    best_score = 0
    best_match = None
    best_method = ""
    for j, primary_row in Primary_Parts_Code.iterrows():
        tfidf_score = similarity_matrix[i, j]
        fuzzy_score = fuzz.token_sort_ratio(row["CLEAN_SURVEYOR_PART"], primary_row["CLEAN_SURVEYOR_PART_NAME"]) / 100.0
        final_score = max(tfidf_score, fuzzy_score)
        method_used = "TF-IDF" if final_score == tfidf_score else "Fuzzy Matching"

        if final_score > best_score:
            best_score = final_score
            best_match = primary_row["Surveyor Part Name"]
            best_method = method_used

    if best_score >= thresh:
        best_match_results.append((
            row["Surveyor Part"],
            best_match,
            best_score,
            best_method
        ))

# Convert results to DataFrame
best_match_df = pd.DataFrame(best_match_results, columns=["Surveyor Part", "Matched Surveyor Part Name", "Similarity Score", "Method Used"])
best_match_df.to_csv("best_part_matches.csv", index=False)

import pandas as pd

# Define the match_primary_part function
def match_primary_part(row, primary_parts_data):
    # Example function logic (replace with your actual logic)
    part_name = row["Surveyor Part"]
    similarity_scores = primary_parts_data["Surveyor Part Name"].apply(lambda x: fuzz.token_sort_ratio(part_name, x))
    best_match_idx = similarity_scores.idxmax()
    final_primary_part_name = primary_parts_data.loc[best_match_idx, "Surveyor Part Name"]
    final_similarity_score = similarity_scores.max() / 100.0
    return pd.Series([final_primary_part_name, final_similarity_score])

# Load your data
primary_parts_data = pd.read_excel(r"C:\Users\Himani Grover\Downloads\Primary_Parts_Code.xlsx")
part_matching_results = matching_df

# Remove duplicates
part_matching_results = part_matching_results.drop_duplicates(subset=["Surveyor Part"])
best_match_df = best_match_df.drop_duplicates(subset=["Surveyor Part"])
primary_parts_data = primary_parts_data.drop_duplicates(subset=["Surveyor Part Name"])

# Merge DataFrames
merged_df = part_matching_results.merge(best_match_df, on="Surveyor Part", how="left")

# Apply Matching Function
merged_df[["Final Primary Part Name", "Final Similarity Score"]] = merged_df.apply(lambda row: match_primary_part(row, primary_parts_data), axis=1, result_type="expand")

# Final Merge with Primary Parts Data
integrated_df = merged_df.merge(primary_parts_data, left_on="Final Primary Part Name", right_on="Surveyor Part Name", how="left")

# Save final integrated data
integrated_df.to_csv("final_integrated_part_mapping.csv", index=False)

# Visualization: Most Commonly Matched Primary Parts
plt.figure(figsize=(12,6))
sns.barplot(x=integrated_df["Final Similarity Score"].head(10), y=integrated_df["Final Primary Part Name"].head(10), palette="coolwarm")
plt.xlabel("Similarity Score")
plt.ylabel("Final Primary Part Name")
plt.title("Top Matched Primary Parts with Similarity Scores")
plt.show()

# Additional Analysis: Most Frequently Damaged Parts
most_common_parts = surveyor_data["TXT_PARTS_NAME"].value_counts().head(10)
plt.figure(figsize=(12,6))
sns.barplot(x=most_common_parts.values, y=most_common_parts.index, palette="viridis")
plt.xlabel("Count")
plt.ylabel("Damaged Part")
plt.title("Top 10 Most Frequently Damaged Parts")
plt.show()

# Compute Percentage of Most Frequently Damaged Parts
most_common_parts = surveyor_data["TXT_PARTS_NAME"].value_counts()
most_common_parts_df = most_common_parts.reset_index()
most_common_parts_df.columns = ["Damaged Part", "Count"]
most_common_parts_df["Percentage"] = (most_common_parts_df["Count"] / most_common_parts_df["Count"].sum()) * 100
most_common_parts_df


# Save the DataFrame to a CSV file
most_common_parts_df.to_csv("most_common_parts.csv", index=False)

# Visualization: Most Frequently Damaged Parts
plt.figure(figsize=(12,6))
sns.barplot(x=most_common_parts_df["Count"].head(10), y=most_common_parts_df["Damaged Part"].head(10), palette="viridis")
plt.xlabel("Count")
plt.ylabel("Damaged Part")
plt.title("Top 10 Most Frequently Damaged Parts")
plt.show()

# Save final integrated data
integrated_df.to_csv("final_integrated_part_mapping.csv", index=False)
most_common_parts_df.to_csv("most_common_damaged_parts.csv", index=False)

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load Claims Data
def load_data():
    df = pd.read_csv("part_matching_results.csv")  # Using correct file name
    print("Available Columns:", df.columns.tolist())  # Debugging step
    return df

# Preprocess Data for Market Basket Analysis
def preprocess_data(data):
    # Verify the expected column is present
    if "Surveyor Part" not in data.columns:
        raise KeyError("Column 'Surveyor Part' not found in dataset.")

    # Group by Claim ID or an alternative unique identifier
    if "Claim ID" in data.columns:
        claim_col = "Claim ID"
    else:
        data["Claim ID"] = range(1, len(data) + 1)  # Generate unique claim IDs if missing
        claim_col = "Claim ID"

    # Create transactions per claim
    grouped = data.groupby(claim_col)["Surveyor Part"].apply(list)
    transactions = grouped.tolist()

    # Convert transactions to one-hot encoded DataFrame
    unique_parts = sorted(set(part for sublist in transactions for part in sublist))
    encoded_data = pd.DataFrame([[1 if part in claim else 0 for part in unique_parts] for claim in transactions],
                                columns=unique_parts).astype(bool)  # Convert to boolean

    return encoded_data, unique_parts

# Apply Apriori Algorithm
def generate_association_rules(encoded_data, min_support=0.02, min_confidence=0.3):
    frequent_itemsets = apriori(encoded_data, min_support=min_support, use_colnames=True)

    if frequent_itemsets.empty:
        print("No frequent itemsets found. Try lowering min_support.")
        return pd.DataFrame()

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return rules

# Get Recommendations Based on Selected Parts
def get_recommendations(selected_parts, rules, top_n=5):
    if rules.empty:
        return []

    recommendations = []
    for part in selected_parts:
        matches = rules[rules["antecedents"].apply(lambda x: part in x)]
        sorted_matches = matches.sort_values(by="confidence", ascending=False).head(top_n)
        recommendations.extend(sorted_matches["consequents"].explode().unique())

    return list(set(recommendations) - set(selected_parts))  # Remove already selected parts

# Example Usage
if __name__ == "__main__":
    data = load_data()

    encoded_data, unique_parts = preprocess_data(data)
    rules = generate_association_rules(encoded_data)

    if not rules.empty:
        selected_parts = ["Front Bumper", "Rear Door"]  # Example user selection
        recommendations = get_recommendations(selected_parts, rules)
        print("Recommended Associated Parts:", recommendations)
    else:
        print("No association rules generated.")

#!pip install streamlit



'''import pandas as pd
import matplotlib.pyplot as plt

from fuzzywuzzy import process
from mlxtend.frequent_patterns import apriori, association_rules

st.title("🚗 Damage Assessment System")

@st.cache_data
def load_data():
    try:
        surveyor_data = pd.read_csv("part_matching_results.csv")
        if "Damaged Part" not in surveyor_data.columns:
            surveyor_data.rename(columns={surveyor_data.columns[0]: "Damaged Part"}, inplace=True)
        if "Count" not in surveyor_data.columns:
            surveyor_data["Count"] = 1
    except FileNotFoundError:
        surveyor_data = pd.DataFrame({
            "Claim ID": [1, 2, 3, 4, 5],
            "Damaged Part": ["Front Bumper", "Rear Door", "Side Mirror", "Hood", "Roof"],
            "Count": [120, 90, 75, 60, 50],
        })
    surveyor_data["Percentage"] = (surveyor_data["Count"] / surveyor_data["Count"].sum()) * 100
    return surveyor_data

data = load_data()''''

# Search functionality
st.subheader("🔍 Search for Damaged Parts")
search_term = st.text_input("Enter part name")
if search_term and "Damaged Part" in data.columns:
    best_match, score = process.extractOne(search_term, data["Damaged Part"].tolist())
    if score >= 60:
        st.write(f"Did you mean: **{best_match}**? (Match Score: {score}%)")
        filtered_data = data[data["Damaged Part"] == best_match]
    else:
        st.write("No close match found.")
        filtered_data = data
else:
    filtered_data = data
st.write(filtered_data)

# Streamlit UI

st.title("🚗 Damage Assessment System")

# Sidebar for Uploading Files
st.sidebar.header("Upload Surveyor Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    if "Damaged Part" not in data.columns:
        data.rename(columns={data.columns[0]: "Damaged Part"}, inplace=True)
    if "Count" not in data.columns:
        data["Count"] = 1  # Ensure count exists
    data["Percentage"] = (data["Count"] / data["Count"].sum()) * 100
    st.sidebar.success("File Uploaded Successfully")

# Search Functionality
st.subheader("🔍 Search for Damaged Parts")
search_term = st.text_input("Enter part name")
if search_term and "Damaged Part" in data.columns:
    best_match, score = process.extractOne(search_term, data["Damaged Part"].tolist())
    st.write(f"Did you mean: **{best_match}**? (Match Score: {score}%)")
    filtered_data = data[data["Damaged Part"] == best_match]
else:
    filtered_data = data
st.write(filtered_data)

# Display Bar Chart
st.subheader("📊 Most Frequently Damaged Parts")
fig, ax = plt.subplots()
ax.barh(data["Damaged Part"], data["Count"], color="skyblue")
ax.set_xlabel("Claim Count")
ax.set_ylabel("Damaged Part")
ax.set_title("Top Damaged Parts by Claim Count")
st.pyplot(fig)

# Function to Generate Recommendations Using Apriori Algorithm
def generate_association_rules(data, min_support=0.05, min_confidence=0.3):
    if "Claim ID" not in data.columns:
        data["Claim ID"] = range(1, len(data) + 1)  # Ensure unique claim IDs
    grouped = data.groupby("Claim ID")["Damaged Part"].apply(list)
    transactions = grouped.tolist()
    unique_parts = sorted(set(part for sublist in transactions for part in sublist))
    encoded_data = pd.DataFrame([[1 if part in claim else 0 for part in unique_parts] for claim in transactions],
                                columns=unique_parts).astype(bool)
    frequent_itemsets = apriori(encoded_data, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return rules

# Recommendations Based on Selection
st.subheader("🛠️ Recommended Associated Parts")
selected_parts = st.multiselect("Select damaged parts", data["Damaged Part"] if "Damaged Part" in data.columns else [])

def get_recommendations(selected_parts, rules, top_n=5):
    if rules.empty:
        return []
    recommendations = []
    for part in selected_parts:
        matches = rules[rules["antecedents"].apply(lambda x: part in x)]
        sorted_matches = matches.sort_values(by="confidence", ascending=False).head(top_n)
        recommendations.extend(sorted_matches["consequents"].explode().unique())
    return list(set(recommendations) - set(selected_parts))  # Remove already selected parts

if selected_parts:
    rules = generate_association_rules(data)
    recommended_parts = get_recommendations(selected_parts, rules)
    st.write("Recommended parts associated with selected damages:")
    st.write(recommended_parts)