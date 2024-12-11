# Import necessary libraries
import re
import csv
import pdfplumber
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import os
from datetime import datetime

# Initialize the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define section titles and their synonyms for classification
section_synonyms = {
    "Education": ["Education", "Academics", "Studies", "Learning", "Degree"],
    "Experience": ["Experience", "Work History", "Professional Experience", "Career"],
    "Skills": ["Skills", "Technical Skills", "Competencies"],
    "Certifications": ["Certifications", "Credentials", "Licenses", "Qualifications"],
    "Summary": ["Summary", "Profile", "About Me", "Introduction"]
}

# Define section labels with synonyms for each section
section_labels = [
    "Experience",  # Experience
    "Skills",   # Skills
    "Education", "Academic Background", "Degree",  # Education
    "Certifications",   # Certifications
    "summary",
]

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to process the CV and segment it into sections, handling nested subsections and synonyms
def segment_cv_into_sections(cv_text):
    """
    Segments the CV into sections based on section titles and their synonyms.

    Args:
        cv_text (str): Text extracted from the CV.

    Returns:
        dict: Dictionary containing sections and their content.
    """
    sections = {title: [] for title in section_synonyms}  # Dictionary to store sections
    current_section = None  # To track the current section
    current_subsection = None  # To track the current subsection (if any)

    # Split the CV text into lines (or paragraphs)
    lines = cv_text.split("\n")

    for line in lines:
        # Check if the line contains a section title or its synonym
        matched = False
        for section, synonyms in section_synonyms.items():
            for synonym in synonyms:
                if re.search(rf"\b{synonym}\b", line, re.IGNORECASE):  # Case insensitive match
                    current_section = section
                    current_subsection = None  # Reset subsection when a new section starts
                    sections[current_section].append({"section_title": line.strip(), "content": []})
                    matched = True
                    break
            if matched:
                break

        # Check for subsections within sections (e.g., job titles under "Experience")
        if current_section and not any(re.search(rf"\b{sec}\b", line, re.IGNORECASE) for sec in section_synonyms):
            if current_subsection:
                # Append content to the current subsection
                sections[current_section][-1]["content"].append(line.strip())
            else:
                # Detect nested subsections like "Bachelor's Degree" or "Software Engineer"
                current_subsection = line.strip()
                sections[current_section][-1]["content"].append(current_subsection)

    return sections

# Function to save sections and subsections into a CSV file, combining all content in one row per section
def save_sections_to_csv(sections, output_csv):
    """
    Saves sections and subsections into a CSV file.

    Args:
        sections (dict): Dictionary containing sections and their content.
        output_csv (str): Path to the output CSV file.
    """
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Section", "Subsection", "Content"])

        # Iterate through sections and aggregate content
        for section, content_list in sections.items():
            # Combine all content for the section into one row
            section_content = []
            for content in content_list:
                section_content.append(content.get("section_title", "N/A"))
                section_content.append(" | ".join(content["content"]))

            # Write the aggregated content for the section
            writer.writerow([section, "All Content", " ".join(section_content)])

# Function to load CSV and extract sections and content
def load_sections_from_csv(csv_file):
    """
    Loads sections and content from a CSV file.

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        dict: Dictionary containing sections and their content.
    """
    df = pd.read_csv(csv_file)
    sections = {}
    for _, row in df.iterrows():
        section = row['Section']
        content = row['Content']
        sections[section] = content
    return sections

# Function to calculate similarity between job description and CV sections
def calculate_similarity(job_desc_sections, cv_sections):
    """
    Calculates similarity between job description and CV sections.

    Args:
        job_desc_sections (dict): Dictionary containing job description sections and their content.
        cv_sections (dict): Dictionary containing CV sections and their content.

    Returns:
        dict: Dictionary containing similarity scores for each section.
    """
    # Sanitize inputs: Ensure all values are strings
    job_desc_sections = {k: str(v) if not isinstance(v, str) else v for k, v in job_desc_sections.items()}
    cv_sections = {k: str(v) if not isinstance(v, str) else v for k, v in cv_sections.items()}

    similarity_scores = {}
    job_desc_embeddings = {section: model.encode(text) for section, text in job_desc_sections.items()}
    cv_embeddings = {section: model.encode(text) for section, text in cv_sections.items()}

    for section in job_desc_sections:
        if section in cv_sections:
            similarity = util.cos_sim(job_desc_embeddings[section], cv_embeddings[section])
            similarity_scores[section] = similarity.item()
        else:
            similarity_scores[section] = None

    return similarity_scores

# Function to calculate weighted score based on section weights
def calculate_weighted_score(similarity_scores, section_weights, job_description_sections):
    """
    Calculates weighted score based on section weights.

    Args:
        similarity_scores (dict): Similarity scores for each section.
        section_weights (dict): Weights for each section.
        job_description_sections (dict): Job description sections and their content.

    Returns:
        float: Weighted score.
    """
    weighted_score = 0
    total_weight = 0

    for section, score in similarity_scores.items():
        # Assign full score if job description section is empty or missing
        score = 1.0 if not job_description_sections.get(section) else score

        if score is not None:
            weight = section_weights.get(section, 1)  # Default weight is 1
            weighted_score += score * weight
            total_weight += weight

    return weighted_score / total_weight if total_weight > 0 else 0

# Function to process all CVs in a directory and aggregate output
def process_all_cvs_in_folder(folder_path, output_folder, aggregated_output):
    """
    Processes all CVs in a directory and aggregates output.

    Args:
        folder_path (str): Path to the folder containing CVs.
        output_folder (str): Path to the output folder.
        aggregated_output (str): Path to the aggregated output CSV file.
    """
    aggregated_data = []  # To store all data in one list

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)

            # Extract text from the PDF
            cv_text = extract_text_from_pdf(pdf_path)

            # Segment the CV into sections and subsections
            segmented_cv = segment_cv_into_sections(cv_text)

            # Save the segmented data into a CSV file for each CV
            output_csv = os.path.join(output_folder, f"{filename.replace('.pdf', '_sections.csv')}")
            save_sections_to_csv(segmented_cv, output_csv)

            print(f"Processed {filename}, saved to {output_csv}")

            # Add to aggregated data
            for section, content_list in segmented_cv.items():
                for content in content_list:
                    aggregated_data.append([
                        filename,
                        section,
                        content.get("section_title", "N/A"),
                        " | ".join(content["content"])
                    ])

    # Save aggregated data to a single CSV file
    with open(aggregated_output, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["File Name", "Section", "Subsection", "Content"])
        writer.writerows(aggregated_data)

    print(f"Aggregated data saved to {aggregated_output}")

# Function to load sections for a specific CV from the aggregated CSV
def load_sections_from_aggregated_csv(aggregated_csv, file_name):
    """
    Loads sections for a specific CV from the aggregated CSV.

    Args:
        aggregated_csv (str): Path to the aggregated CSV file.
        file_name (str): Name of the CV file.

    Returns:
        dict: Dictionary containing sections and their content.
    """
    df = pd.read_csv(aggregated_csv)
    # Filter rows for the specific file name
    cv_data = df[df['File Name'] == file_name]
    # Convert sections and their content to a dictionary
    sections = {row['Section']: row['Content'] for _, row in cv_data.iterrows()}
    return sections

# Function to calculate similarity between job description and CV sections
def calculate_similarity(job_desc_sections, cv_sections):
    """
    Calculates similarity between job description and CV sections.

    Args:
        job_desc_sections (dict): Dictionary containing job description sections and their content.
        cv_sections (dict): Dictionary containing CV sections and their content.

    Returns:
        dict: Dictionary containing similarity scores for each section.
    """
    # Sanitize inputs: Ensure all values are strings
    job_desc_sections = {k: str(v) if not isinstance(v, str) else v for k, v in job_desc_sections.items()}
    cv_sections = {k: str(v) if not isinstance(v, str) else v for k, v in cv_sections.items()}

    similarity_scores = {}
    job_desc_embeddings = {section: model.encode(text) for section, text in job_desc_sections.items()}
    cv_embeddings = {section: model.encode(text) for section, text in cv_sections.items()}

    for section in job_desc_sections:
        if section in cv_sections:
            similarity = util.cos_sim(job_desc_embeddings[section], cv_embeddings[section])
            similarity_scores[section] = similarity.item()
        else:
            similarity_scores[section] = None

    return similarity_scores

# Function to calculate weighted score based on section weights
def calculate_weighted_score(similarity_scores, section_weights, job_description_sections):
    """
    Calculates weighted score based on section weights.

    Args:
        similarity_scores (dict): Similarity scores for each section.
        section_weights (dict): Weights for each section.
        job_description_sections (dict): Job description sections and their content.

    Returns:
        float: Weighted score.
    """
    weighted_score = 0
    total_weight = 0

    for section, score in similarity_scores.items():
        # Assign full score if job description section is empty or missing
        score = 1.0 if not job_description_sections.get(section) else score

        if score is not None:
            weight = section_weights.get(section, 1)  # Default weight is 1
            weighted_score += score * weight
            total_weight += weight

    return weighted_score / total_weight if total_weight > 0 else 0

# Main execution logic
if __name__ == "__main__":
    # Path to your CV PDF
    pdf_path = "/content/drive/MyDrive/DESIGNER/11155153.pdf"

    # Extract text from the PDF
    cv_text = extract_text_from_pdf(pdf_path)

    # Segment the CV into sections
    sections = segment_cv_into_sections(cv_text)

    # Save sections to a CSV file
    output_csv = '/content/drive/MyDrive/cv_sections.csv'
    save_sections_to_csv(sections, output_csv)

    # Load Job Description and CV sections from their respective CSV files
    job_description_sections = load_sections_from_csv('/content/drive/MyDrive/classified_job_description.csv')
    cv_sections = load_sections_from_csv('/content/drive/MyDrive/cv_sections_combined.csv')

    # Calculate similarity and weighted scores
    similarity_scores = calculate_similarity(job_description_sections, cv_sections)
    section_weights = {
        "Education": 0.2,
        "Experience": 0.4,
        "Skills": 0.3,
        "Certifications": 0.1
    }
    weighted_score = calculate_weighted_score(similarity_scores, section_weights, job_description_sections)
    print(f"Weighted Score: {weighted_score}")

    # Process all CVs in the folder
    folder_path = "/content/drive/MyDrive/DESIGNER"  # Replace with your folder path
    output_folder = "/content/drive/MyDrive/Processed_CVs"  # Replace with your output folder
    aggregated_output = "/content/drive/MyDrive/aggregated_cv_sections.csv"  # Replace with the aggregated file path
    process_all_cvs_in_folder(folder_path, output_folder, aggregated_output)

    # Load the aggregated CSV
    aggregated_data = pd.read_csv(aggregated_output)

    # Get the unique file names from the aggregated CSV
    file_names = aggregated_data['File Name'].unique()

    # Prepare output data
    results = []

    # Process each file
    for file_name in file_names:
        # Load sections for the specific CV
        cv_sections = load_sections_from_aggregated_csv(aggregated_output, file_name)

        # Calculate similarity and weighted score
        similarity_scores = calculate_similarity(job_description_sections, cv_sections)
        weighted_score = calculate_weighted_score(similarity_scores, section_weights, job_description_sections)

        # Append results
        results.append({
            "File Name": file_name,
            "Weighted Score": weighted_score,
            "Similarity Scores": similarity_scores  # Optional: can store section-wise scores
        })

        # Print processing details
        print(f"\nProcessed {file_name}")
        for section, score in similarity_scores.items():
            if score is not None:
                print(f"  Section: {section}, Similarity Score: {score:.2f}")
            else:
                print(f"  Section: {section}, No matching section in CV.")

    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.drop(columns=["Similarity Scores"], inplace=True)  # Drop detailed scores for simplicity
    results_df.to_csv('/content/drive/MyDrive/cv_scores.csv', index=False)

    print(f"\nAggregated similarity scores saved to /content/drive/MyDrive/cv_scores.csv")
