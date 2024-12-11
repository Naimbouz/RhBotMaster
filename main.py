import os
from text_extraction import extract_text_from_pdf
from similarity_calculation import calculate_similarity, calculate_weighted_score
from data_processing import load_sections_from_csv, save_sections_to_csv, load_sections_from_aggregated_csv, process_all_cvs_in_folder

# Main execution logic
if __name__ == "__main__":
    # Example paths
    folder_path = "path/to/cv/folder"
    output_folder = "path/to/output/folder"
    aggregated_output = "path/to/aggregated_output.csv"

    # Process all CVs in the folder
    process_all_cvs_in_folder(folder_path, output_folder, aggregated_output)

    # Example of loading sections from a CSV
    csv_file = "path/to/csv/file.csv"
    sections = load_sections_from_csv(csv_file)

    # Example of calculating similarity
    job_desc_sections = {"Experience": "...", "Skills": "..."}  # Example job description sections
    similarity_scores = calculate_similarity(job_desc_sections, sections)

    # Example of calculating weighted score
    section_weights = {"Experience": 2, "Skills": 1.5}  # Example weights
    weighted_score = calculate_weighted_score(similarity_scores, section_weights, job_desc_sections)
    print(f"Weighted Score: {weighted_score}")
