import pandas as pd

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
    sections = {row['Section']: row['Content'] for _, row in df.iterrows()}
    return sections

# Function to save sections and subsections into a CSV file, combining all content in one row per section
def save_sections_to_csv(sections, output_csv):
    """
    Saves sections and subsections into a CSV file.

    Args:
        sections (dict): Dictionary containing sections and their content.
        output_csv (str): Path to the output CSV file.
    """
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Section', 'Content'])
        for section, content in sections.items():
            writer.writerow([section, content])

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
    cv_data = df[df['File Name'] == file_name]
    sections = {row['Section']: row['Content'] for _, row in cv_data.iterrows()}
    return sections

# Function to process all CVs in a directory and aggregate output
def process_all_cvs_in_folder(folder_path, output_folder, aggregated_output):
    """
    Processes all CVs in a directory and aggregates output.

    Args:
        folder_path (str): Path to the folder containing CVs.
        output_folder (str): Path to the output folder.
        aggregated_output (str): Path to the aggregated output CSV file.
    """
    all_sections = {}
    for cv_file in os.listdir(folder_path):
        if cv_file.endswith('.pdf'):
            cv_path = os.path.join(folder_path, cv_file)
            cv_text = extract_text_from_pdf(cv_path)
            sections = segment_cv_into_sections(cv_text)
            save_sections_to_csv(sections, os.path.join(output_folder, cv_file.replace('.pdf', '.csv')))
            all_sections[cv_file] = sections
    # Combine all sections into a single CSV file
    with open(aggregated_output, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['File Name', 'Section', 'Content'])
        for file_name, sections in all_sections.items():
            for section, content in sections.items():
                writer.writerow([file_name, section, content])
