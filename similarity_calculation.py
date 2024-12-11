from sentence_transformers import SentenceTransformer, util

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

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
    similarity_scores = {}
    for section, cv_content in cv_sections.items():
        job_content = job_desc_sections.get(section, "")
        if cv_content and job_content:
            cv_embedding = model.encode(cv_content, convert_to_tensor=True)
            job_embedding = model.encode(job_content, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(cv_embedding, job_embedding).item()
            similarity_scores[section] = similarity
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
