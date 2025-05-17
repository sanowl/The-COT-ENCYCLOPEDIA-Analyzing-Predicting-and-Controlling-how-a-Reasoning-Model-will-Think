import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
import numpy as np

def mock_llm_identify_criteria_from_cots(cot_responses):
    print("[Mock LLM] Identifying criteria from CoT responses...")
    criteria_list = [
        {"criterion_name": "Analytical Perspective", "pattern_A_text": "Top-down: Approach starts with the overall structure then breaks it down.", "pattern_B_text": "Bottom-up: Approach starts with details and builds up to a conclusion."},
        {"criterion_name": "Idea Development", "pattern_A_text": "Multiple Paths: Explores several lines of thought or solutions simultaneously.", "pattern_B_text": "Single Path: Focuses on developing one idea or solution path deeply."},
        {"criterion_name": "Verification Focus", "pattern_A_text": "Hypothesis-driven: Verifies against a predefined hypothesis or expectation.", "pattern_B_text": "Data-driven: Verifies based on emerging data or results during the process."},
        {"criterion_name": "Problem Decomposition", "pattern_A_text": "Stepwise Breakdown: Problem is broken into discrete, sequential steps.", "pattern_B_text": "Holistic Assessment: Problem is considered as a whole without explicit sub-steps."},
        {"criterion_name": "Reasoning Type", "pattern_A_text": "Inductive: Generalizes from specific examples or observations.", "pattern_B_text": "Deductive: Applies general rules to specific instances."},
        {"criterion_name": "Clarity of Explanation", "pattern_A_text": "Explicit Steps: Each reasoning step is clearly articulated.", "pattern_B_text": "Implicit Logic: Some reasoning steps are assumed or not fully detailed."},
        {"criterion_name": "Error Handling", "pattern_A_text": "Proactive Check: Potential errors are anticipated and checked.", "pattern_B_text": "Reactive Correction: Errors are corrected after they are found."}
    ]
    num_cots = len(cot_responses) if cot_responses else 1
    all_criteria = []
    for i in range(num_cots):
        for crit in criteria_list:
            all_criteria.append({**crit, "source_cot_id": i})
    print(f"[Mock LLM] Extracted {len(all_criteria)} raw criteria instances.")
    return all_criteria

def mock_llm_generate_rubric_for_criterion(medoid_criterion_dict):
    name = medoid_criterion_dict['criterion_name']
    pattern_a = medoid_criterion_dict['pattern_A_text']
    pattern_b = medoid_criterion_dict['pattern_B_text']
    print(f"[Mock LLM] Generating rubric for medoid criterion: {name}")
    rubric_text = f"**Rubric for {name}**\n\n"
    rubric_text += f"**Pattern A: {pattern_a.split(':')[0].strip()}**\n"
    rubric_text += f"  - Definition: {pattern_a}\n"
    rubric_text += f"  - Characteristics: [LLM-generated Characteristic A1, Characteristic A2]\n"
    rubric_text += f"  - Example: [LLM-generated Example of Pattern A]\n\n"
    rubric_text += f"**Pattern B: {pattern_b.split(':')[0].strip()}**\n"
    rubric_text += f"  - Definition: {pattern_b}\n"
    rubric_text += f"  - Characteristics: [LLM-generated Characteristic B1, Characteristic B2]\n"
    rubric_text += f"  - Example: [LLM-generated Example of Pattern B]\n"
    rubric_text += "Guidance for classification: [LLM-generated guidance on how to distinguish A vs B]"
    return rubric_text

def mock_llm_classify_response_with_rubric(cot_response_text, rubric_text):
    print(f"[Mock LLM] Classifying response against rubric...")
    classified_pattern = np.random.choice(['A', 'B'])
    report = f"**Pattern Analysis Report for Response:**\n"
    report += f"Response snippet: \"{cot_response_text[:100]}...\"\n\n"
    report += f"**Initial Observations:** [LLM-generated summary of reasoning approach in response]\n\n"
    if classified_pattern == 'A':
        report += f"**Evidence for Pattern A:** [LLM quotes segments from response aligning with Pattern A]\n"
        report += f"  - Explanation: [LLM explains how segments match rubric for A]\n"
    else:
        report += f"**Evidence for Pattern B:** [LLM quotes segments from response aligning with Pattern B]\n"
        report += f"  - Explanation: [LLM explains how segments match rubric for B]\n"
    report += f"\n**Pattern Determination:** The response predominantly exhibits **Pattern {classified_pattern}**.\n"
    report += f"  - Justification: [LLM explains why this pattern is dominant]\n"
    report += f"\n**Conclusion:** Final pattern determination: Pattern {classified_pattern}\n"
    return classified_pattern, report

class CoTEncyclopediaFramework:
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2'):
        print("Initializing CoT Encyclopedia Framework...")
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
        except Exception as e:
            print(f"Error loading SentenceTransformer model: {e}")
            print("Ensure 'sentence-transformers' is installed and model name is correct.")
            self.embedding_model = None
        self.raw_criteria_list = []
        self.criteria_embeddings_matrix = None
        self.compressed_criteria_clusters = {}
        self.medoid_criteria = {}
        self.rubrics = {}

    def step1_identify_criteria(self, cot_responses):
        print("\n--- Step 1: Classification Criteria Identification ---")
        if not cot_responses:
            print("No CoT responses provided. Using default sample criteria.")
        self.raw_criteria_list = mock_llm_identify_criteria_from_cots(cot_responses)
        print(f"Identified {len(self.raw_criteria_list)} raw criteria entries.")
        if not self.raw_criteria_list:
            print("No criteria were identified.")
        return self.raw_criteria_list

    def step2_embed_criteria(self):
        print("\n--- Step 2: Classification Criteria Embedding ---")
        if not self.raw_criteria_list:
            print("No criteria to embed. Run Step 1 first.")
            return None
        if not self.embedding_model:
            print("Embedding model not available.")
            return None
        texts_to_embed = [
            f"{c['criterion_name']}. Pattern A: {c['pattern_A_text']} Pattern B: {c['pattern_B_text']}"
            for c in self.raw_criteria_list
        ]
        self.criteria_embeddings_matrix = self.embedding_model.encode(texts_to_embed)
        print(f"Embedded {len(self.raw_criteria_list)} criteria into matrix of shape {self.criteria_embeddings_matrix.shape}")
        return self.criteria_embeddings_matrix

    def _find_medoid_index(self, cluster_points_indices, embeddings_matrix):
        cluster_embeddings = embeddings_matrix[cluster_points_indices]
        dist_matrix = cosine_distances(cluster_embeddings)
        sum_of_distances = np.sum(dist_matrix, axis=1)
        medoid_local_idx = np.argmin(sum_of_distances)
        return cluster_points_indices[medoid_local_idx]

    def step3_compress_criteria_via_clustering(self, n_clusters=3):
        print("\n--- Step 3: Criteria Compression via Hierarchical Clustering ---")
        if self.criteria_embeddings_matrix is None or self.criteria_embeddings_matrix.shape[0] == 0:
            print("No embeddings to cluster. Run Step 2 first.")
            return None
        num_samples = self.criteria_embeddings_matrix.shape[0]
        if num_samples < n_clusters:
            print(f"Warning: Number of samples ({num_samples}) is less than n_clusters ({n_clusters}).")
            print(f"Adjusting n_clusters to {num_samples} for AgglomerativeClustering if num_samples > 0.")
            n_clusters = max(1, num_samples)
        if num_samples == 0:
            print("Cannot cluster 0 samples.")
            return None
        clustering_model = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
        cluster_labels = clustering_model.fit_predict(self.criteria_embeddings_matrix)
        self.compressed_criteria_clusters = {}
        self.medoid_criteria = {}
        print(f"Clustered criteria into {n_clusters} groups.")
        for i in range(n_clusters):
            cluster_indices = [idx for idx, label in enumerate(cluster_labels) if label == i]
            if not cluster_indices:
                continue
            self.compressed_criteria_clusters[i] = [self.raw_criteria_list[idx] for idx in cluster_indices]
            medoid_original_idx = self._find_medoid_index(cluster_indices, self.criteria_embeddings_matrix)
            self.medoid_criteria[i] = self.raw_criteria_list[medoid_original_idx]
            print(f"  Cluster {i}: {len(cluster_indices)} criteria. Medoid: '{self.medoid_criteria[i]['criterion_name']}'")
        return self.medoid_criteria

    def step4_generate_rubrics(self):
        print("\n--- Step 4: Rubric Generation ---")
        if not self.medoid_criteria:
            print("No medoid criteria found. Run Step 3 first.")
            return None
        self.rubrics = {}
        for cluster_id, medoid_crit_dict in self.medoid_criteria.items():
            rubric_text = mock_llm_generate_rubric_for_criterion(medoid_crit_dict)
            self.rubrics[cluster_id] = rubric_text
        print(f"Generated {len(self.rubrics)} rubrics.")
        return self.rubrics

    def step5_analyze_response_with_rubrics(self, cot_response_to_analyze, cluster_id_for_rubric):
        print("\n--- Step 5: Pattern Analysis Report Generation ---")
        if cluster_id_for_rubric not in self.rubrics:
            print(f"No rubric found for cluster_id {cluster_id_for_rubric}. Run Step 4 first.")
            return None, None
        rubric = self.rubrics[cluster_id_for_rubric]
        medoid_name = self.medoid_criteria[cluster_id_for_rubric]['criterion_name']
        print(f"Analyzing response using rubric for '{medoid_name}' (Cluster {cluster_id_for_rubric}).")
        classified_pattern, report_text = mock_llm_classify_response_with_rubric(cot_response_to_analyze, rubric)
        print(f"Response classified as Pattern {classified_pattern} for '{medoid_name}'.")
        return classified_pattern, report_text

def main():
    sample_cot_responses = [
        "To solve this math problem, I first broke it into three parts. Then I addressed each part sequentially, checking my work at each stage. Finally, I combined the results.",
        "Considering the user's question about historical accuracy, I compared several sources. One source suggested X, another Y. I synthesized these by looking for common themes and noting discrepancies.",
        "The code wasn't running. I hypothesized it was a type error. I checked variable types. That wasn't it. Then I thought maybe an API limit. I checked the logs. That was it."
    ]
    framework = CoTEncyclopediaFramework()
    if not framework.embedding_model:
        print("Exiting due to embedding model loading failure.")
        return

    framework.step1_identify_criteria(sample_cot_responses)
    framework.step2_embed_criteria()
    num_final_clusters = 3
    medoids = framework.step3_compress_criteria_via_clustering(n_clusters=num_final_clusters)
    if not medoids:
        print("Clustering failed or produced no medoids. Exiting.")
        return
    all_rubrics = framework.step4_generate_rubrics()
    if not all_rubrics:
        print("Rubric generation failed. Exiting.")
        return
    
    print("\n--- Conceptual Use Cases ---")
    new_cot_response = "The model directly answered the question without showing intermediate steps. It seemed to rely on its internal knowledge base rather than explicit calculation."
    print(f"\nAnalyzing a new CoT response: \"{new_cot_response[:50]}...\"")
    if framework.rubrics:
        first_cluster_id = list(framework.rubrics.keys())[0]
        classified_as, report = framework.step5_analyze_response_with_rubrics(
            new_cot_response,
            first_cluster_id
        )
        if report:
            print(f"\nFull Analysis Report for Cluster {first_cluster_id} ('{framework.medoid_criteria[first_cluster_id]['criterion_name']}'):")
            print(report)
    else:
        print("No rubrics available to analyze the new CoT response.")

    print("\nConceptual: Optimal Reasoning Pattern Control")
    print("  - This would involve predicting optimal strategies and prompting an LLM.")
    print("  - E.g., If 'Top-down' (Pattern A of 'Analytical Perspective') is optimal for a question:")
    print("    LLM_Prompt: \"Solve [Question] using a Top-down approach: First, conceptualize the overall structure...\"")
    print("\n--- Framework Execution Complete ---")

if __name__ == '__main__':
    main()
