import pandas as pd
import numpy as np
from scipy.stats import kendalltau, wilcoxon


import numpy as np
import pandas as pd
from scipy.stats import kendalltau, wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def compute_rank_differences(overall, specific, k):
    """
    Compute the signed and absolute rank differences for models in the top-k of the overall ranking.
    The signed difference is defined as: overall position - specific position.
    """
    # Select top-k models from the overall ranking.
    overall_top_k = overall[:k]
    
    # Identify common models that appear in both rankings.
    common_models = [model for model in overall_top_k if model in specific]
    
    # Get the positions of these models in the overall and specific rankings.
    overall_positions = [overall.index(model) for model in common_models]
    specific_positions = [specific.index(model) for model in common_models]
    
    # Compute signed differences and then the absolute differences.
    differences = np.array([o - s for o, s in zip(overall_positions, specific_positions)])
    absolute_differences = np.abs(differences)
    return common_models, differences, absolute_differences

def compute_kendall_for_topk(overall, specific, k):
    """
    Compute Kendall tau correlation for the top-k models in the overall ranking.
    """
    # Get models that are in both rankings
    common_models = [model for model in overall if model in specific]
    
    # If we want to limit to top-k, take only top-k from common models based on overall ranking
    if k < len(common_models):
        # Sort common models by their position in the overall ranking
        common_models_sorted = sorted(common_models, key=lambda m: overall.index(m))
        common_models = common_models_sorted[:k]
    
    # Get positions of these models in both rankings
    overall_positions = [overall.index(model) for model in common_models]
    specific_positions = [specific.index(model) for model in common_models]
    
    # Compute Kendall tau
    if len(common_models) > 1:
        tau, p_value = kendalltau(overall_positions, specific_positions)
        return tau, p_value, len(common_models)
    else:
        return None, None, len(common_models)

def analyze_all_concepts(overall_ranking, concepts_rankings, concept_names, top_k_values=[1, 5, 10]):
    """
    Analyze rank differences and correlations between overall ranking and each concept-specific ranking.
    
    Parameters:
    - overall_ranking: List of model names in order of overall performance
    - concepts_rankings: List of lists, each containing models ranked for a specific concept
    - concept_names: List of names corresponding to each concept
    - top_k_values: List of k values to analyze
    
    Returns:
    - Dictionary with results
    """
    results = {
        'rank_differences': {},
        'kendall_tau': {},
        'summary': {}
    }
    
    all_signed_diffs = defaultdict(list)
    all_abs_diffs = defaultdict(list)
    all_taus = defaultdict(list)
    
    # For each concept
    for concept_idx, (concept_name, concept_ranking) in enumerate(zip(concept_names, concepts_rankings)):
        concept_results = {
            'rank_differences': {},
            'kendall_tau': {}
        }
        
        # For each k value
        for k in top_k_values:
            # Compute rank differences
            common_models, signed_diffs, abs_diffs = compute_rank_differences(overall_ranking, concept_ranking, k)
            
            concept_results['rank_differences'][k] = {
                'common_models': common_models,
                'signed_differences': signed_diffs.tolist(),
                'absolute_differences': abs_diffs.tolist(),
                'mean_signed_diff': float(np.mean(signed_diffs)) if len(signed_diffs) > 0 else None,
                'mean_abs_diff': float(np.mean(abs_diffs)) if len(abs_diffs) > 0 else None,
                'median_signed_diff': float(np.median(signed_diffs)) if len(signed_diffs) > 0 else None,
                'median_abs_diff': float(np.median(abs_diffs)) if len(abs_diffs) > 0 else None
            }
            
            # Perform Wilcoxon test if there are enough data points
            if len(signed_diffs) > 1:
                try:
                    stat, p = wilcoxon(signed_diffs)
                    concept_results['rank_differences'][k]['wilcoxon_stat'] = float(stat)
                    concept_results['rank_differences'][k]['wilcoxon_p'] = float(p)
                except Exception as e:
                    concept_results['rank_differences'][k]['wilcoxon_error'] = str(e)
            
            # Extend the aggregated lists
            all_signed_diffs[k].extend(signed_diffs)
            all_abs_diffs[k].extend(abs_diffs)
            
            # Compute Kendall tau
            tau, p_value, num_common = compute_kendall_for_topk(overall_ranking, concept_ranking, k)
            
            if tau is not None:
                concept_results['kendall_tau'][k] = {
                    'tau': float(tau),
                    'p_value': float(p_value),
                    'num_common_models': num_common
                }
                all_taus[k].append(tau)
        
        results['rank_differences'][concept_name] = concept_results['rank_differences']
        results['kendall_tau'][concept_name] = concept_results['kendall_tau']
    
    # Compute aggregate statistics
    for k in top_k_values:
        results['summary'][k] = {
            'mean_signed_diff': float(np.mean(all_signed_diffs[k])) if all_signed_diffs[k] else None,
            'mean_abs_diff': float(np.mean(all_abs_diffs[k])) if all_abs_diffs[k] else None,
            'median_signed_diff': float(np.median(all_signed_diffs[k])) if all_signed_diffs[k] else None, 
            'median_abs_diff': float(np.median(all_abs_diffs[k])) if all_abs_diffs[k] else None,
            'mean_kendall_tau': float(np.mean(all_taus[k])) if all_taus[k] else None,
            'median_kendall_tau': float(np.median(all_taus[k])) if all_taus[k] else None
        }
        
        # Perform aggregate Wilcoxon test
        if len(all_signed_diffs[k]) > 1:
            try:
                stat, p = wilcoxon(all_signed_diffs[k])
                results['summary'][k]['wilcoxon_stat'] = float(stat)
                results['summary'][k]['wilcoxon_p'] = float(p)
            except Exception as e:
                results['summary'][k]['wilcoxon_error'] = str(e)
    
    return results

def visualize_results(results, concept_names, top_k_values=[1, 5, 10]):
    """
    Create visualizations for the analysis results.
    
    Parameters:
    - results: Dictionary with analysis results
    - concept_names: List of concept names
    - top_k_values: List of k values analyzed
    
    Returns:
    - Figure object
    """
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Colors for different k values
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # 1. Plot mean absolute rank differences by concept
    ax = axes[0, 0]
    width = 0.8 / len(top_k_values)
    x = np.arange(len(concept_names))
    
    for i, k in enumerate(top_k_values):
        means = []
        for concept in concept_names:
            if k in results['rank_differences'][concept] and results['rank_differences'][concept][k]['mean_abs_diff'] is not None:
                means.append(results['rank_differences'][concept][k]['mean_abs_diff'])
            else:
                means.append(0)
        
        ax.bar(x + (i - len(top_k_values)/2 + 0.5) * width, means, width, label=f'Top-{k}', color=colors[i])
    
    ax.set_xlabel('Concept')
    ax.set_ylabel('Mean Absolute Rank Difference')
    ax.set_title('Mean Absolute Rank Differences by Concept')
    ax.set_xticks(x)
    ax.set_xticklabels(concept_names, rotation=45, ha='right')
    ax.legend()
    
    # 2. Plot Kendall tau correlations by concept
    ax = axes[0, 1]
    
    for i, k in enumerate(top_k_values):
        taus = []
        for concept in concept_names:
            if (k in results['kendall_tau'][concept] and 
                'tau' in results['kendall_tau'][concept][k]):
                taus.append(results['kendall_tau'][concept][k]['tau'])
            else:
                taus.append(None)
        
        # Filter out None values
        valid_indices = [i for i, val in enumerate(taus) if val is not None]
        valid_taus = [taus[i] for i in valid_indices]
        valid_concepts = [concept_names[i] for i in valid_indices]
        
        if valid_taus:
            x_positions = np.arange(len(valid_concepts)) + (i - len(top_k_values)/2 + 0.5) * width
            ax.bar(x_positions, valid_taus, width, label=f'Top-{k}', color=colors[i])
    
    ax.set_xlabel('Concept')
    ax.set_ylabel('Kendall Tau Correlation')
    ax.set_title('Kendall Tau Correlations by Concept')
    ax.set_xticks(np.arange(len(concept_names)))
    ax.set_xticklabels(concept_names, rotation=45, ha='right')
    ax.axhline(y=0, color='r', linestyle='-')
    ax.legend()
    
    # 3. Plot summary statistics
    ax = axes[1, 0]
    
    summary_metrics = ['mean_abs_diff', 'median_abs_diff']
    x = np.arange(len(summary_metrics))
    width = 0.8 / len(top_k_values)
    
    for i, k in enumerate(top_k_values):
        values = [results['summary'][k].get(metric) for metric in summary_metrics]
        ax.bar(x + (i - len(top_k_values)/2 + 0.5) * width, values, width, label=f'Top-{k}', color=colors[i])
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')
    ax.set_title('Rank Difference Summary Statistics')
    ax.set_xticks(x)
    ax.set_xticklabels(['Mean Absolute Diff', 'Median Absolute Diff'])
    ax.legend()
    
    # 4. Plot Kendall tau summary
    ax = axes[1, 1]
    
    summary_metrics = ['mean_kendall_tau', 'median_kendall_tau']
    x = np.arange(len(summary_metrics))
    
    for i, k in enumerate(top_k_values):
        values = [results['summary'][k].get(metric) for metric in summary_metrics]
        ax.bar(x + (i - len(top_k_values)/2 + 0.5) * width, values, width, label=f'Top-{k}', color=colors[i])
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')
    ax.set_title('Kendall Tau Summary Statistics')
    ax.set_xticks(x)
    ax.set_xticklabels(['Mean Kendall Tau', 'Median Kendall Tau'])
    ax.axhline(y=0, color='r', linestyle='-')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('concept_ranking_analysis.png', dpi=300)
    
    return fig

def run_full_analysis(overall_ranking, concepts_rankings, concept_names):
    """
    Run the complete analysis and generate visualizations.
    
    Parameters:
    - overall_ranking: List of model names in general ranking
    - concepts_rankings: List of lists with concept-specific rankings
    - concept_names: List of concept names
    
    Returns:
    - Dictionary with analysis results and figures
    """
    # Define top-k values to analyze
    top_k_values = [ 5, 10]
    print(overall_ranking)
    
    # Run analysis
    results = analyze_all_concepts(overall_ranking, concepts_rankings, concept_names, top_k_values)
    
    # Generate visualizations
    # fig = visualize_results(results, concept_names, top_k_values)
    
    # Print summary statistics
    print("\n===== SUMMARY STATISTICS =====")
    for k in top_k_values:
        print(f"\nTop-{k} Results:")
        summary = results['summary'][k]
        
        print(f"Mean absolute rank difference: {summary.get('mean_abs_diff', 'N/A'):.3f}")
        print(f"Median absolute rank difference: {summary.get('median_abs_diff', 'N/A'):.3f}")
        print(f"Mean Kendall tau correlation: {summary.get('mean_kendall_tau', 'N/A'):.3f}")
        print(f"Median Kendall tau correlation: {summary.get('median_kendall_tau', 'N/A'):.3f}")
        
        if 'wilcoxon_p' in summary:
            p_value = summary['wilcoxon_p']
            stat = summary['wilcoxon_stat']
            print(f"Aggregate Wilcoxon test: statistic={stat:.3f}, p-value={p_value:.3g}")
            if p_value < 0.05:
                print("  * Significant difference between overall and specific rankings")
    
    # Print concept-specific results
    print("\n===== CONCEPT-SPECIFIC RESULTS =====")
    for concept in concept_names:
        print(f"\n{concept}:")
        
        for k in top_k_values:
            if k in results['rank_differences'][concept]:
                diff_result = results['rank_differences'][concept][k]
                print(f"  Top-{k} mean absolute difference: {diff_result.get('mean_abs_diff', 'N/A')}")
                
                if 'wilcoxon_p' in diff_result:
                    p_value = diff_result['wilcoxon_p']
                    significant = "Significant" if p_value < 0.05 else "Not significant"
                    print(f"  Top-{k} Wilcoxon test: {significant} (p={p_value:.3g})")
            
            if k in results['kendall_tau'][concept] and 'tau' in results['kendall_tau'][concept][k]:
                tau = results['kendall_tau'][concept][k]['tau']
                p_value = results['kendall_tau'][concept][k]['p_value']
                print(f"  Top-{k} Kendall tau: {tau:.3f} (p={p_value:.3g})")
    
    return {
        'results': results,
    }

# Example usage with the provided structure
def main():
    # Sample data - replace with your actual rankings
    overall_ranking = ["ModelA", "ModelB", "ModelC", "ModelD", "ModelE", "ModelF", "ModelG", "ModelH", "ModelI", "ModelJ"]
    
    # Define concept-specific rankings
    concept1 = ["ModelB", "ModelA", "ModelD", "ModelC", "ModelF", "ModelE", "ModelH", "ModelG", "ModelI", "ModelJ"]
    concept2 = ["ModelC", "ModelA", "ModelB", "ModelF", "ModelD", "ModelG", "ModelE", "ModelI", "ModelH", "ModelJ"]
    concept3 = ["ModelA", "ModelD", "ModelC", "ModelB", "ModelE", "ModelG", "ModelF", "ModelH", "ModelJ", "ModelI"]
    concept4 = ["ModelE", "ModelB", "ModelC", "ModelA", "ModelF", "ModelD", "ModelI", "ModelG", "ModelH", "ModelJ"]
    concept5 = ["ModelD", "ModelF", "ModelA", "ModelC", "ModelB", "ModelE", "ModelG", "ModelI", "ModelJ", "ModelH"]
    concept6 = ["ModelB", "ModelE", "ModelA", "ModelD", "ModelC", "ModelF", "ModelH", "ModelI", "ModelG", "ModelJ"]
    
    # Gather all concepts into a list
    concepts = [concept1, concept2, concept3, concept4, concept5, concept6]
    concept_names = ["Concept1", "Concept2", "Concept3", "Concept4", "Concept5", "Concept6"]
    
    # Run analysis
    analysis_results = run_full_analysis(overall_ranking, concepts, concept_names)
    
    return analysis_results

# For individual concept analysis (as in your original snippet)
def analyze_single_concept(overall_ranking, concept_ranking, top_k_values=[1, 5, 10]):
    """
    Analyze a single concept against the overall ranking.
    """
    from scipy.stats import wilcoxon
    
    for k in top_k_values:
        common_models, differences, abs_diffs = compute_rank_differences(overall_ranking, concept_ranking, k)
        
        print(f"\nTop-{k} models: {common_models}")
        print(f"Signed rank differences (overall - specific): {differences}")
        print(f"Absolute differences: {abs_diffs}")
        
        # Ensure that there is more than one sample point for the test.
        if len(differences) > 1:
            try:
                stat, p = wilcoxon(differences)
                print(f"Wilcoxon signed-rank test statistic: {stat:.3f}, p-value: {p:.3g}")
            except Exception as e:
                print(f"Wilcoxon test could not be performed: {e}")
        else:
            print("Not enough data points to perform the Wilcoxon signed-rank test.")

    # Compute Kendall tau for the same top-k subsets
    for k in top_k_values:
        tau, p_value, num_common = compute_kendall_for_topk(overall_ranking, concept_ranking, k)
        if tau is not None:
            print(f"\nTop-{k} Kendall tau: {tau:.3f}, p-value: {p_value:.3g}")
        else:
            print(f"\nTop-{k} Kendall tau: Not enough common models")

if __name__ == "__main__":
    # Sample data for demonstration
    
    # Raw data as a dictionary (model: score)
    full_rank = {
        'openai_gpt_4o_2024_05_13': 0.796606,
        'mistralai_bakllava_v1_hf': 0.71620,
        'google_gemini_1.5_pro_preview_0409': 0.6888686,
        'writer_palmyra_vision_003': 0.65315075,
        'InternVL-Chat-V1-5': 0.6508005262250713,
        'microsoft_llava_1.5_13b_hf': 0.6322801,
        'google_gemini_1.0_pro_vision_001': 0.6291944,
        'llava-next-72b': 0.6057774169497487,
        'openai_gpt_4_1106_vision_preview': 0.59245004,
        'internlm-xcomposer2-4khd-7b': 0.5997462509122599,
        'llava-v1.6-34b': 0.5956283725489049,
        'microsoft_llava_1.5_7b_hf': 0.574705,
        'huggingfacem4_idefics2_8b': 0.5249094,
        'llava-v1.6-vicuna-13b': 0.537007405723124,
        'idefics2-8b': 0.44029060214316085,
        'llava_1.6_13b': 0.41535495956838203,
        'llava_1.6_vicuna_7b': 0.4087238898810243,
        'anthropic_claude_3_haiku_20240307': 0.175932,
        'anthropic_claude_3_opus_20240229': 0.165100,
        'anthropic_claude_3_sonnet_20240229': 0.155222,
        'llava_1.6_vicuna_13b': 0.128068,
        'llava_1.6_vicuna_7b': 0.102671,
        'llava_1.6_mistral_7b': 0.039684,
        'google_gemini_pro_vision': 0.030795,
        'google_paligemma_3b_mix_448': 0.024409,
        'llava_13b': 0.036919636473105516,
        'openai_gpt_4_vision_preview': -0.061939,
        'huggingfacem4_idefics_9b_instruct': -0.368634,
        'huggingfacem4_idefics_80b_instruct': -0.370931,
        'instructblip-vicuna-13b': -0.36329906660686884,
        'qwen_qwen_vl': -0.603366,
        'llava_7b': -0.5613344012693971,
        'openflamingo_openflamingo_9b_vitl_mpt7b': -0.636209,
        'qwen_vl_chat': -0.6493873776041786,
        'huggingfacem4_idefics_80b': -0.738342,
        'huggingfacem4_idefics_9b': -0.759691,
        'instructblip-vicuna-7b': -0.74785261683665736,
        'qwen_qwen_vl_chat': -0.764303
    }
    #make it a list of the keys
    overall_ranking = list(full_rank.keys())
    print(overall_ranking)
    concept1= [
        "writer_palmyra_vision_003",
        "google_gemini_1.0_pro_vision_001",
        "google_gemini_1.5_pro_preview_0409",
        "internlm-xcomposer2-4khd-7b",
        "openai_gpt_4o_2024_05_13",
        "openai_gpt_4_1106_vision_preview",
        "anthropic_claude_3_haiku_20240307",
        "anthropic_claude_3_sonnet_20240229",
        "anthropic_claude_3_opus_20240229",
        "idefics2-8b",
        "huggingfacem4_idefics_80b_instruct",
        "llava-next-72b",
        "llava-v1.6-vicuna-13b",
        "openai_gpt_4_vision_preview",
        "google_gemini_pro_vision",
        "InternVL-Chat-V1-5",
        "llava-v1.6-34b",
        "microsoft_llava_1.5_13b_hf",
        "mistralai_bakllava_v1_hf",
        "microsoft_llava_1.5_7b_hf",
        "llava_1.6_mistral_7b",
        "openflamingo_openflamingo_9b_vitl_mpt7b",
        "llava_1.6_13b",
        "instructblip-vicuna-13b",
        "llava_13b",
        "llava_1.6_vicuna_7b",
        "instructblip-vicuna-7b",
        "llava_7b",
        "google_paligemma_3b_mix_448",
        "huggingfacem4_idefics_80b",
        "huggingfacem4_idefics_9b_instruct",
        "huggingfacem4_idefics_9b",
        "qwen_vl_chat",
        "qwen_qwen_vl"
    ]

    concept2 = [
        'mistralai_bakllava_v1_hf',
        "huggingfacem4_idefics_80b_instruct",
        "huggingfacem4_idefics_9b_instruct",
        "internlm-xcomposer2-4khd-7b",
        "llava_1.6_vicuna_7b",
        "InternVL-Chat-V1-5",
        "llava_1.6_mistral_7b",
        "instructblip-vicuna-7b",
        "qwen_qwen_vl",
        "anthropic_claude_3_opus_20240229",
        "anthropic_claude_3_sonnet_20240229",
        "llava_13b",
        "llava_7b",
        "microsoft_llava_1.5_7b_hf",
        "llava-next-72b",
        "anthropic_claude_3_haiku_20240307",
        "instructblip-vicuna-13b",
        "openai_gpt_4_vision_preview",
        "openai_gpt_4o_2024_05_13",
        "llava_1.6_13b",
        "microsoft_llava_1.5_13b_hf",
        "google_gemini_1.0_pro_vision_001",
        "llava-v1.6-vicuna-13b",
        "huggingfacem4_idefics_80b",
        "writer_palmyra_vision_003",
        "llava-v1.6-34b",
        "openflamingo_openflamingo_9b_vitl_mpt7b",
        "google_gemini_pro_vision",
        "google_paligemma_3b_mix_448",
        "huggingfacem4_idefics_9b",
        "openai_gpt_4_1106_vision_preview",
        "idefics2-8b",
        "google_gemini_1.5_pro_preview_0409",
        "qwen_vl_chat"
    ]


    concept3 =  [
        "openai_gpt_4o_2024_05_13",
        'mistralai_bakllava_v1_hf',
        "google_paligemma_3b_mix_448",
        "writer_palmyra_vision_003",
        "InternVL-Chat-V1-5",
        "llava_13b",
        "internlm-xcomposer2-4khd-7b",
        "huggingfacem4_idefics_80b_instruct",
        "google_gemini_1.0_pro_vision_001",
        "llava_1.6_vicuna_7b",
        "google_gemini_1.5_pro_preview_0409",
        "anthropic_claude_3_opus_20240229",
        "openai_gpt_4_1106_vision_preview",
        "llava-next-72b",
        "llava_7b",
        "llava-v1.6-34b",
        "huggingfacem4_idefics_9b_instruct",
        "anthropic_claude_3_haiku_20240307",
        "llava-v1.6-vicuna-13b",
        "microsoft_llava_1.5_7b_hf",
        "microsoft_llava_1.5_13b_hf",
        "llava_1.6_13b",
        "anthropic_claude_3_sonnet_20240229",
        "openai_gpt_4_vision_preview",
        "google_gemini_pro_vision",
        "openflamingo_openflamingo_9b_vitl_mpt7b",
        "idefics2-8b",
        "llava_1.6_mistral_7b",
        "instructblip-vicuna-13b",
        "instructblip-vicuna-7b",
        "qwen_qwen_vl",
        "huggingfacem4_idefics_80b",
        "huggingfacem4_idefics_9b",
        "qwen_vl_chat"
    ]

    concept4 = [
        "google_gemini_1.0_pro_vision_001",
        "openai_gpt_4o_2024_05_13",
        "openai_gpt_4_1106_vision_preview",
        "writer_palmyra_vision_003",
        "google_gemini_1.5_pro_preview_0409",
        "anthropic_claude_3_opus_20240229",
        "anthropic_claude_3_haiku_20240307",
        "InternVL-Chat-V1-5",
        "internlm-xcomposer2-4khd-7b",
        "llava_1.6_mistral_7b",
        "llava-v1.6-34b",
        "llava-v1.6-vicuna-13b",
        "llava-next-72b",
        "llava_1.6_vicuna_7b",
        "llava_1.6_13b",
        "huggingfacem4_idefics_80b_instruct",
        "llava_13b",
        "idefics2-8b",
        "mistralai_bakllava_v1_hf",
        "microsoft_llava_1.5_13b_hf",
        "microsoft_llava_1.5_7b_hf",
        "llava_7b",
        "openflamingo_openflamingo_9b_vitl_mpt7b",
        "google_gemini_pro_vision",
        "anthropic_claude_3_sonnet_20240229",
        "instructblip-vicuna-7b",
        "instructblip-vicuna-13b",
        "openai_gpt_4_vision_preview",
        "huggingfacem4_idefics_9b_instruct",
        "google_paligemma_3b_mix_448",
        "qwen_qwen_vl",
        "huggingfacem4_idefics_80b",
        "huggingfacem4_idefics_9b",
        "qwen_vl_chat"
    ]

    concept5 = [
        "huggingfacem4_idefics_9b",
        "google_gemini_pro_vision",
        "InternVL-Chat-V1-5",
        "google_gemini_1.0_pro_vision_001",
        "openai_gpt_4o_2024_05_13",
        "anthropic_claude_3_opus_20240229",
        "internlm-xcomposer2-4khd-7b",
        "huggingfacem4_idefics_80b_instruct",
        "huggingfacem4_idefics_80b",
        "llava-next-72b",
        "llava-v1.6-34b",
        "llava-v1.6-vicuna-13b",
        "llava_1.6_vicuna_7b",
        "openai_gpt_4_vision_preview",
        "anthropic_claude_3_haiku_20240307",
        "google_gemini_1.5_pro_preview_0409",
        "llava_1.6_13b",
        "huggingfacem4_idefics_9b_instruct",
        "microsoft_llava_1.5_7b_hf",
        "mistralai_bakllava_v1_hf",
        "microsoft_llava_1.5_13b_hf",
        "qwen_vl_chat",
        "llava_13b",
        "llava_1.6_mistral_7b",
        "llava_7b",
        "writer_palmyra_vision_003",
        "openai_gpt_4_1106_vision_preview",
        "openflamingo_openflamingo_9b_vitl_mpt7b",
        "instructblip-vicuna-13b",
        "instructblip-vicuna-7b",
        "anthropic_claude_3_sonnet_20240229",
        "idefics2-8b",
        "google_paligemma_3b_mix_448",
        "qwen_qwen_vl"
    ]

    concept6 = [
        'openai_gpt_4o_2024_05_13', 
        "openai_gpt_4_1106_vision_preview",
        'mistralai_bakllava_v1_hf',
        'google_gemini_1.5_pro_preview_0409', 
        'writer_palmyra_vision_003', 
        "openai_gpt_4_vision_preview",
        'InternVL-Chat-V1-5', 
        'microsoft_llava_1.5_13b_hf', 
        "anthropic_claude_3_opus_20240229",
        "internlm-xcomposer2-4khd-7b",
        "llava-v1.6-34b",
        "llava-v1.6-vicuna-13b",
        "anthropic_claude_3_haiku_20240307",
        "llava_1.6_vicuna_7b",
        'llava-next-72b',
        "llava_1.6_13b",
        "idefics2-8b",
        'google_gemini_1.0_pro_vision_001', 
        "google_paligemma_3b_mix_448",
        "llava_13b",
        "microsoft_llava_1.5_7b_hf",
        "google_gemini_pro_vision",
        "anthropic_claude_3_sonnet_20240229",
        "openflamingo_openflamingo_9b_vitl_mpt7b",
        "llava_7b",
        "llava_1.6_mistral_7b",
        "huggingfacem4_idefics_9b_instruct",
        "huggingfacem4_idefics_80b",
        "instructblip-vicuna-13b",
        "huggingfacem4_idefics_80b_instruct",
        "instructblip-vicuna-7b",
        "qwen_qwen_vl",
        "huggingfacem4_idefics_9b",
        "qwen_vl_chat"
    ]

    concept7 = [
        "openai_gpt_4o_2024_05_13",
        "writer_palmyra_vision_003",
        "internlm-xcomposer2-4khd-7b",
        "llava-next-72b",
        "llava_13b",
        "google_gemini_1.5_pro_preview_0409",
        "llava-v1.6-34b",
        "google_paligemma_3b_mix_448",
        "huggingfacem4_idefics_80b",
        "anthropic_claude_3_opus_20240229",
        "llava_1.6_13b",
        "llava_7b",
        "mistralai_bakllava_v1_hf",
        "microsoft_llava_1.5_7b_hf",
        "microsoft_llava_1.5_13b_hf",
        "google_gemini_pro_vision",
        "openflamingo_openflamingo_9b_vitl_mpt7b",
        "anthropic_claude_3_sonnet_20240229",
        "openai_gpt_4_vision_preview",
        "google_gemini_1.0_pro_vision_001",
        "openai_gpt_4_1106_vision_preview",
        "qwen_qwen_vl",
        "llava_1.6_mistral_7b",
        "llava-v1.6-vicuna-13b",
        "idefics2-8b",
        "InternVL-Chat-V1-5",
        "instructblip-vicuna-13b",
        "anthropic_claude_3_haiku_20240307",
        "huggingfacem4_idefics_9b",
        "huggingfacem4_idefics_80b_instruct",
        "huggingfacem4_idefics_9b_instruct",
        "qwen_vl_chat",
        "llava_1.6_vicuna_7b",
        "instructblip-vicuna-7b"
    ]
    concept8 = [
        "idefics2-8b",
        "openai_gpt_4o_2024_05_13",
        "qwen_qwen_vl",
        "openai_gpt_4_1106_vision_preview",
        "google_gemini_1.5_pro_preview_0409",
        "writer_palmyra_vision_003",
        "anthropic_claude_3_haiku_20240307",
        "llava_7b",
        "llava_1.6_vicuna_7b",
        "internlm-xcomposer2-4khd-7b",
        "instructblip-vicuna-13b",
        "llava-next-72b",
        "InternVL-Chat-V1-5",
        "mistralai_bakllava_v1_hf",
        "microsoft_llava_1.5_13b_hf",
        "openflamingo_openflamingo_9b_vitl_mpt7b",
        "microsoft_llava_1.5_7b_hf",
        "google_gemini_pro_vision",
        "openai_gpt_4_vision_preview",
        "llava-v1.6-34b",
        "llava_13b",
        "instructblip-vicuna-7b",
        "llava-v1.6-vicuna-13b",
        "llava_1.6_13b",
        "anthropic_claude_3_sonnet_20240229",
        "google_gemini_1.0_pro_vision_001",
        "google_paligemma_3b_mix_448",
        "huggingfacem4_idefics_80b",
        "huggingfacem4_idefics_80b_instruct",
        "huggingfacem4_idefics_9b_instruct",
        "huggingfacem4_idefics_9b",
        "anthropic_claude_3_opus_20240229",
        "llava_1.6_mistral_7b",
        "qwen_vl_chat"
    ]

    concept9 = [
        "openai_gpt_4o_2024_05_13",
        "openai_gpt_4_vision_preview",
        "openai_gpt_4_1106_vision_preview",
        "anthropic_claude_3_opus_20240229",
        "idefics2-8b",
        "huggingfacem4_idefics_80b_instruct",
        "llava_1.6_vicuna_7b",
        "llava-v1.6-vicuna-13b",
        "InternVL-Chat-V1-5",
        "anthropic_claude_3_haiku_20240307",
        "mistralai_bakllava_v1_hf",
        "microsoft_llava_1.5_13b_hf",
        "microsoft_llava_1.5_7b_hf",
        "internlm-xcomposer2-4khd-7b",
        "llava-next-72b",
        "llava-v1.6-34b",
        "llava_1.6_mistral_7b",
        "llava_7b",
        "writer_palmyra_vision_003",
        "huggingfacem4_idefics_9b_instruct",
        "llava_1.6_13b",
        "llava_13b",
        "google_gemini_1.5_pro_preview_0409",
        "instructblip-vicuna-13b",
        "google_gemini_pro_vision",
        "instructblip-vicuna-7b",
        "google_gemini_1.0_pro_vision_001",
        "qwen_qwen_vl",
        "huggingfacem4_idefics_80b",
        "huggingfacem4_idefics_9b",
        "anthropic_claude_3_sonnet_20240229",
        "google_paligemma_3b_mix_448",
        "openflamingo_openflamingo_9b_vitl_mpt7b",
        "qwen_vl_chat"
    ]

    concept10 = [
        "anthropic_claude_3_opus_20240229",
        "llava-next-72b",
        "InternVL-Chat-V1-5",
        "instructblip-vicuna-13b",
        "llava-v1.6-34b",
        "llava_1.6_mistral_7b",
        "openai_gpt_4_vision_preview",
        "microsoft_llava_1.5_7b_hf",
        "microsoft_llava_1.5_13b_hf",
        "google_gemini_pro_vision",
        "openflamingo_openflamingo_9b_vitl_mpt7b",
        "mistralai_bakllava_v1_hf",
        "internlm-xcomposer2-4khd-7b",
        "llava_13b",
        "llava-v1.6-vicuna-13b",
        "llava_7b",
        "instructblip-vicuna-7b",
        "llava_1.6_vicuna_7b",
        "llava_1.6_13b",
        "huggingfacem4_idefics_80b",
        "google_paligemma_3b_mix_448",
        "openai_gpt_4_1106_vision_preview",
        "google_gemini_1.5_pro_preview_0409",
        "huggingfacem4_idefics_9b",
        "huggingfacem4_idefics_80b_instruct",
        "writer_palmyra_vision_003",
        "anthropic_claude_3_haiku_20240307",
        "anthropic_claude_3_sonnet_20240229",
        "google_gemini_1.0_pro_vision_001",
        "huggingfacem4_idefics_9b_instruct",
        "openai_gpt_4o_2024_05_13",
        "qwen_qwen_vl",
        "idefics2-8b",
        "qwen_vl_chat"
    ]



    # Gather all concepts into a list
    concepts = [concept1, concept2, concept3, concept4, concept5, concept6, concept7, concept8, concept9, concept10]
    concept_names = ["Concept1", "Concept2", "Concept3", "Concept4", "Concept5", "Concept6", "Concept7", "Concept8", "Concept9", "Concept10"]
    
    # Option 1: Run full analysis on all concepts
    print("Running full analysis on all concepts...")
    results = run_full_analysis(overall_ranking, concepts, concept_names)
    
    # Option 2: Analyze just one concept (as in your original code snippet)
    print("\n\nAnalyzing single concept (Concept6)...")
    analyze_single_concept(overall_ranking, concept2)