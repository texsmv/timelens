"""
Statistical testing utilities for comparing selected vs non-selected groups.
"""
import numpy as np
from scipy import stats
from typing import Dict, List, Union, Tuple, Any


def t_test_two_groups(values: List[float], group_labels: List[int]) -> Dict[str, Any]:
    """
    Perform independent samples t-test between two groups.
    
    Args:
        values: List of numerical values
        group_labels: List of 0s and 1s indicating group membership (0=Group B, 1=Group A/Selected)
    
    Returns:
        Dictionary with test results including statistic, p-value, and interpretation
    """
    try:
        values_array = np.array(values)
        labels_array = np.array(group_labels)
        
        # Split data into two groups
        group_a = values_array[labels_array == 1]  # Selected group
        group_b = values_array[labels_array == 0]  # Non-selected group
        
        if len(group_a) < 2 or len(group_b) < 2:
            return {
                "test_name": "Student's t-test",
                "error": "Insufficient data: Each group needs at least 2 observations",
                "valid": False
            }
        
        # Perform t-test
        statistic, p_value = stats.ttest_ind(group_a, group_b)
        
        # Calculate descriptive statistics
        mean_a = np.mean(group_a)
        mean_b = np.mean(group_b)
        std_a = np.std(group_a, ddof=1)
        std_b = np.std(group_b, ddof=1)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(group_a) - 1) * std_a**2 + (len(group_b) - 1) * std_b**2) / 
                            (len(group_a) + len(group_b) - 2))
        cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0
        
        return {
            "test_name": "Student's t-test",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "degrees_of_freedom": len(group_a) + len(group_b) - 2,
            "group_a_stats": {
                "name": "Selected",
                "n": len(group_a),
                "mean": float(mean_a),
                "std": float(std_a)
            },
            "group_b_stats": {
                "name": "Non-selected",
                "n": len(group_b),
                "mean": float(mean_b),
                "std": float(std_b)
            },
            "effect_size": {
                "cohens_d": float(cohens_d),
                "interpretation": _interpret_cohens_d(cohens_d)
            },
            "interpretation": _interpret_p_value(p_value),
            "valid": True
        }
    
    except Exception as e:
        return {
            "test_name": "Student's t-test",
            "error": f"Error performing t-test: {str(e)}",
            "valid": False
        }


def mann_whitney_test(values: List[float], group_labels: List[int]) -> Dict[str, Any]:
    """
    Perform Mann-Whitney U test (non-parametric alternative to t-test).
    
    Args:
        values: List of numerical values
        group_labels: List of 0s and 1s indicating group membership
    
    Returns:
        Dictionary with test results
    """
    try:
        values_array = np.array(values)
        labels_array = np.array(group_labels)
        
        # Split data into two groups
        group_a = values_array[labels_array == 1]  # Selected group
        group_b = values_array[labels_array == 0]  # Non-selected group
        
        if len(group_a) < 1 or len(group_b) < 1:
            return {
                "test_name": "Mann-Whitney U test",
                "error": "Insufficient data: Each group needs at least 1 observation",
                "valid": False
            }
        
        # Perform Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')
        
        # Calculate descriptive statistics
        median_a = np.median(group_a)
        median_b = np.median(group_b)
        
        return {
            "test_name": "Mann-Whitney U test",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "group_a_stats": {
                "name": "Selected",
                "n": len(group_a),
                "median": float(median_a),
                "q25": float(np.percentile(group_a, 25)),
                "q75": float(np.percentile(group_a, 75))
            },
            "group_b_stats": {
                "name": "Non-selected",
                "n": len(group_b),
                "median": float(median_b),
                "q25": float(np.percentile(group_b, 25)),
                "q75": float(np.percentile(group_b, 75))
            },
            "interpretation": _interpret_p_value(p_value),
            "valid": True
        }
    
    except Exception as e:
        return {
            "test_name": "Mann-Whitney U test",
            "error": f"Error performing Mann-Whitney U test: {str(e)}",
            "valid": False
        }


def chi_square_test(categories: List[Union[str, int]], group_labels: List[int]) -> Dict[str, Any]:
    """
    Perform Chi-squared test of independence for categorical variables.
    
    Args:
        categories: List of categorical values
        group_labels: List of 0s and 1s indicating group membership
    
    Returns:
        Dictionary with test results
    """
    try:
        categories_array = np.array(categories)
        labels_array = np.array(group_labels)
        
        # Create contingency table
        unique_categories = np.unique(categories_array)
        contingency_table = []
        
        for category in unique_categories:
            # Count occurrences in each group
            category_mask = categories_array == category
            count_group_a = np.sum(labels_array[category_mask] == 1)  # Selected
            count_group_b = np.sum(labels_array[category_mask] == 0)  # Non-selected
            contingency_table.append([count_group_a, count_group_b])
        
        contingency_table = np.array(contingency_table)
        
        # Check if we have enough data
        if contingency_table.size == 0 or np.sum(contingency_table) < 5:
            return {
                "test_name": "Chi-squared test",
                "error": "Insufficient data: Need at least 5 total observations",
                "valid": False
            }
        
        # Perform chi-squared test
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Calculate Cramér's V (effect size)
        n = np.sum(contingency_table)
        cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
        
        # Create detailed breakdown
        breakdown = []
        for i, category in enumerate(unique_categories):
            breakdown.append({
                "category": str(category),
                "selected_count": int(contingency_table[i, 0]),
                "non_selected_count": int(contingency_table[i, 1]),
                "selected_percentage": float(contingency_table[i, 0] / np.sum(contingency_table[:, 0]) * 100) if np.sum(contingency_table[:, 0]) > 0 else 0,
                "non_selected_percentage": float(contingency_table[i, 1] / np.sum(contingency_table[:, 1]) * 100) if np.sum(contingency_table[:, 1]) > 0 else 0
            })
        
        return {
            "test_name": "Chi-squared test",
            "statistic": float(chi2_stat),
            "p_value": float(p_value),
            "degrees_of_freedom": int(dof),
            "contingency_table": contingency_table.tolist(),
            "expected_frequencies": expected.tolist(),
            "effect_size": {
                "cramers_v": float(cramers_v),
                "interpretation": _interpret_cramers_v(cramers_v)
            },
            "breakdown": breakdown,
            "interpretation": _interpret_p_value(p_value),
            "valid": True
        }
    
    except Exception as e:
        return {
            "test_name": "Chi-squared test",
            "error": f"Error performing chi-squared test: {str(e)}",
            "valid": False
        }


def _interpret_p_value(p_value: float) -> str:
    """Interpret p-value for statistical significance."""
    if p_value < 0.001:
        return "Highly significant (p < 0.001)"
    elif p_value < 0.01:
        return "Very significant (p < 0.01)"
    elif p_value < 0.05:
        return "Significant (p < 0.05)"
    elif p_value < 0.1:
        return "Marginally significant (p < 0.1)"
    else:
        return "Not significant (p ≥ 0.05)"


def _interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "Negligible effect"
    elif abs_d < 0.5:
        return "Small effect"
    elif abs_d < 0.8:
        return "Medium effect"
    else:
        return "Large effect"


def _interpret_cramers_v(v: float) -> str:
    """Interpret Cramér's V effect size."""
    if v < 0.1:
        return "Negligible association"
    elif v < 0.3:
        return "Small association"
    elif v < 0.5:
        return "Medium association"
    else:
        return "Large association"


def run_numerical_tests(values: List[float], group_labels: List[int]) -> Dict[str, Any]:
    """
    Run both t-test and Mann-Whitney U test for numerical data.
    
    Args:
        values: List of numerical values
        group_labels: List of 0s and 1s indicating group membership
    
    Returns:
        Dictionary containing results from both tests
    """
    return {
        "t_test": t_test_two_groups(values, group_labels),
        "mann_whitney": mann_whitney_test(values, group_labels)
    }


def run_categorical_tests(categories: List[Union[str, int]], group_labels: List[int]) -> Dict[str, Any]:
    """
    Run chi-squared test for categorical data.
    
    Args:
        categories: List of categorical values
        group_labels: List of 0s and 1s indicating group membership
    
    Returns:
        Dictionary containing chi-squared test results
    """
    return {
        "chi_square": chi_square_test(categories, group_labels)
    }
