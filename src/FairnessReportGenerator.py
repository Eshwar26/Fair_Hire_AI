import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

class FairnessReportGenerator:
    """
    Generate comprehensive reports for fairness analysis in AI recruitment models.
    """
    
    def __init__(self, model_results, dataset_info=None):
        """
        Initialize the report generator.
        
        Parameters:
        -----------
        model_results : list
            List of dictionaries containing model results
        dataset_info : dict, optional
            Information about the dataset used
        """
        self.model_results = model_results
        self.dataset_info = dataset_info or {}
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    def generate_markdown_report(self, output_path):
        """
        Generate a comprehensive markdown report.
        
        Parameters:
        -----------
        output_path : str
            Path to save the report
            
        Returns:
        --------
        str
            Path to the generated report
        """
        report = []
        
        # Add report header
        report.append("# Fairness Analysis Report for AI-Driven Recruitment System")
        report.append(f"\nGenerated on: {self.timestamp}\n")
        
        # Add executive summary
        report.append("## Executive Summary")
        report.append(self._generate_executive_summary())
        
        # Add dataset information
        report.append("## Dataset Information")
        report.append(self._generate_dataset_info())
        
        # Add model comparison
        report.append("## Model Comparison")
        report.append(self._generate_model_comparison())
        
        # Add fairness analysis
        report.append("## Fairness Analysis")
        report.append(self._generate_fairness_analysis())
        
        # Add bias mitigation effectiveness
        report.append("## Bias Mitigation Effectiveness")
        report.append(self._generate_bias_mitigation_analysis())
        
        # Add feature importance analysis
        report.append("## Feature Importance Analysis")
        report.append(self._generate_feature_importance_analysis())
        
        # Add intersectional analysis
        report.append("## Intersectional Fairness Analysis")
        report.append(self._generate_intersectional_analysis())
        
        # Add recommendations
        report.append("## Recommendations")
        report.append(self._generate_recommendations())
        
        # Add technical implementation details
        report.append("## Technical Implementation Details")
        report.append(self._generate_technical_details())
        
        # Add references
        report.append("## References")
        report.append(self._generate_references())
        
        #  Write report to file
        # with open(output_path, 'w') as f:
        #     f.write('\n'.join(report))
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
            
        return output_path
    
    def _generate_executive_summary(self):
        """Generate the executive summary section"""
        
        # Find best performing model
        best_performance_model = max(self.model_results, 
                                    key=lambda x: x['Performance']['f1'])
        
        # Find most fair model (lowest demographic parity difference)
        fair_models = [m for m in self.model_results if 'demographic_parity_diff' in m['Fairness']]
        if fair_models:
            most_fair_model = min(fair_models, 
                                key=lambda x: abs(x['Fairness']['demographic_parity_diff']))
        else:
            most_fair_model = {'Model': 'Not available'}
        
        # Calculate average metrics
        avg_accuracy = np.mean([m['Performance']['accuracy'] for m in self.model_results])
        avg_f1 = np.mean([m['Performance']['f1'] for m in self.model_results])
        
        summary = [
            "This report presents a comprehensive analysis of fairness in AI-driven recruitment models for software engineering positions.",
            "The analysis focuses on identifying and mitigating bias in the candidate selection process.",
            "",
            "### Key Findings",
            "",
            f"- The best performing model in terms of predictive accuracy is **{best_performance_model['Model']}** with an F1 score of {best_performance_model['Performance']['f1']:.2f}.",
            f"- The model with the most balanced selection rates across demographic groups is **{most_fair_model['Model']}**.",
            f"- Overall, the models achieve an average accuracy of {avg_accuracy:.2f} and F1 score of {avg_f1:.2f}.",
            "",
            "### Bias Concerns",
        ]
        
        # Add information about biases found
        gender_bias_models = []
        for model in self.model_results:
            if 'gender' in model['Fairness'] and 'demographic_parity_diff' in model['Fairness']:
                if abs(model['Fairness']['demographic_parity_diff']) > 0.1:  # Threshold for significant bias
                    gender_bias_models.append(model['Model'])
        
        if gender_bias_models:
            summary.append(f"- Significant gender bias was detected in {len(gender_bias_models)} models: {', '.join(gender_bias_models)}.")
        else:
            summary.append("- No significant gender bias was detected in the evaluated models.")
        
        # Add summary of mitigation effectiveness
        mitigation_techniques = set()
        for model in self.model_results:
            if ' with ' in model['Model']:
                technique = model['Model'].split(' with ')[1]
                mitigation_techniques.add(technique)
        
        if mitigation_techniques:
            summary.append(f"- The following bias mitigation techniques were evaluated: {', '.join(mitigation_techniques)}.")
            
            # Find most effective technique
            baseline_models = [m for m in self.model_results if ' with ' not in m['Model']]
            mitigation_models = [m for m in self.model_results if ' with ' in m['Model']]
            
            if baseline_models and mitigation_models:
                avg_baseline_diff = np.mean([abs(m['Fairness'].get('demographic_parity_diff', 0)) 
                                           for m in baseline_models if 'demographic_parity_diff' in m['Fairness']])
                
                for technique in mitigation_techniques:
                    technique_models = [m for m in mitigation_models if technique in m['Model']]
                    avg_technique_diff = np.mean([abs(m['Fairness'].get('demographic_parity_diff', 0)) 
                                               for m in technique_models if 'demographic_parity_diff' in m['Fairness']])
                    
                    reduction = avg_baseline_diff - avg_technique_diff
                    if reduction > 0:
                        summary.append(f"- The **{technique}** technique reduced bias by approximately {reduction:.2f} (demographic parity difference).")
        
        summary.extend([
            "",
            "### Recommendations",
            "",
            "1. Implement the recommended fair recruitment model with appropriate bias mitigation techniques.",
            "2. Establish continuous monitoring of selection rates across demographic groups.",
            "3. Periodically retrain the model with diverse data to prevent bias drift.",
            "4. Combine automated screening with human oversight to ensure fairness in the final selection."
        ])
        
        return '\n'.join(summary)
    
    def _generate_dataset_info(self):
        """Generate the dataset information section"""
        
        if not self.dataset_info:
            return "No detailed dataset information available."
        
        info = [
            f"The analysis was performed on a dataset with the following characteristics:",
            "",
            f"- **Total samples**: {self.dataset_info.get('n_samples', 'Unknown')}",
            f"- **Features**: {self.dataset_info.get('n_features', 'Unknown')}",
            f"- **Training set size**: {self.dataset_info.get('train_size', 'Unknown')}",
            f"- **Test set size**: {self.dataset_info.get('test_size', 'Unknown')}",
            "",
            "### Demographic Distribution",
            ""
        ]
        
        if 'gender_distribution' in self.dataset_info:
            info.append("#### Gender Distribution")
            info.append("| Gender | Percentage |")
            info.append("|--------|------------|")
            for gender, pct in self.dataset_info['gender_distribution'].items():
                info.append(f"| {gender} | {pct:.1f}% |")
            info.append("")
            
        if 'ethnicity_distribution' in self.dataset_info:
            info.append("#### Ethnicity Distribution")
            info.append("| Ethnicity | Percentage |")
            info.append("|-----------|------------|")
            for ethnicity, pct in self.dataset_info['ethnicity_distribution'].items():
                info.append(f"| {ethnicity} | {pct:.1f}% |")
            info.append("")
            
        info.extend([
            "### Features",
            "",
            "The dataset includes the following types of features:",
            "",
            "- Technical skills (programming languages, frameworks, etc.)",
            "- Education information (degree, institution)",
            "- Experience metrics (years of experience, previous roles)",
            "- Project contributions and open source activity",
            "- Assessment scores (coding tests, problem-solving evaluations)",
            "",
            "### Protected Attributes",
            "",
            "The following protected attributes were used for fairness evaluation:",
            "",
            "- Gender",
            "- Ethnicity",
            "- Age",
            "",
            "### Proxy Variables",
            "",
            "The dataset includes potential proxy variables that might correlate with protected attributes:",
            "",
            "- Names (which may suggest gender or ethnicity)",
            "- Educational institutions (which may correlate with socioeconomic status)",
            "- Geographic location",
            "- Employment gaps"
        ])
        
        return '\n'.join(info)
    
    def _generate_model_comparison(self):
        """Generate the model comparison section"""
        
        if not self.model_results:
            return "No model comparison data available."
        
        # Create model comparison table
        comparison = [
            "### Performance Metrics",
            "",
            "| Model | Accuracy | Precision | Recall | F1 Score |",
            "|-------|----------|-----------|--------|----------|"
        ]
        
        for result in self.model_results:
            model_name = result['Model']
            perf = result['Performance']
            row = f"| {model_name} | {perf['accuracy']:.4f} | {perf['precision']:.4f} | {perf['recall']:.4f} | {perf['f1']:.4f} |"
            comparison.append(row)
            
        comparison.extend([
            "",
            "### Fairness Metrics",
            "",
            "| Model | Demographic Parity Diff | Male Selection Rate | Female Selection Rate |",
            "|-------|--------------------------|---------------------|------------------------|"
        ])
        
        for result in self.model_results:
            model_name = result['Model']
            fairness = result['Fairness']
            
            # Extract metrics if available
            demo_parity_diff = fairness.get('demographic_parity_diff', 'N/A')
            if demo_parity_diff != 'N/A':
                demo_parity_diff = f"{demo_parity_diff:.4f}"
                
            # Extract selection rates by gender
            male_rate = 'N/A'
            female_rate = 'N/A'
            
            if 'gender' in fairness:
                gender_df = fairness['gender']
                if 'Male' in gender_df['Group'].values:
                    male_idx = gender_df['Group'] == 'Male'
                    male_rate = f"{gender_df.loc[male_idx, 'Selection Rate'].iloc[0]:.4f}"
                    
                if 'Female' in gender_df['Group'].values:
                    female_idx = gender_df['Group'] == 'Female'
                    female_rate = f"{gender_df.loc[female_idx, 'Selection Rate'].iloc[0]:.4f}"
            
            row = f"| {model_name} | {demo_parity_diff} | {male_rate} | {female_rate} |"
            comparison.append(row)
            
        comparison.extend([
            "",
            "### Performance-Fairness Trade-off",
            "",
            "The relationship between model performance and fairness can be visualized in the accompanying charts. Generally, there is a trade-off between achieving high predictive accuracy and ensuring fairness across demographic groups.",
            "",
            "Key observations:",
            "",
            "1. Models with bias mitigation techniques tend to have slightly lower accuracy but better fairness metrics.",
            "2. The demographic parity difference (a measure of unfairness) varies significantly across different model architectures.",
            "3. Calibrated models offer a good balance between performance and fairness."
        ])
        
        return '\n'.join(comparison)
    
    def _generate_fairness_analysis(self):
        """Generate the fairness analysis section"""
        
        fairness_analysis = [
            "This section analyzes the fairness of each model based on various metrics.",
            "",
            "### Gender Fairness",
            "",
            "The selection rates and true positive rates across gender groups indicate the level of bias in each model:"
        ]
        
        # Create comparison across models
        for result in self.model_results:
            model_name = result['Model']
            fairness_analysis.append(f"\n#### {model_name}")
            
            if 'gender' in result['Fairness']:
                gender_df = result['Fairness']['gender']
                fairness_analysis.append("| Gender | Count | Selection Rate | True Positive Rate | False Positive Rate |")
                fairness_analysis.append("|--------|-------|---------------|-------------------|---------------------|")
                
                for _, row in gender_df.iterrows():
                    fairness_analysis.append(
                        f"| {row['Group']} | {int(row['Count'])} | {row['Selection Rate']:.4f} | "
                        f"{row['True Positive Rate']:.4f} | {row['False Positive Rate']:.4f} |"
                    )
                
                # Add disparity analysis if demographic parity diff is available
                if 'demographic_parity_diff' in result['Fairness']:
                    diff = result['Fairness']['demographic_parity_diff']
                    fairness_analysis.append(f"\nDemographic Parity Difference: {diff:.4f}")
                    
                    if abs(diff) > 0.1:
                        fairness_analysis.append("\n⚠️ **Significant gender bias detected.** The selection rates between males and females differ by more than 10 percentage points.")
                    elif abs(diff) > 0.05:
                        fairness_analysis.append("\n⚠️ **Moderate gender bias detected.** The selection rates between males and females differ by more than 5 percentage points.")
                    else:
                        fairness_analysis.append("\n✅ **Low gender bias.** The selection rates between males and females are relatively balanced.")
            else:
                fairness_analysis.append("\nNo gender fairness data available for this model.")
        
        # Add ethnicity fairness analysis
        fairness_analysis.append("\n### Ethnicity Fairness")
        
        for result in self.model_results:
            if 'ethnicity' in result['Fairness']:
                model_name = result['Model']
                fairness_analysis.append(f"\n#### {model_name}")
                
                ethnicity_df = result['Fairness']['ethnicity']
                fairness_analysis.append("| Ethnicity | Count | Selection Rate | True Positive Rate |")
                fairness_analysis.append("|-----------|-------|---------------|-------------------|")
                
                for _, row in ethnicity_df.iterrows():
                    fairness_analysis.append(
                        f"| {row['Group']} | {int(row['Count'])} | {row['Selection Rate']:.4f} | "
                        f"{row['True Positive Rate']:.4f} |"
                    )
                
                # Calculate maximum disparity
                if len(ethnicity_df) > 1:
                    max_rate = ethnicity_df['Selection Rate'].max()
                    min_rate = ethnicity_df['Selection Rate'].min()
                    max_disparity = max_rate - min_rate
                    
                    fairness_analysis.append(f"\nMaximum Selection Rate Disparity: {max_disparity:.4f}")
                    
                    if max_disparity > 0.2:
                        fairness_analysis.append("\n⚠️ **High ethnicity bias detected.** The selection rates across ethnic groups vary by more than 20 percentage points.")
                    elif max_disparity > 0.1:
                        fairness_analysis.append("\n⚠️ **Moderate ethnicity bias detected.** The selection rates across ethnic groups vary by more than 10 percentage points.")
                    else:
                        fairness_analysis.append("\n✅ **Low ethnicity bias.** The selection rates across ethnic groups are relatively balanced.")
        
        # Add 80% rule analysis
        fairness_analysis.append("\n### Disparate Impact Analysis (80% Rule)")
        fairness_analysis.append("\nThe 80% rule (or four-fifths rule) is a legal guideline used to determine adverse impact in employment decisions. It states that the selection rate for any protected group should be at least 80% of the selection rate for the group with the highest selection rate.")
        
        for result in self.model_results:
            model_name = result['Model']
            fairness_analysis.append(f"\n#### {model_name}")
            
            if 'gender' in result['Fairness']:
                gender_df = result['Fairness']['gender']
                
                if len(gender_df) > 1:
                    max_rate = gender_df['Selection Rate'].max()
                    min_rate = gender_df['Selection Rate'].min()
                    
                    if max_rate > 0:
                        ratio = min_rate / max_rate
                        fairness_analysis.append(f"Gender Selection Rate Ratio: {ratio:.4f}")
                        
                        if ratio < 0.8:
                            fairness_analysis.append("\n⚠️ **Fails the 80% rule.** This model shows evidence of disparate impact by gender.")
                        else:
                            fairness_analysis.append("\n✅ **Passes the 80% rule.** This model does not show evidence of disparate impact by gender.")
            
            if 'ethnicity' in result['Fairness']:
                ethnicity_df = result['Fairness']['ethnicity']
                
                if len(ethnicity_df) > 1:
                    max_rate = ethnicity_df['Selection Rate'].max()
                    min_rate = ethnicity_df['Selection Rate'].min()
                    
                    if max_rate > 0:
                        ratio = min_rate / max_rate
                        fairness_analysis.append(f"\nEthnicity Selection Rate Ratio: {ratio:.4f}")
                        
                        if ratio < 0.8:
                            fairness_analysis.append("\n⚠️ **Fails the 80% rule.** This model shows evidence of disparate impact by ethnicity.")
                        else:
                            fairness_analysis.append("\n✅ **Passes the 80% rule.** This model does not show evidence of disparate impact by ethnicity.")
        
        return '\n'.join(fairness_analysis)
    
    def _generate_bias_mitigation_analysis(self):
        """Generate the bias mitigation effectiveness analysis section"""
        
        mitigation_analysis = [
            "This section evaluates the effectiveness of different bias mitigation techniques.",
            ""
        ]
        
        # Identify baseline and mitigation models
        baseline_models = [m for m in self.model_results if ' with ' not in m['Model']]
        mitigation_models = [m for m in self.model_results if ' with ' in m['Model']]
        
        if not baseline_models or not mitigation_models:
            return "No bias mitigation comparison available."
        
        # Extract techniques
        techniques = set()
        for model in mitigation_models:
            if ' with ' in model['Model']:
                technique = model['Model'].split(' with ')[1]
                techniques.add(technique)
        
        # Compare each technique with baseline
        for technique in techniques:
            technique_models = [m for m in mitigation_models if technique in m['Model']]
            
            if not technique_models:
                continue
                
            mitigation_analysis.append(f"### {technique} Technique")
            mitigation_analysis.append("\nComparison with baseline models:")
            mitigation_analysis.append("\n| Metric | Baseline Average | With Mitigation | Difference |")
            mitigation_analysis.append("|--------|------------------|----------------|------------|")
            
            # Calculate average metrics
            baseline_acc = np.mean([m['Performance']['accuracy'] for m in baseline_models])
            baseline_f1 = np.mean([m['Performance']['f1'] for m in baseline_models])
            
            technique_acc = np.mean([m['Performance']['accuracy'] for m in technique_models])
            technique_f1 = np.mean([m['Performance']['f1'] for m in technique_models])
            
            # Performance metrics
            acc_diff = technique_acc - baseline_acc
            f1_diff = technique_f1 - baseline_f1
            
            mitigation_analysis.append(f"| Accuracy | {baseline_acc:.4f} | {technique_acc:.4f} | {acc_diff:+.4f} |")
            mitigation_analysis.append(f"| F1 Score | {baseline_f1:.4f} | {technique_f1:.4f} | {f1_diff:+.4f} |")
            
            # Fairness metrics - focus on demographic parity difference
            baseline_fair_models = [m for m in baseline_models if 'demographic_parity_diff' in m['Fairness']]
            technique_fair_models = [m for m in technique_models if 'demographic_parity_diff' in m['Fairness']]
            
            if baseline_fair_models and technique_fair_models:
                baseline_dpd = np.mean([abs(m['Fairness']['demographic_parity_diff']) for m in baseline_fair_models])
                technique_dpd = np.mean([abs(m['Fairness']['demographic_parity_diff']) for m in technique_fair_models])
                dpd_diff = technique_dpd - baseline_dpd
                
                mitigation_analysis.append(f"| Demographic Parity Difference | {baseline_dpd:.4f} | {technique_dpd:.4f} | {dpd_diff:+.4f} |")
                
                if dpd_diff < 0:
                    mitigation_analysis.append(f"\n✅ **{technique} reduced bias** by {abs(dpd_diff):.4f} demographic parity difference points.")
                else:
                    mitigation_analysis.append(f"\n⚠️ **{technique} did not reduce bias** and actually increased it by {dpd_diff:.4f} points.")
            
            # Selection rate comparison
            mitigation_analysis.append("\n#### Selection Rate Changes")
            
            baseline_gender_models = [m for m in baseline_models if 'gender' in m['Fairness']]
            technique_gender_models = [m for m in technique_models if 'gender' in m['Fairness']]
            
            if baseline_gender_models and technique_gender_models:
                # Average selection rates for gender groups in baseline
                baseline_male_rates = []
                baseline_female_rates = []
                
                for model in baseline_gender_models:
                    gender_df = model['Fairness']['gender']
                    
                    if 'Male' in gender_df['Group'].values:
                        male_idx = gender_df['Group'] == 'Male'
                        baseline_male_rates.append(gender_df.loc[male_idx, 'Selection Rate'].iloc[0])
                        
                    if 'Female' in gender_df['Group'].values:
                        female_idx = gender_df['Group'] == 'Female'
                        baseline_female_rates.append(gender_df.loc[female_idx, 'Selection Rate'].iloc[0])
                
                # Average selection rates for gender groups with technique
                technique_male_rates = []
                technique_female_rates = []
                
                for model in technique_gender_models:
                    gender_df = model['Fairness']['gender']
                    
                    if 'Male' in gender_df['Group'].values:
                        male_idx = gender_df['Group'] == 'Male'
                        technique_male_rates.append(gender_df.loc[male_idx, 'Selection Rate'].iloc[0])
                        
                    if 'Female' in gender_df['Group'].values:
                        female_idx = gender_df['Group'] == 'Female'
                        technique_female_rates.append(gender_df.loc[female_idx, 'Selection Rate'].iloc[0])
                
                # Calculate averages
                avg_baseline_male = np.mean(baseline_male_rates) if baseline_male_rates else 0
                avg_baseline_female = np.mean(baseline_female_rates) if baseline_female_rates else 0
                avg_technique_male = np.mean(technique_male_rates) if technique_male_rates else 0
                avg_technique_female = np.mean(technique_female_rates) if technique_female_rates else 0
                
                mitigation_analysis.append("\n| Group | Baseline Selection Rate | With Mitigation | Change |")
                mitigation_analysis.append("|-------|--------------------------|----------------|--------|")
                mitigation_analysis.append(f"| Male | {avg_baseline_male:.4f} | {avg_technique_male:.4f} | {avg_technique_male - avg_baseline_male:+.4f} |")
                mitigation_analysis.append(f"| Female | {avg_baseline_female:.4f} | {avg_technique_female:.4f} | {avg_technique_female - avg_baseline_female:+.4f} |")
        
        # Summary of effectiveness
        mitigation_analysis.append("\n### Overall Effectiveness Ranking")
        
        # For each technique, calculate improvement in fairness
        technique_improvements = []
        
        for technique in techniques:
            technique_models = [m for m in mitigation_models if technique in m['Model']]
            technique_fair_models = [m for m in technique_models if 'demographic_parity_diff' in m['Fairness']]
            
            if baseline_fair_models and technique_fair_models:
                baseline_dpd = np.mean([abs(m['Fairness']['demographic_parity_diff']) for m in baseline_fair_models])
                technique_dpd = np.mean([abs(m['Fairness']['demographic_parity_diff']) for m in technique_fair_models])
                dpd_improvement = baseline_dpd - technique_dpd
                
                # Also consider performance impact
                baseline_f1 = np.mean([m['Performance']['f1'] for m in baseline_models])
                technique_f1 = np.mean([m['Performance']['f1'] for m in technique_models])
                f1_impact = technique_f1 - baseline_f1
                
                technique_improvements.append((technique, dpd_improvement, f1_impact))
        
        # Rank techniques by fairness improvement
        if technique_improvements:
            technique_improvements.sort(key=lambda x: x[1], reverse=True)
            
            mitigation_analysis.append("\nRanking of techniques by bias reduction effectiveness:")
            mitigation_analysis.append("\n| Rank | Technique | Bias Reduction | Performance Impact |")
            mitigation_analysis.append("|------|-----------|----------------|-------------------|")
            
            for i, (technique, fairness_imp, perf_impact) in enumerate(technique_improvements):
                mitigation_analysis.append(f"| {i+1} | {technique} | {fairness_imp:.4f} | {perf_impact:+.4f} |")
            
            # Most effective technique
            best_technique = technique_improvements[0][0]
            mitigation_analysis.append(f"\n**{best_technique}** is the most effective technique for reducing bias while maintaining performance.")
        
        return '\n'.join(mitigation_analysis)
    
    def _generate_feature_importance_analysis(self):
        """Generate the feature importance analysis section"""
        
        # Check if feature importance is available
        models_with_importance = []
        for result in self.model_results:
            if hasattr(result, 'feature_importance') and callable(result.feature_importance):
                importance = result.feature_importance()
                if importance is not None:
                    models_with_importance.append((result['Model'], importance))
        
        if not models_with_importance:
            return "No feature importance analysis available."
        
        feature_analysis = [
            "This section analyzes the importance of different features in the recruitment models and their potential relationship to bias.",
            "",
            "### Feature Importance Rankings"
        ]
        
        # Generate feature importance tables for each model
        for model_name, importance_df in models_with_importance:
            feature_analysis.append(f"\n#### {model_name}")
            feature_analysis.append("\n| Feature | Importance |")
            feature_analysis.append("|---------|------------|")
            
            # Get top 10 features
            top_features = importance_df.sort_values('Importance', ascending=False).head(10)
            
            for _, row in top_features.iterrows():
                feature_analysis.append(f"| {row['Feature']} | {row['Importance']:.4f} |")
        
        # Add analysis of proxy variables
        feature_analysis.extend([
            "",
            "### Potential Proxy Variables",
            "",
            "Some features may act as proxies for protected attributes, indirectly introducing bias:",
            "",
            "1. **Educational Institution** - May correlate with socioeconomic status and race",
            "2. **Location** - May correlate with race, ethnicity, or socioeconomic status",
            "3. **Names** - May suggest gender or ethnicity",
            "4. **Work Gaps** - May disproportionately affect women due to family responsibilities",
            "",
            "### Recommendations for Feature Selection",
            "",
            "Based on the feature importance analysis, we recommend:",
            "",
            "1. **Focus on skill-based assessments** - Coding tests, problem-solving scores, and technical skills demonstrate high predictive power with lower bias risk",
            "2. **Anonymize personal details** - Remove names, specific educational institutions, and demographic information during initial screening",
            "3. **Use balanced feature sets** - Incorporate diverse signals of candidate quality rather than over-relying on traditional credentials",
            "4. **Regularize models** - Apply proper regularization to prevent overfitting to biased patterns in the training data"
        ])
        
        return '\n'.join(feature_analysis)
    
    def _generate_intersectional_analysis(self):
        """Generate the intersectional analysis section"""
        
        # Check if intersectional data is available
        models_with_intersectional = []
        for result in self.model_results:
            has_intersectional = False
            for key in result['Fairness']:
                if '_' in key and any(g in key for g in ['Male', 'Female']):
                    has_intersectional = True
                    break
            
            if has_intersectional:
                models_with_intersectional.append(result)
        
        if not models_with_intersectional:
            return "No intersectional analysis available."
        
        intersectional = [
            "This section explores how biases may compound across multiple demographic dimensions, particularly the intersection of gender and ethnicity.",
            "",
            "### Intersectional Selection Rates"
        ]
        
        for result in models_with_intersectional:
            model_name = result['Model']
            intersectional.append(f"\n#### {model_name}")
            
            # Extract intersectional metrics
            intersectional_metrics = {}
            overall_rate = 0
            
            for key, value in result['Fairness'].items():
                if '_' in key and '_selection_rate' in key:
                    # Parse group names from the key
                    gender, ethnicity = key.replace('_selection_rate', '').split('_', 1)
                    intersectional_metrics[(gender, ethnicity)] = value
                    overall_rate += value
            
            if intersectional_metrics:
                # Calculate average selection rate
                overall_rate = overall_rate / len(intersectional_metrics) if intersectional_metrics else 0
                
                # Create table
                intersectional.append("\n| Gender | Ethnicity | Selection Rate | Difference from Average |")
                intersectional.append("|--------|-----------|---------------|-------------------------|")
                
                for (gender, ethnicity), rate in sorted(intersectional_metrics.items()):
                    diff = rate - overall_rate
                    intersectional.append(f"| {gender} | {ethnicity} | {rate:.4f} | {diff:+.4f} |")
                
                # Identify highest and lowest groups
                highest_group = max(intersectional_metrics.items(), key=lambda x: x[1])
                lowest_group = min(intersectional_metrics.items(), key=lambda x: x[1])
                
                max_disparity = highest_group[1] - lowest_group[1]
                
                intersectional.append(f"\nMaximum Intersectional Disparity: {max_disparity:.4f}")
                intersectional.append(f"- Highest selection rate: {highest_group[0][0]} {highest_group[0][1]} ({highest_group[1]:.4f})")
                intersectional.append(f"- Lowest selection rate: {lowest_group[0][0]} {lowest_group[0][1]} ({lowest_group[1]:.4f})")
                
                if max_disparity > 0.2:
                    intersectional.append("\n⚠️ **Significant intersectional bias detected.** Selection rates vary by more than 20 percentage points across different gender-ethnicity combinations.")
                elif max_disparity > 0.1:
                    intersectional.append("\n⚠️ **Moderate intersectional bias detected.** Selection rates vary by more than 10 percentage points across different gender-ethnicity combinations.")
                else:
                    intersectional.append("\n✅ **Low intersectional bias.** Selection rates are relatively balanced across different gender-ethnicity combinations.")
        
        # Add analysis of compound discrimination
        intersectional.extend([
            "",
            "### Compound Discrimination Analysis",
            "",
            "Intersectional analysis reveals how different forms of discrimination may interact and compound:",
            "",
            "1. **Compounding Effect** - When multiple marginalized identities intersect, bias effects may be multiplicative rather than additive",
            "2. **Unique Challenges** - Groups at specific intersections may face unique challenges not captured by looking at single attributes",
            "3. **Hidden Disparities** - Overall metrics for a single protected attribute may mask significant disparities at intersections",
            "",
            "### Recommendations for Intersectional Fairness",
            "",
            "1. **Targeted Interventions** - Design bias mitigation strategies that specifically address the most affected intersectional groups",
            "2. **Disaggregated Monitoring** - Regularly monitor selection rates across intersectional categories, not just individual protected attributes",
            "3. **Representation in Training Data** - Ensure sufficient representation of all intersectional groups in training data",
            "4. **Customized Thresholds** - Consider setting group-specific thresholds where legally permissible to balance selection rates"
        ])
        
        return '\n'.join(intersectional)
    
    def _generate_recommendations(self):
        """Generate the recommendations section"""
        
        recommendations = [
            "Based on the comprehensive analysis of model performance and fairness, we offer the following recommendations for implementing an unbiased AI recruitment system for software engineering roles:",
            "",
            "### Model Selection"
        ]
        
        # Find the best model considering both performance and fairness
        best_model = None
        best_score = -float('inf')
        
        for result in self.model_results:
            if 'demographic_parity_diff' in result['Fairness']:
                # Combine accuracy and fairness (lower demographic parity diff is better)
                perf_score = result['Performance']['f1']
                fairness_score = 1.0 - abs(result['Fairness']['demographic_parity_diff'])
                combined_score = 0.6 * perf_score + 0.4 * fairness_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_model = result['Model']
        
        if best_model:
            recommendations.append(f"\nRecommended Model: **{best_model}**")
            recommendations.append("\nThis model offers the best balance between predictive performance and fairness across demographic groups.")
        
        # General recommendations
        recommendations.extend([
            "",
            "### Implementation Guidelines",
            "",
            "1. **Preprocessing and Feature Engineering**",
            "   - Remove direct identifiers (names, photos) during initial screening",
            "   - Apply consistent normalization across all numerical features",
            "   - Use blind evaluations for technical assessments",
            "",
            "2. **Bias Mitigation Techniques**",
            "   - Apply data rebalancing or reweighing to ensure fair representation",
            "   - Consider calibrated probability estimates for better threshold selection",
            "   - Implement threshold adjustments to ensure equitable selection rates",
            "",
            "3. **Model Training and Validation**",
            "   - Use stratified sampling to ensure representation of all groups",
            "   - Apply regularization to prevent overfitting to biased patterns",
            "   - Validate on diverse test sets that represent all demographic groups",
            "",
            "4. **Post-Processing and Decision Making**",
            "   - Apply group-specific thresholds where legally permissible",
            "   - Implement human review for borderline cases",
            "   - Document decision rationales for transparency and accountability",
            "",
            "### Monitoring and Maintenance",
            "",
            "1. **Continuous Fairness Monitoring**",
            "   - Track selection rates across demographic groups over time",
            "   - Monitor for concept drift in both performance and fairness metrics",
            "   - Establish regular fairness audits by independent reviewers",
            "",
            "2. **Feedback Loops**",
            "   - Collect post-hire performance data to validate model predictions",
            "   - Incorporate successful candidate outcomes back into model training",
            "   - Adjust feature importance based on actual job performance",
            "",
            "3. **Documentation and Transparency**",
            "   - Maintain comprehensive documentation of model architecture and decisions",
            "   - Create clear explanation mechanisms for rejected candidates",
            "   - Provide appeal processes for candidates who believe they were unfairly evaluated",
            "",
            "### Legal and Ethical Considerations",
            "",
            "1. **Compliance with Anti-Discrimination Laws**",
            "   - Ensure the system adheres to relevant employment laws (e.g., Title VII, EEOC guidelines)",
            "   - Apply the four-fifths rule to evaluate adverse impact",
            "   - Document business necessity for any practices with disparate impact",
            "",
            "2. **Transparency and Explainability**",
            "   - Provide clear explanations of how the AI system influences hiring decisions",
            "   - Ensure human oversight and accountability for all automated decisions",
            "   - Maintain logs of all model outputs for audit purposes",
            "",
            "3. **Data Privacy and Security**",
            "   - Follow best practices for securing candidate data",
            "   - Implement data retention policies that comply with relevant regulations",
            "   - Obtain appropriate consent for automated processing of applications"
        ])
        
        return '\n'.join(recommendations)
    
    def _generate_technical_details(self):
        """Generate the technical implementation details section"""
        
        technical_details = [
            "This section provides technical details on the implementation of the AI recruitment system, focusing on the model architecture, bias mitigation techniques, and evaluation methodology.",
            "",
            "### Model Architecture",
            "",
            "The recommended model architecture consists of the following components:",
            "",
            "1. **Feature Preprocessing**",
            "   - Standardization of numerical features (z-score normalization)",
            "   - One-hot encoding of categorical variables",
            "   - Text embedding for unstructured data (e.g., project descriptions)",
            "",
            "2. **Core Model**",
            "   - Ensemble approach combining multiple base classifiers",
            "   - Gradient boosting for technical skill assessment",
            "   - Calibrated probability outputs for better threshold selection",
            "",
            "3. **Post-Processing Layer**",
            "   - Group-specific threshold optimization",
            "   - Fairness constraints implementation",
            "   - Confidence scoring for human review decisions",
            "",
            "### Bias Mitigation Implementation",
            "",
            "1. **Pre-Processing Techniques**",
            "   ```python",
            "   # Sample reweighing implementation",
            "   def compute_sample_weights(y_train, protected_attributes):",
            "       # Calculate expected vs. observed probabilities",
            "       weights = np.ones(len(y_train))",
            "       for gender in protected_attributes['gender'].unique():",
            "           for ethnicity in protected_attributes['ethnicity'].unique():",
            "               for outcome in [0, 1]:",
            "                   # Create mask for this group and outcome",
            "                   mask = ((protected_attributes['gender'] == gender) & ",
            "                           (protected_attributes['ethnicity'] == ethnicity) & ",
            "                           (y_train == outcome))",
            "                   ",
            "                   # Calculate expected and observed probabilities",
            "                   expected = sum(y_train == outcome) / len(y_train)",
            "                   observed = sum(mask) / len(y_train)",
            "                   ",
            "                   # Adjust weights",
            "                   if observed > 0:",
            "                       weights[mask] = expected / observed",
            "       return weights",
            "   ```",
            "",
            "2. **In-Processing Techniques**",
            "   ```python",
            "   # Fairness constraint in the loss function",
            "   def fair_loss(y_true, y_pred, protected_attributes):",
            "       # Base loss (e.g., binary cross-entropy)",
            "       base_loss = log_loss(y_true, y_pred)",
            "       ",
            "       # Fairness penalty (demographic parity)",
            "       male_mask = protected_attributes['gender'] == 'Male'",
            "       female_mask = protected_attributes['gender'] == 'Female'",
            "       ",
            "       male_selection = np.mean(y_pred[male_mask])",
            "       female_selection = np.mean(y_pred[female_mask])",
            "       ",
            "       fairness_penalty = abs(male_selection - female_selection)",
            "       ",
            "       # Combined loss with fairness constraint",
            "       alpha = 0.3  # Fairness weight",
            "       combined_loss = base_loss + alpha * fairness_penalty",
            "       ",
            "       return combined_loss",
            "   ```",
            "",
            "3. **Post-Processing Techniques**",
            "   ```python",
            "   # Group-specific threshold optimization",
            "   def optimize_thresholds(model, X_val, y_val, protected_val):",
            "       probas = model.predict_proba(X_val)[:,1]",
            "       thresholds = {}",
            "       ",
            "       for gender in protected_val['gender'].unique():",
            "           gender_mask = protected_val['gender'] == gender",
            "           ",
            "           best_threshold = 0.5  # Default",
            "           best_balanced_accuracy = 0.0",
            "           ",
            "           # Try different thresholds",
            "           for threshold in np.arange(0.1, 0.9, 0.05):",
            "               preds = (probas[gender_mask] >= threshold).astype(int)",
            "               bal_acc = balanced_accuracy_score(y_val[gender_mask], preds)",
            "               ",
            "               if bal_acc > best_balanced_accuracy:",
            "                   best_balanced_accuracy = bal_acc",
            "                   best_threshold = threshold",
            "           ",
            "           thresholds[gender] = best_threshold",
            "       ",
            "       return thresholds",
            "   ```",
            "",
            "### Fairness Evaluation Methodology",
            "",
            "The system evaluates fairness using the following metrics and methodology:",
            "",
            "1. **Demographic Parity**",
            "   ```python",
            "   def demographic_parity_difference(y_pred, protected_attributes):",
            "       # Create masks for different groups",
            "       male_mask = protected_attributes['gender'] == 'Male'",
            "       female_mask = protected_attributes['gender'] == 'Female'",
            "       ",
            "       # Calculate selection rates",
            "       male_selection_rate = y_pred[male_mask].mean()",
            "       female_selection_rate = y_pred[female_mask].mean()",
            "       ",
            "       return male_selection_rate - female_selection_rate",
            "   ```",
            "",
            "2. **Equal Opportunity**",
            "   ```python",
            "   def equal_opportunity_difference(y_true, y_pred, protected_attributes):",
            "       # Create masks for different groups",
            "       male_mask = protected_attributes['gender'] == 'Male'",
            "       female_mask = protected_attributes['gender'] == 'Female'",
            "       ",
            "       # Calculate true positive rates",
            "       male_tpr = recall_score(y_true[male_mask], y_pred[male_mask])",
            "       female_tpr = recall_score(y_true[female_mask], y_pred[female_mask])",
            "       ",
            "       return male_tpr - female_tpr",
            "   ```",
            "",
            "3. **Intersectional Fairness**",
            "   ```python",
            "   def intersectional_disparities(y_pred, protected_attributes):",
            "       # Create a DataFrame to store selection rates",
            "       group_rates = []",
            "       ",
            "       # Calculate selection rates for each intersection",
            "       for gender in protected_attributes['gender'].unique():",
            "           for ethnicity in protected_attributes['ethnicity'].unique():",
            "               mask = ((protected_attributes['gender'] == gender) & ",
            "                       (protected_attributes['ethnicity'] == ethnicity))",
            "               ",
            "               # Skip if group is too small",
            "               if sum(mask) < 30:",
            "                   continue",
            "                   ",
            "               selection_rate = y_pred[mask].mean()",
            "               group_rates.append({",
            "                   'Gender': gender,",
            "                   'Ethnicity': ethnicity,",
            "                   'Selection Rate': selection_rate,",
            "                   'Count': sum(mask)",
            "               })",
            "       ",
            "       return pd.DataFrame(group_rates)",
            "   ```"
        ]
        
        return '\n'.join(technical_details)
    
    def _generate_references(self):
        """Generate the references section"""
        
        references = [
            "1. Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2021). A survey on bias and fairness in machine learning. ACM Computing Surveys (CSUR), 54(6), 1-35.",
            "",
            "2. Barocas, S., Hardt, M., & Narayanan, A. (2019). Fairness and machine learning. fairmlbook.org.",
            "",
            "3. Kamiran, F., & Calders, T. (2012). Data preprocessing techniques for classification without discrimination. Knowledge and Information Systems, 33(1), 1-33.",
            "",
            "4. Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. Advances in Neural Information Processing Systems, 29, 3315-3323.",
            "",
            "5. Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R. (2012). Fairness through awareness. Proceedings of the 3rd Innovations in Theoretical Computer Science Conference, 214-226.",
            "",
            "6. Zafar, M. B., Valera, I., Gomez Rodriguez, M., & Gummadi, K. P. (2017). Fairness beyond disparate treatment & disparate impact: Learning classification without disparate mistreatment. Proceedings of the 26th International Conference on World Wide Web, 1171-1180.",
            "",
            "7. Pleiss, G., Raghavan, M., Wu, F., Kleinberg, J., & Weinberger, K. Q. (2017). On fairness and calibration. Advances in Neural Information Processing Systems, 30, 5680-5689.",
            "",
            "8. Feldman, M., Friedler, S. A., Moeller, J., Scheidegger, C., & Venkatasubramanian, S. (2015). Certifying and removing disparate impact. Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 259-268.",
            "",
            "9. Crenshaw, K. (1989). Demarginalizing the intersection of race and sex: A Black feminist critique of antidiscrimination doctrine, feminist theory and antiracist politics. University of Chicago Legal Forum, 139-167.",
            "",
            "10. Kearns, M., Neel, S., Roth, A., & Wu, Z. S. (2018). Preventing fairness gerrymandering: Auditing and learning for subgroup fairness. International Conference on Machine Learning, 2564-2572.",
            "",
            "11. Holstein, K., Wortman Vaughan, J., Daumé III, H., Dudik, M., & Wallach, H. (2019). Improving fairness in machine learning systems: What do industry practitioners need? Proceedings of the 2019 CHI Conference on Human Factors in Computing Systems, 1-16.",
            "",
            "12. Raghavan, M., Barocas, S., Kleinberg, J., & Levy, K. (2020). Mitigating bias in algorithmic hiring: Evaluating claims and practices. Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency, 469-481.",
            "",
            "13. Beutel, A., Chen, J., Doshi, T., Qian, H., Woodruff, A., Luu, C., ... & Chi, E. H. (2019). Putting fairness principles into practice: Challenges, metrics, and improvements. Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society, 453-459.",
            "",
            "14. Garg, S., Perot, V., Limtiaco, N., Taly, A., Chi, E. H., & Beutel, A. (2020). Counterfactual fairness in text classification through robustness. Proceedings of the 2020 AAAI/ACM Conference on AI, Ethics, and Society, 219-226.",
            "",
            "15. Pessach, D., & Shmueli, E. (2020). Algorithmic fairness. arXiv preprint arXiv:2001.09784."
        ]
        
        return '\n'.join(references)
    
    def generate_html_report(self, output_path):
        """
        Generate an HTML report for easier viewing and interaction.
        
        Parameters:
        -----------
        output_path : str
            Path to save the HTML report
            
        Returns:
        --------
        str
            Path to the generated report
        """
        # Use a simple template engine or manual HTML construction
        html_head = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Fairness Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 0; color: #333; }
                .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
                h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
                h2 { color: #2980b9; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 10px; }
                h3 { color: #3498db; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .alert { padding: 15px; margin: 10px 0; border-radius: 4px; }
                .warning { background-color: #ffe6e6; border-left: 5px solid #ff0000; }
                .success { background-color: #e6ffe6; border-left: 5px solid #00cc00; }
                .info { background-color: #e6f7ff; border-left: 5px solid #0099ff; }
                .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 0.9em; color: #777; }
            </style>
        </head>
        <body>
            <div class="container">
        """
        
        html_foot = """
            <div class="footer">
                <p>Generated on: %s</p>
            </div>
            </div>
        </body>
        </html>
        """ % self.timestamp
        
        # Convert markdown to HTML (simple implementation)
        md_report = self.generate_markdown_report(output_path + ".md")
        
        # with open(md_report, 'r') as f:
        #     md_content = f.read()
        
        with open(md_report, 'r', encoding='utf-8') as f:
            md_content = f.read()    
            
        
        # Very basic markdown to HTML conversion
        html_content = md_content
        # Headers
        html_content = html_content.replace("# ", "<h1>").replace(" #", "</h1>")
        html_content = html_content.replace("## ", "<h2>").replace(" ##", "</h2>")
        html_content = html_content.replace("### ", "<h3>").replace(" ###", "</h3>")
        # Lists
        html_content = html_content.replace("- ", "<li>").replace("\n", "</li>\n")
        # Tables (simplified)
        html_content = html_content.replace("| ", "<td>").replace(" |", "</td>")
        html_content = html_content.replace("\n", "</tr>\n<tr>")
        
        # Combine HTML parts
        html_report = html_head + html_content + html_foot
        
        # # Write HTML report
        # with open(output_path, 'w') as f:
        #     f.write(html_report)
            
        # with open(output_path, 'w', encoding='utf-8') as f:
        #     f.write(html_report)
            
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_report)    
            
            
        return output_path
#         import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime
# import os

# class FairnessReportGenerator:
#     """
#     Generate comprehensive reports for fairness analysis in AI recruitment models.
#     """
    
#     def __init__(self, model_results, dataset_info=None):
#         """
#         Initialize the report generator.
        
#         Parameters:
#         -----------
#         model_results : list
#             List of dictionaries containing model results
#         dataset_info : dict, optional
#             Information about the dataset used
#         """
#         self.model_results = model_results
#         self.dataset_info = dataset_info or {}
#         self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
#     def generate_markdown_report(self, output_path):
#         """
#         Generate a comprehensive markdown report.
        
#         Parameters:
#         -----------
#         output_path : str
#             Path to save the report
            
#         Returns:
#         --------
#         str
#             Path to the generated report
#         """
#         report = []
        
#         # Add report header
#         report.append("# Fairness Analysis Report for AI-Driven Recruitment System")
#         report.append(f"\nGenerated on: {self.timestamp}\n")
        
#         # Add executive summary
#         report.append("## Executive Summary")
#         report.append(self._generate_executive_summary())
        
#         # Add dataset information
#         report.append("## Dataset Information")
#         report.append(self._generate_dataset_info())
        
#         # Add model comparison
#         report.append("## Model Comparison")
#         report.append(self._generate_model_comparison())
        
#         # Add fairness analysis
#         report.append("## Fairness Analysis")
#         report.append(self._generate_fairness_analysis())
        
#         # Add bias mitigation effectiveness
#         report.append("## Bias Mitigation Effectiveness")