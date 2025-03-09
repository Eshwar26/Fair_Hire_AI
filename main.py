import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
# Replace these with the actual file paths where you've saved the code
import sys
sys.path.append('./src')  # Add source directory to path

# Create output directory
output_dir = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_dir, exist_ok=True)

# Function to check if data exists, otherwise generate it
# def check_or_generate_data():
#     """Check if dataset exists, otherwise generate it"""
#     if not (os.path.exists('recruitment_dataset.csv') and 
#             os.path.exists('recruitment_train.csv') and
#             os.path.exists('recruitment_test.csv')):
#         print("Generating dataset...")
#         # Execute dataset generation script
#         exec(open('dataset_generation.py').read())
#     else:
#         print("Dataset files found, loading existing data.")
# def check_or_generate_data():
#     """Check if dataset exists, otherwise generate it"""
#     if not (os.path.exists('recruitment_dataset.csv') and 
#             os.path.exists('recruitment_train.csv') and
#             os.path.exists('recruitment_test.csv')):
#         print("Generating dataset...")
#         # Use the correct path to dataset_generation.py
#         exec(open('src/dataset_generation.py').read())  # Updated path
#     else:
#         print("Dataset files found, loading existing data.")
        

def check_or_generate_data():
    """Check if dataset exists, otherwise generate it"""
    if not (os.path.exists('recruitment_dataset.csv') and 
            os.path.exists('recruitment_train.csv') and
            os.path.exists('recruitment_test.csv')):
        print("Generating dataset...")
        # Import the function and call it directly
        from src.dataset_generation import generate_recruitment_dataset
        generate_recruitment_dataset(5000)
    else:
        print("Dataset files found, loading existing data.")




def load_data():
    try:
        # Load the datasets
        df = pd.read_csv('recruitment_dataset.csv')
        train_data = pd.read_csv('recruitment_train.csv')
        test_data = pd.read_csv('recruitment_test.csv')
        
        # Extract target from training and test data
        y_train = train_data.pop('truly_qualified')
        X_train = train_data
        
        y_test = test_data.pop('truly_qualified')
        X_test = test_data
        
        # Create protected attribute datasets with the correct size
        protected_attrs = df[['gender', 'ethnicity', 'age']]
        
        # For a clean solution, re-split the protected attributes to match your train/test sets
        dummy_indices = np.arange(len(df))
        train_indices, test_indices = train_test_split(
            dummy_indices, 
            test_size=len(X_test)/len(df),  # Match your existing test size ratio
            random_state=42                 # Use the same random state
        )
        
        protected_train = protected_attrs.iloc[train_indices].reset_index(drop=True)
        protected_test = protected_attrs.iloc[test_indices].reset_index(drop=True)
        
        # Ensure the counts match
        assert len(protected_train) == len(X_train), "Protected attribute count doesn't match train set"
        assert len(protected_test) == len(X_test), "Protected attribute count doesn't match test set"
        
        # Dataset info for reporting
        dataset_info = {
            'n_samples': len(df),
            'n_features': len(X_train.columns),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'gender_distribution': df['gender'].value_counts(normalize=True).to_dict(),
            'ethnicity_distribution': df['ethnicity'].value_counts(normalize=True).to_dict()
        }
        
        print(f"Dataset loaded: {len(df)} total samples")
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        return df, X_train, y_train, X_test, y_test, protected_train, protected_test, dataset_info
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# def load_data():
#     """Load the recruitment dataset and train/test splits"""
#     try:
#         df = pd.read_csv('recruitment_dataset.csv')
#         train_data = pd.read_csv('recruitment_train.csv')
#         test_data = pd.read_csv('recruitment_test.csv')
        
#         # Extract target from training and test data
#         y_train = train_data.pop('truly_qualified')
#         X_train = train_data
        
#         y_test = test_data.pop('truly_qualified')
#         X_test = test_data
        
#         # Get protected attributes from the original dataset
#         protected_train = df.loc[df.index.isin(X_train.index), ['gender', 'ethnicity', 'age']]
#         protected_test = df.loc[df.index.isin(X_test.index), ['gender', 'ethnicity', 'age']]
        
#         # Make sure protected_train and protected_test are aligned with X_train and X_test
#         if len(protected_train) != len(X_train) or len(protected_test) != len(X_test):
#             print("Warning: Protected attributes not aligned with features. Creating new alignment.")
#             # Create a train/test split of the protected attributes
#             dummy_indices = np.arange(len(df))
#             _, test_indices = train_test_split(dummy_indices, test_size=0.2, random_state=42)
            
#             protected_train = df.loc[~df.index.isin(test_indices), ['gender', 'ethnicity', 'age']].reset_index(drop=True)
#             protected_test = df.loc[df.index.isin(test_indices), ['gender', 'ethnicity', 'age']].reset_index(drop=True)
        
#         # Basic dataset info
#         dataset_info = {
#             'n_samples': len(df),
#             'n_features': len(X_train.columns),
#             'train_size': len(X_train),
#             'test_size': len(X_test),
#             'gender_distribution': df['gender'].value_counts(normalize=True).to_dict(),
#             'ethnicity_distribution': df['ethnicity'].value_counts(normalize=True).to_dict()
#         }
        
#         # Print basic dataset info
#         print(f"Dataset loaded: {len(df)} total samples")
#         print(f"Training set: {len(X_train)} samples")
#         print(f"Test set: {len(X_test)} samples")
        
#         return df, X_train, y_train, X_test, y_test, protected_train, protected_test, dataset_info
        
#     except Exception as e:
#         print(f"Error loading data: {e}")
#         return None

def run_fair_recruitment_system():
    """Main function to run the entire fair recruitment system"""
    print("Starting Fair AI Recruitment System")
    print("=" * 80)
    
    # Check or generate dataset
    check_or_generate_data()
    
    # Load data
    data = load_data()
    if data is None:
        print("Error: Could not load data. Exiting.")
        return
    
    df, X_train, y_train, X_test, y_test, protected_train, protected_test, dataset_info = data
    
    # Create validation split for threshold optimization
    X_train_part, X_val, y_train_part, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Create corresponding protected attributes
    protected_train_part = protected_train.iloc[:len(X_train_part)]
    protected_val = protected_train.iloc[len(X_train_part):]
    
    print("\nInitializing models and bias mitigation techniques...")
    
    # Import our FairRecruitmentModel class
    # from FairRecruitmentModel import FairRecruitmentModel
    from src.FairRecruitmentModel import FairRecruitmentModel
    
    # Define models to evaluate
    models = [
        # Baseline models without bias mitigation
        FairRecruitmentModel(
            name="Logistic Regression Baseline",
            model_class=LogisticRegression,
            params={'random_state': 42, 'max_iter': 1000}
        ),
        FairRecruitmentModel(
            name="Random Forest Baseline",
            model_class=RandomForestClassifier,
            params={'random_state': 42, 'n_estimators': 100}
        ),
        
        # Models with bias mitigation techniques
        FairRecruitmentModel(
            name="Logistic Regression with Reweighing",
            model_class=LogisticRegression,
            params={'random_state': 42, 'max_iter': 1000},
            bias_mitigation='reweighing'
        ),
        FairRecruitmentModel(
            name="Random Forest with Reweighing",
            model_class=RandomForestClassifier,
            params={'random_state': 42, 'n_estimators': 100},
            bias_mitigation='reweighing'
        ),
        FairRecruitmentModel(
            name="Logistic Regression with Sampling",
            model_class=LogisticRegression,
            params={'random_state': 42, 'max_iter': 1000},
            bias_mitigation='sampling'
        ),
        FairRecruitmentModel(
            name="Random Forest with Sampling",
            model_class=RandomForestClassifier,
            params={'random_state': 42, 'n_estimators': 100},
            bias_mitigation='sampling'
        ),
        FairRecruitmentModel(
            name="Calibrated Logistic Regression",
            model_class=LogisticRegression,
            params={'random_state': 42, 'max_iter': 1000},
            bias_mitigation='calibration'
        )
    ]
    
    # Train and evaluate each model
    results = []
    feature_importance_data = {}
    
    print("\nTraining and evaluating models:")
    print("-" * 80)
    
    for model in models:
        print(f"Processing {model.name}...")
        
        # Train the model
        model.fit(X_train_part, y_train_part, protected_train_part)
        
        
        if model.model is None:
            print(f"WARNING: Model {model.name} failed to train properly. Skipping evaluation.")
            continue
    
    # Perform threshold optimization if applicable
        if 'Calibrated' in model.name or 'Logistic' in model.name:
            try:
                thresholds = model.threshold_optimization(X_val, y_val, protected_val)
                print(f"Optimized thresholds: {thresholds}")
            except Exception as e:
                print(f"Error during threshold optimization: {e}")
                thresholds = None
        else:
            thresholds = None
        
        # # Perform threshold optimization if applicable
        # if 'Calibrated' in model.name or 'Logistic' in model.name:
        #     thresholds = model.threshold_optimization(X_val, y_val, protected_val)
        #     print(f"Optimized thresholds: {thresholds}")
        # else:
        #     thresholds = None
            
        
        # Evaluate on test set
        eval_results = model.evaluate(X_test, y_test, protected_test, thresholds)
        
        # Store results
        results.append({
            'Model': model.name,
            'Performance': eval_results['performance'],
            'Fairness': eval_results['fairness']
        })
        
        # Get feature importance if available
        importance = model.feature_importance()
        if importance is not None:
            feature_importance_data[model.name] = importance
        
        # Print summary
        print(f"\nModel: {model.name}")
        print(f"Accuracy: {eval_results['performance']['accuracy']:.4f}")
        print(f"F1 Score: {eval_results['performance']['f1']:.4f}")
        
        if 'demographic_parity_diff' in eval_results['fairness']:
            print(f"Demographic Parity Difference (Male-Female): {eval_results['fairness']['demographic_parity_diff']:.4f}")
        
        print("\n" + "-"*50 + "\n")
    
    # Find the best model considering both performance and fairness
    best_model_index = 0
    best_score = 0
    
    for i, result in enumerate(results):
        # Combine accuracy and fairness (lower demographic parity diff is better)
        if 'demographic_parity_diff' in result['Fairness']:
            perf_score = result['Performance']['f1']
            fairness_score = 1.0 - abs(result['Fairness']['demographic_parity_diff'])
            combined_score = 0.6 * perf_score + 0.4 * fairness_score
            
            if combined_score > best_score:
                best_score = combined_score
                best_model_index = i
    
    best_model = models[best_model_index]
    print(f"Best model: {best_model.name}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Import visualization component
    from src.FairnessVisualizer import FairnessVisualizer
    
    # Create results DataFrame for visualizations
    results_df = pd.DataFrame({
        'Model': [r['Model'] for r in results],
        'Accuracy': [r['Performance']['accuracy'] for r in results],
        'F1': [r['Performance']['f1'] for r in results],
        'Demographic_Parity_Diff': [r['Fairness'].get('demographic_parity_diff', np.nan) for r in results]
    })
    
    # Add selection rates by gender
    male_rates = []
    female_rates = []
    
    for result in results:
        if 'gender' in result['Fairness']:
            gender_df = result['Fairness']['gender']
            
            male_rate = gender_df.loc[gender_df['Group'] == 'Male', 'Selection Rate'].iloc[0] if 'Male' in gender_df['Group'].values else np.nan
            female_rate = gender_df.loc[gender_df['Group'] == 'Female', 'Selection Rate'].iloc[0] if 'Female' in gender_df['Group'].values else np.nan
            
            male_rates.append(male_rate)
            female_rates.append(female_rate)
        else:
            male_rates.append(np.nan)
            female_rates.append(np.nan)
    
    results_df['Male_Selection_Rate'] = male_rates
    results_df['Female_Selection_Rate'] = female_rates
    
    # Save results
    results_df.to_csv(f'{output_dir}/model_comparison_results.csv', index=False)
    
    # Create visualizer
    visualizer = FairnessVisualizer(results_df)
    
    # Generate visualizations
    visualizer.fairness_performance_tradeoff(save_path=f'{output_dir}/fairness_tradeoff.png')
    visualizer.selection_rate_comparison(save_path=f'{output_dir}/gender_selection_rates.png')
    visualizer.bias_mitigation_comparison(save_path=f'{output_dir}/bias_mitigation.png')
    
    if feature_importance_data:
        visualizer.feature_importance_heatmap(feature_importance_data, save_path=f'{output_dir}/feature_importance.png')
    
    # Prepare data for intersectional analysis
    # Create example dataset
    intersectional_data = []
    
    # Extract intersectional data from best model
    best_result = results[best_model_index]
    for key, value in best_result['Fairness'].items():
        if '_' in key and '_selection_rate' in key:
            gender, ethnicity = key.replace('_selection_rate', '').split('_', 1)
            intersectional_data.append({
                'Gender': gender,
                'Ethnicity': ethnicity,
                'Selection Rate': value
            })
    
    if intersectional_data:
        intersectional_df = pd.DataFrame(intersectional_data)
        visualizer.intersectional_analysis(intersectional_df, save_path=f'{output_dir}/intersectional_analysis.png')
    
    # Prepare data for before-after comparison
    if len(results) >= 2:
        # Find a baseline model and its corresponding model with bias mitigation
        baseline_model = next((r for r in results if 'Baseline' in r['Model']), None)
        mitigated_model = next((r for r in results if 'with' in r['Model']), None)
        
        if baseline_model and mitigated_model and 'gender' in baseline_model['Fairness'] and 'gender' in mitigated_model['Fairness']:
            visualizer.before_after_bias_mitigation(
                baseline_model['Fairness']['gender'],
                mitigated_model['Fairness']['gender'],
                save_path=f'{output_dir}/before_after_mitigation.png'
            )
    
    # Generate comprehensive report
    print("\nGenerating final report...")
    
    # Import report generator
    from src.FairnessReportGenerator import FairnessReportGenerator
    
    # Create report generator
    report_generator = FairnessReportGenerator(results, dataset_info)
    
    # Generate reports
    md_report_path = report_generator.generate_markdown_report(f'{output_dir}/fairness_report.md')
    html_report_path = report_generator.generate_html_report(f'{output_dir}/fairness_report.html')
    
    print(f"\nProcess complete! Results saved to {output_dir}/")
    print(f"Markdown report: {md_report_path}")
    print(f"HTML report: {html_report_path}")
    print("\nRecommended next steps:")
    print("1. Review the fairness report")
    print("2. Implement the recommended model with appropriate bias mitigation")
    print("3. Set up continuous monitoring for fairness metrics")
    print("4. Establish a feedback loop to incorporate post-hire performance")

if __name__ == "__main__":
    run_fair_recruitment_system()
    