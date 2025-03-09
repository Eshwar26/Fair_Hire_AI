import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

class FairnessVisualizer:
    """
    Class for visualizing fairness metrics and model comparisons.
    """
    
    def __init__(self, results_df=None):
        """
        Initialize the visualizer.
        
        Parameters:
        -----------
        results_df : pandas.DataFrame, optional
            DataFrame containing model comparison results
        """
        self.results_df = results_df
        
        # Default color scheme
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'highlight': '#2ca02c',
            'warning': '#d62728'
        }
        
        # Set Seaborn style
        sns.set_style('whitegrid')
        
    def load_results(self, filepath):
        """
        Load model comparison results from a CSV file.
        
        Parameters:
        -----------
        filepath : str
            Path to the CSV file
        """
        self.results_df = pd.read_csv(filepath)
        
    def fairness_performance_tradeoff(self, save_path=None, show=True):
        """
        Plot the fairness-performance tradeoff.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure
        show : bool, default=True
            Whether to display the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if self.results_df is None:
            raise ValueError("No results data available. Load results first.")
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract metrics
        accuracies = self.results_df['Accuracy'].values
        fairness_diffs = np.abs(self.results_df['Demographic_Parity_Diff'].values)
        model_names = self.results_df['Model'].values
        
        # Create scatter plot
        scatter = ax.scatter(accuracies, fairness_diffs, 
                           s=100, 
                           c=range(len(model_names)), 
                           cmap='viridis',
                           alpha=0.8)
        
        # Add model name annotations
        for i, name in enumerate(model_names):
            # Shorten name for display if too long
            display_name = name.split(' with ')[0] if ' with ' in name else name
            ax.annotate(display_name, 
                       (accuracies[i], fairness_diffs[i]),
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=9)
        
        # Add ideal point marker (high accuracy, low unfairness)
        ax.scatter([1.0], [0.0], marker='*', s=200, color='red', label='Ideal Point')
        
        # Add arrows to show direction of improvement
        ax.arrow(0.7, 0.2, 0.1, 0, width=0.005, head_width=0.02, head_length=0.02, 
                fc=self.colors['highlight'], ec=self.colors['highlight'])
        ax.arrow(0.7, 0.2, 0, -0.1, width=0.005, head_width=0.02, head_length=0.02, 
                fc=self.colors['highlight'], ec=self.colors['highlight'])
        
        # Add labels and title
        ax.set_xlabel('Accuracy', fontsize=12)
        ax.set_ylabel('Unfairness (|Demographic Parity Difference|)', fontsize=12)
        ax.set_title('Performance vs. Fairness Trade-off', fontsize=14, fontweight='bold')
        
        # Set axis limits
        ax.set_xlim([min(accuracies) - 0.05, 1.0])
        ax.set_ylim([0, max(fairness_diffs) * 1.1])
        
        # Add gridlines
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add diagonal line representing equal trade-off
        max_unfairness = max(fairness_diffs)
        ax.plot([0, 1], [max_unfairness, 0], '--', color='gray', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
            
        return fig
    
    def selection_rate_comparison(self, protected_attribute='Gender', save_path=None, show=True):
        """
        Plot selection rates across demographic groups for different models.
        
        Parameters:
        -----------
        protected_attribute : str, default='Gender'
            Protected attribute to compare ('Gender' or 'Ethnicity')
        save_path : str, optional
            Path to save the figure
        show : bool, default=True
            Whether to display the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if self.results_df is None:
            raise ValueError("No results data available. Load results first.")
            
        fig, ax = plt.subplots(figsize=(12, 7))
        
        model_names = self.results_df['Model'].values
        
        # For Gender comparison
        if protected_attribute == 'Gender':
            male_rates = self.results_df['Male_Selection_Rate'].values
            female_rates = self.results_df['Female_Selection_Rate'].values
            
            # Set bar positions
            bar_width = 0.35
            x = np.arange(len(model_names))
            
            # Create grouped bars
            rects1 = ax.bar(x - bar_width/2, male_rates, bar_width, 
                          label='Male', color=self.colors['primary'])
            rects2 = ax.bar(x + bar_width/2, female_rates, bar_width,
                          label='Female', color=self.colors['secondary'])
            
            # Add fairness threshold indicators
            for i, (male, female) in enumerate(zip(male_rates, female_rates)):
                # Calculate disparity
                ratio = min(male, female) / max(male, female) if max(male, female) > 0 else 1
                # 80% rule threshold
                if ratio < 0.8:
                    ax.plot([x[i]-bar_width, x[i]+bar_width], [1.02*max(male, female)]*2, 
                            'r--', marker='o', markersize=5)
                    
            ax.axhline(y=0.8, linestyle='--', color='gray', alpha=0.5, 
                       label='80% Rule Threshold (Approx)')
        
        # Add labels and title
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Selection Rate', fontsize=12)
        ax.set_title(f'Selection Rate by {protected_attribute} Across Models', 
                    fontsize=14, fontweight='bold')
        
        # Set x-axis ticks
        ax.set_xticks(range(len(model_names)))
        # Shorten model names for display
        shortened_names = [name.split(' with ')[0] if ' with ' in name else name 
                           for name in model_names]
        ax.set_xticklabels(shortened_names, rotation=45, ha='right')
        
        # Add legend
        ax.legend()
        
        # Add value annotations on bars
        for i, v in enumerate(male_rates):
            ax.text(i - bar_width/2, v + 0.02, f'{v:.2f}', 
                   ha='center', va='bottom', fontsize=9)
        
        for i, v in enumerate(female_rates):
            ax.text(i + bar_width/2, v + 0.02, f'{v:.2f}', 
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
            
        return fig
    
    def feature_importance_heatmap(self, feature_importance_data, save_path=None, show=True):
        """
        Plot feature importance heatmap.
        
        Parameters:
        -----------
        feature_importance_data : dict
            Dictionary mapping model names to feature importance DataFrames
        save_path : str, optional
            Path to save the figure
        show : bool, default=True
            Whether to display the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if not feature_importance_data:
            raise ValueError("No feature importance data provided.")
            
        # Collect all features and their importances
        all_features = set()
        for model_name, importance_df in feature_importance_data.items():
            all_features.update(importance_df['Feature'].values)
            
        all_features = sorted(list(all_features))
        
        # Create a matrix of feature importances
        importance_matrix = np.zeros((len(feature_importance_data), len(all_features)))
        model_names = list(feature_importance_data.keys())
        
        for i, model_name in enumerate(model_names):
            importance_df = feature_importance_data[model_name]
            for j, feature in enumerate(all_features):
                feature_row = importance_df[importance_df['Feature'] == feature]
                if len(feature_row) > 0:
                    # Get the importance value (either 'Importance' or 'Coefficient')
                    if 'Importance' in feature_row.columns:
                        importance_matrix[i, j] = feature_row['Importance'].values[0]
                    else:
                        importance_matrix[i, j] = abs(feature_row['Coefficient'].values[0])
        
        # Create custom colormap (white to blue)
        colors = [(1, 1, 1), (0.1, 0.3, 0.7)]
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, len(model_names) * 0.5 + 2))
        
        # Normalize the importance values for better visualization
        normalized_matrix = importance_matrix.copy()
        for i in range(len(model_names)):
            row_max = normalized_matrix[i].max()
            if row_max > 0:
                normalized_matrix[i] = normalized_matrix[i] / row_max
        
        # Create heatmap
        sns.heatmap(normalized_matrix, annot=False, cmap=cmap, 
                   xticklabels=all_features, yticklabels=model_names, ax=ax)
        
        # Add labels and title
        ax.set_title('Normalized Feature Importance Across Models', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Model')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
            
        return fig
    
    def bias_mitigation_comparison(self, save_path=None, show=True):
        """
        Compare the effectiveness of different bias mitigation techniques.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure
        show : bool, default=True
            Whether to display the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if self.results_df is None:
            raise ValueError("No results data available. Load results first.")
            
        # Extract bias mitigation technique from model name
        self.results_df['Technique'] = self.results_df['Model'].apply(
            lambda x: x.split(' with ')[-1] if ' with ' in x else 'Baseline'
        )
        
        # Group by technique
        technique_groups = self.results_df.groupby('Technique')
        
        # Compute average metrics by technique
        avg_metrics = technique_groups.agg({
            'Accuracy': 'mean',
            'F1': 'mean',
            'Demographic_Parity_Diff': lambda x: np.mean(np.abs(x))
        }).reset_index()
        
        # Sort by fairness
        avg_metrics = avg_metrics.sort_values('Demographic_Parity_Diff')
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot accuracy
        axes[0].bar(avg_metrics['Technique'], avg_metrics['Accuracy'], color=self.colors['primary'])
        axes[0].set_title('Average Accuracy by Technique')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_ylim([0.7, 1.0])  # Adjust as needed
        
        # Plot F1 score
        axes[1].bar(avg_metrics['Technique'], avg_metrics['F1'], color=self.colors['secondary'])
        axes[1].set_title('Average F1 Score by Technique')
        axes[1].set_ylabel('F1 Score')
        axes[1].set_ylim([0.7, 1.0])  # Adjust as needed
        
        # Plot fairness (lower is better)
        axes[2].bar(avg_metrics['Technique'], avg_metrics['Demographic_Parity_Diff'], 
                  color=self.colors['warning'])
        axes[2].set_title('Average Unfairness by Technique')
        axes[2].set_ylabel('|Demographic Parity Difference|')
        axes[2].set_ylim([0, 0.3])  # Adjust as needed
        
        # Rotate x-axis labels
        for ax in axes:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
            
        return fig
        
    def intersectional_analysis(self, intersectional_data, save_path=None, show=True):
        """
        Visualize intersectional fairness analysis.
        
        Parameters:
        -----------
        intersectional_data : pandas.DataFrame
            DataFrame with columns for gender, ethnicity, and selection rates
        save_path : str, optional
            Path to save the figure
        show : bool, default=True
            Whether to display the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if intersectional_data is None:
            raise ValueError("No intersectional data provided.")
            
        # Pivot the data for heatmap
        pivot_data = intersectional_data.pivot(
            index='Gender', 
            columns='Ethnicity', 
            values='Selection Rate'
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax)
        
        # Add labels and title
        ax.set_title('Selection Rate by Gender and Ethnicity (Intersectional Analysis)', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
            
        return fig
    
    def before_after_bias_mitigation(self, before_data, after_data, protected_attribute='Gender', save_path=None, show=True):
        """
        Compare selection rates before and after bias mitigation.
        
        Parameters:
        -----------
        before_data : pandas.DataFrame
            DataFrame with selection rates before mitigation
        after_data : pandas.DataFrame
            DataFrame with selection rates after mitigation
        protected_attribute : str, default='Gender'
            Protected attribute to analyze
        save_path : str, optional
            Path to save the figure
        show : bool, default=True
            Whether to display the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if before_data is None or after_data is None:
            raise ValueError("Both before and after data must be provided.")
            
        # Merge datasets
        merged_data = pd.DataFrame({
            'Group': before_data['Group'],
            'Before': before_data['Selection Rate'],
            'After': after_data['Selection Rate']
        })
        
        # Calculate improvement
        merged_data['Improvement'] = merged_data['After'] - merged_data['Before']
        
        # Reshape for plotting
        plot_data = pd.melt(merged_data, 
                            id_vars=['Group', 'Improvement'],
                            value_vars=['Before', 'After'],
                            var_name='Stage', 
                            value_name='Selection Rate')
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot before/after comparison
        sns.barplot(x='Group', y='Selection Rate', hue='Stage', data=plot_data, ax=ax1)
        ax1.set_title(f'Selection Rate by {protected_attribute} Before and After Bias Mitigation', 
                     fontsize=12, fontweight='bold')
        ax1.set_ylabel('Selection Rate')
        ax1.set_ylim([0, 1])
        
        # Add bar annotations
        for i, row in merged_data.iterrows():
            ax1.text(i-0.2, row['Before']+0.02, f"{row['Before']:.2f}", 
                    ha='center', va='bottom', fontsize=9)
            ax1.text(i+0.2, row['After']+0.02, f"{row['After']:.2f}", 
                    ha='center', va='bottom', fontsize=9)
        
        # Plot improvement
        colors = ['red' if x < 0 else 'green' for x in merged_data['Improvement']]
        sns.barplot(x='Group', y='Improvement', data=merged_data, ax=ax2, palette=colors)
        ax2.set_title(f'Change in Selection Rate After Bias Mitigation', 
                     fontsize=12, fontweight='bold')
        ax2.set_ylabel('Change in Selection Rate')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add bar annotations
        for i, improvement in enumerate(merged_data['Improvement']):
            ax2.text(i, improvement + (0.01 if improvement >= 0 else -0.03),
                    f"{improvement:+.2f}", ha='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
            
        return fig
    
    def confusion_matrix_comparison(self, matrices, group_names, model_name, save_path=None, show=True):
        """
        Compare confusion matrices across different demographic groups.
        
        Parameters:
        -----------
        matrices : list
            List of confusion matrices (2D numpy arrays)
        group_names : list
            List of group names corresponding to each matrix
        model_name : str
            Name of the model
        save_path : str, optional
            Path to save the figure
        show : bool, default=True
            Whether to display the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if not matrices or not group_names:
            raise ValueError("Matrices and group names must be provided.")
            
        n_groups = len(matrices)
        
        # Create figure
        fig, axes = plt.subplots(1, n_groups, figsize=(4*n_groups, 4))
        
        # If only one group, wrap axes in a list
        if n_groups == 1:
            axes = [axes]
        
        # Plot each confusion matrix
        for i, (matrix, group) in enumerate(zip(matrices, group_names)):
            # Normalize matrix
            matrix_normalized = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
            
            # Create heatmap
            sns.heatmap(matrix_normalized, annot=matrix, fmt='d', cmap='Blues', 
                       ax=axes[i], cbar=False, square=True)
            
            # Add labels
            axes[i].set_xlabel('Predicted label')
            axes[i].set_ylabel('True label')
            axes[i].set_title(f'{group}')
            axes[i].set_xticklabels(['Not Qualified', 'Qualified'])
            axes[i].set_yticklabels(['Not Qualified', 'Qualified'])
        
        # Add overall title
        plt.suptitle(f'Confusion Matrices by Group for {model_name}', 
                    fontsize=14, fontweight='bold', y=1.05)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
            
        return fig
    
    def model_fairness_radar(self, model_metrics, save_path=None, show=True):
        """
        Create a radar chart to compare multiple fairness metrics across models.
        
        Parameters:
        -----------
        model_metrics : dict
            Dictionary mapping model names to dictionaries of metrics
        save_path : str, optional
            Path to save the figure
        show : bool, default=True
            Whether to display the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if not model_metrics:
            raise ValueError("Model metrics must be provided.")
            
        # Extract metrics and model names
        metrics = list(next(iter(model_metrics.values())).keys())
        models = list(model_metrics.keys())
        
        # Number of variables
        N = len(metrics)
        
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Angle for each metric
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the polygon
        
        # Plot each model
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        
        for i, model in enumerate(models):
            # Get values for this model
            values = [model_metrics[model][metric] for metric in metrics]
            values += values[:1]  # Close the polygon
            
            # Plot values
            ax.plot(angles, values, linewidth=2, linestyle='solid', 
                   color=colors[i], label=model)
            ax.fill(angles, values, color=colors[i], alpha=0.1)
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Set labels
        plt.xticks(angles[:-1], metrics)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title('Model Comparison Across Fairness Metrics', 
                 fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
            
        return fig
    
    def create_fairness_dashboard(self, data, feature_importance_data=None, save_dir=None, show=True):
        """
        Create a comprehensive fairness analysis dashboard.
        
        Parameters:
        -----------
        data : dict
            Dictionary containing all necessary data for visualizations
        feature_importance_data : dict, optional
            Dictionary mapping model names to feature importance DataFrames
        save_dir : str, optional
            Directory to save the figures
        show : bool, default=True
            Whether to display the plots
            
        Returns:
        --------
        dict
            Dictionary of figure objects
        """
        figures = {}
        
        # 1. Fairness-Performance Tradeoff
        figures['tradeoff'] = self.fairness_performance_tradeoff(
            save_path=f"{save_dir}/fairness_tradeoff.png" if save_dir else None,
            show=show
        )
        
        # 2. Selection Rate Comparison by Gender
        figures['gender_selection'] = self.selection_rate_comparison(
            protected_attribute='Gender',
            save_path=f"{save_dir}/gender_selection.png" if save_dir else None,
            show=show
        )
        
        # 3. Bias Mitigation Comparison
        figures['bias_mitigation'] = self.bias_mitigation_comparison(
            save_path=f"{save_dir}/bias_mitigation.png" if save_dir else None,
            show=show
        )
        
        # 4. Feature Importance Heatmap (if data provided)
        if feature_importance_data:
            figures['feature_importance'] = self.feature_importance_heatmap(
                feature_importance_data,
                save_path=f"{save_dir}/feature_importance.png" if save_dir else None,
                show=show
            )
        
        # 5. Intersectional Analysis (if data provided)
        if 'intersectional_data' in data:
            figures['intersectional'] = self.intersectional_analysis(
                data['intersectional_data'],
                save_path=f"{save_dir}/intersectional.png" if save_dir else None,
                show=show
            )
        
        # 6. Before-After Comparison (if data provided)
        if 'before_data' in data and 'after_data' in data:
            figures['before_after'] = self.before_after_bias_mitigation(
                data['before_data'],
                data['after_data'],
                save_path=f"{save_dir}/before_after.png" if save_dir else None,
                show=show
            )
        
        return figures