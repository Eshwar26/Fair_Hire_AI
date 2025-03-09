import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# For bias mitigation
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV

# Define the FairRecruitmentModel class
class FairRecruitmentModel:
    def __init__(self, name, model_class, params=None, bias_mitigation=None):
        """
        Initialize a recruitment model with fairness evaluation.
        
        Parameters:
        -----------
        name : str
            Name of the model
        model_class : class
            Sklearn estimator class
        params : dict, optional
            Parameters for the model
        bias_mitigation : str, optional
            Bias mitigation technique to apply
            Options: None, 'reweighing', 'sampling', 'threshold_optimization'
        """
        self.name = name
        self.model_class = model_class
        self.params = params or {}
        self.bias_mitigation = bias_mitigation
        self.model = None
        self.fairness_metrics = {}
    
    
    
    def fit(self, X_train, y_train, protected_train=None):
        """
        Train the model with optional bias mitigation.
    
        Parameters:
        -----------
        X_train : pandas.DataFrame
            Training features
        y_train : array-like
            Training labels
        protected_train : pandas.DataFrame, optional
            Protected attributes for training data
        """
        print(f"Starting training for {self.name}...")
    
    # Print dimensions for debugging
        print(f"X_train shape: {X_train.shape}, y_train length: {len(y_train)}")
        if protected_train is not None:
            print(f"protected_train shape: {protected_train.shape}")
    
    # Reset indices to ensure alignment
        X_train_modified = X_train.copy().reset_index(drop=True)
        y_train_modified = y_train.copy().reset_index(drop=True)
    
    # Store for later feature importance
        self.X_train = X_train_modified
    
    # Fill missing values with the median (middle value)
        X_train_modified = X_train_modified.fillna(X_train_modified.median())
    
    # Make sure protected attributes are aligned if provided
        if protected_train is not None:
            protected_train_reset = protected_train.copy().reset_index(drop=True)
        
        # Ensure all have the same length
            min_len = min(len(X_train_modified), len(y_train_modified), len(protected_train_reset))
            if not (len(X_train_modified) == len(y_train_modified) == len(protected_train_reset)):
                print(f"WARNING: Dimensions mismatched. Truncating to {min_len} samples.")
                X_train_modified = X_train_modified.iloc[:min_len]
                y_train_modified = y_train_modified.iloc[:min_len]
                protected_train_reset = protected_train_reset.iloc[:min_len]
    
    # Apply bias mitigation techniques
        if self.bias_mitigation is None:
        # Standard training without bias mitigation
            self.model = self.model_class(**self.params)
            self.model.fit(X_train_modified, y_train_modified)
            print("Standard training completed successfully")
        
        elif self.bias_mitigation == 'reweighing' and protected_train is not None:
        # Compute sample weights to balance the dataset
            sample_weights = self._compute_reweighing_weights(y_train_modified, protected_train_reset)
        
        # Initialize and train the model with sample weights
            self.model = self.model_class(**self.params)
            self.model.fit(X_train_modified, y_train_modified, sample_weight=sample_weights)
            print("Reweighing training completed successfully")
            
        elif self.bias_mitigation == 'sampling' and protected_train is not None:
            try:
            # Apply SMOTE for minority class oversampling
                smote = SMOTE(random_state=42)
            
            # Need to encode categorical variables before applying SMOTE
                X_with_protected = X_train_modified.copy()
            
            # Create one-hot encoded versions of protected attributes
                for col in protected_train_reset.columns:
                # Get dummies for categorical values
                    dummies = pd.get_dummies(protected_train_reset[col], prefix=col)
                # Add to feature matrix
                    X_with_protected = pd.concat([X_with_protected, dummies], axis=1)
            
            # Fill any missing values
                X_with_protected = X_with_protected.fillna(X_with_protected.median())
            
                print(f"Running SMOTE on data with shape: {X_with_protected.shape}")
            # Apply SMOTE
                X_resampled, y_resampled = smote.fit_resample(X_with_protected, y_train_modified)
                print(f"SMOTE completed. Resampled shape: {X_resampled.shape}")
            
            # Drop the one-hot encoded protected attributes for model training
            # Get the original column names
                original_columns = X_train_modified.columns
                X_resampled = X_resampled[original_columns]
            
            # Train model on resampled data
                self.model = self.model_class(**self.params)
                self.model.fit(X_resampled, y_resampled)
                print("SMOTE-based training completed successfully")
            except Exception as e:
                print(f"SMOTE error: {str(e)}")
                # Fall back to standard training if SMOTE fails
                print("Falling back to standard training...")
                self.model = self.model_class(**self.params)
                self.model.fit(X_train_modified, y_train_modified)
      
         
    # def fit(self, X_train, y_train, protected_train=None):
    #     """
    #     Train the model with optional bias mitigation.
        
    #     Parameters:
    #     -----------
    #     X_train : pandas.DataFrame
    #         Training features
    #     y_train : array-like
    #         Training labels
    #     protected_train : pandas.DataFrame, optional
    #         Protected attributes for training data
    #     """
    #     print(f"Starting training for {self.name}...")
        
    #     X_train_modified = X_train.copy()
    #     # Fill missing values with the median (middle value)
    #     X_train_modified = X_train_modified.fillna(X_train_modified.median())
    #     y_train_modified = y_train.copy()
        
    #     # Apply bias mitigation techniques
    #     if self.bias_mitigation == 'reweighing' and protected_train is not None:
    #         # Compute sample weights to balance the dataset
    #         sample_weights = self._compute_reweighing_weights(y_train, protected_train)
            
    #         # Initialize and train the model with sample weights
    #         self.model = self.model_class(**self.params)
    #         self.model.fit(X_train_modified, y_train_modified, sample_weight=sample_weights)
            
    #     elif self.bias_mitigation == 'sampling' and protected_train is not None:
    #         # Apply SMOTE for minority class oversampling
    #         smote = SMOTE(random_state=42)
            
    #         # Need to encode categorical variables before applying SMOTE
    #         X_with_protected = X_train_modified.copy()
            
    #         # Create one-hot encoded versions of protected attributes
    #         for col in protected_train.columns:
    #             # Get dummies for categorical values
    #             dummies = pd.get_dummies(protected_train[col], prefix=col)
    #             # Add to feature matrix
    #             X_with_protected = pd.concat([X_with_protected, dummies], axis=1)
            
    #         # Fill any missing values
    #         X_with_protected = X_with_protected.fillna(X_with_protected.median())
            
    #         # Apply SMOTE
    #         X_resampled, y_resampled = smote.fit_resample(X_with_protected, y_train)
            
    #         # Drop the one-hot encoded protected attributes for model training
    #         # Get the original column names
    #         original_columns = X_train_modified.columns
    #         X_resampled = X_resampled[original_columns]
            
    #         # Train model on resampled data
    #         self.model = self.model_class(**self.params)
    #         self.model.fit(X_resampled, y_resampled)

        
    # def fit(self, X_train, y_train, protected_train=None):
    #     """
    #     Train the model with optional bias mitigation.
        
    #     Parameters:
    #     -----------
    #     X_train : pandas.DataFrame
    #         Training features
    #     y_train : array-like
    #         Training labels
    #     protected_train : pandas.DataFrame, optional
    #         Protected attributes for training data
    #     """
    #     X_train_modified = X_train.copy()
    #     # Fill missing values with the median (middle value)
    #     X_train_modified = X_train_modified.fillna(X_train_modified.median())
    #     y_train_modified = y_train.copy()
        
    #     # Apply bias mitigation techniques
    #     if self.bias_mitigation == 'reweighing' and protected_train is not None:
    #         # Compute sample weights to balance the dataset
    #         sample_weights = self._compute_reweighing_weights(y_train, protected_train)
            
    #         # Initialize and train the model with sample weights
    #         self.model = self.model_class(**self.params)
    #         self.model.fit(X_train_modified, y_train_modified, sample_weight=sample_weights)
            
    #     elif self.bias_mitigation == 'sampling' and protected_train is not None:
    # # Apply SMOTE for minority class oversampling
    #         smote = SMOTE(random_state=42)
    
    # # Need to encode categorical variables before applying SMOTE
    #         X_with_protected = X_train_modified.copy()
    
    # # Create one-hot encoded versions of protected attributes
    #         for col in protected_train.columns:
    #     # Get dummies for categorical values
    #             dummies = pd.get_dummies(protected_train[col], prefix=col)
    #     # Add to feature matrix
    #             X_with_protected = pd.concat([X_with_protected, dummies], axis=1)
    
    # # Fill any missing values
    #         X_with_protected = X_with_protected.fillna(X_with_protected.median())
    
    # # Apply SMOTE
    #         X_resampled, y_resampled = smote.fit_resample(X_with_protected, y_train)
    
    # # Drop the one-hot encoded protected attributes for model training
    # # Get the original column names
    #         original_columns = X_train_modified.columns
    #         X_resampled = X_resampled[original_columns]
    
    # # Train model on resampled data
    #         self.model = self.model_class(**self.params)
    #         self.model.fit(X_resampled, y_resampled)
    
    
    
    
    
    def _compute_reweighing_weights(self, y, protected_attributes):
        """
        Compute sample weights to mitigate bias through reweighing.
        """
        # Ensure indices are aligned by converting to numpy arrays
        y_array = np.array(y)
        n_samples = len(y_array)

        # Initialize weights
        weights = np.ones(n_samples)

        # Reset index on protected attributes to ensure alignment
        protected_attrs = protected_attributes.copy().reset_index(drop=True)

        # For simplicity, consider only gender and ethnicity
        for gender in protected_attrs['gender'].unique():
            for ethnicity in protected_attrs['ethnicity'].unique():
                for outcome in [0, 1]:  # For both positive and negative outcomes
                    # Create mask for this group and outcome
                    gender_mask = np.array(protected_attrs['gender'] == gender)
                    ethnicity_mask = np.array(protected_attrs['ethnicity'] == ethnicity)
                    outcome_mask = (y_array == outcome)

                    # Combine masks ensuring they're all the same length
                    if len(gender_mask) == len(ethnicity_mask) == len(outcome_mask):
                        mask = gender_mask & ethnicity_mask & outcome_mask

                        # Skip if no samples in this group
                        if sum(mask) == 0:
                            continue
                        
                        # Expected probability if unbiased
                        expected_prob = sum(outcome_mask) / n_samples

                        # Observed probability in this group
                        observed_prob = sum(mask) / n_samples

                        # Weight adjustment for this group
                        if observed_prob > 0:
                            weights[mask] = expected_prob / observed_prob

        return weights
    
    
    
    
    
    
    
    
    
    
    
    # def _compute_reweighing_weights(self, y, protected_attributes):
    #     """
    #     Compute sample weights to mitigate bias through reweighing.
        
    #     Parameters:
    #     -----------
    #     y : array-like
    #         Target labels
    #     protected_attributes : pandas.DataFrame
    #         Protected attributes
            
    #     Returns:
    #     --------
    #     array-like
    #         Sample weights
    #     """
    #     # Calculate group frequencies
    #     n_samples = len(y)
        
    #     # Initialize weights
    #     weights = np.ones(n_samples)
        
    #     # For simplicity, consider only gender and ethnicity
    #     for gender in protected_attributes['gender'].unique():
    #         for ethnicity in protected_attributes['ethnicity'].unique():
    #             for outcome in [0, 1]:  # For both positive and negative outcomes
    #                 # Create mask for this group and outcome
    #                 mask = ((protected_attributes['gender'] == gender) & 
    #                         (protected_attributes['ethnicity'] == ethnicity) & 
    #                         (y == outcome))
                    
    #                 # Skip if no samples in this group
    #                 if sum(mask) == 0:
    #                     continue
                    
    #                 # Expected probability if unbiased
    #                 expected_prob = sum(y == outcome) / n_samples
                    
    #                 # Observed probability in this group
    #                 observed_prob = sum(mask) / n_samples
                    
    #                 # Weight adjustment for this group
    #                 if observed_prob > 0:
    #                     weights[mask] = expected_prob / observed_prob
        
    #     return weights
    
    
    def predict(self, X_test):
        """
        Make predictions using the trained model.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
    
    # Handle missing values
        X_test_modified = X_test.copy()
        X_test_modified = X_test_modified.fillna(X_test_modified.median())
        return self.model.predict(X_test_modified)

    
    
    
    # def predict(self, X_test):
    #     """
    #     Make predictions using the trained model.
        
    #     Parameters:
    #     -----------
    #     X_test : pandas.DataFrame
    #         Test features
            
    #     Returns:
    #     --------
    #     array-like
    #         Predicted labels
    #     """
    #     if self.model is None:
    #         raise ValueError("Model not trained. Call fit() first.")
            
    #     X_test_modified = X_test.copy()
    #     X_test_modified = X_test_modified.fillna(X_test_modified.median())    
            
    #     return self.model.predict(X_test)
    
    # def predict_proba(self, X_test):
    #     """
    #     Make probability predictions using the trained model.
        
    #     Parameters:
    #     -----------
    #     X_test : pandas.DataFrame
    #         Test features
            
    #     Returns:
    #     --------
    #     array-like
    #         Predicted probabilities
    #     """
    #     if self.model is None:
    #         raise ValueError("Model not trained. Call fit() first.")
        
    #     X_test_modified = X_test.copy()
    #     X_test_modified = X_test_modified.fillna(X_test_modified.median())
            
    #     if hasattr(self.model, 'predict_proba'):
    #         return self.model.predict_proba(X_test)
    #     else:
    #         raise AttributeError("Model does not support probability predictions.")
    def predict_proba(self, X_test):
        """
        Make probability predictions using the trained model.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
    
    # Handle missing values
        X_test_modified = X_test.copy()
        X_test_modified = X_test_modified.fillna(X_test_modified.median())
    
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_test_modified)
        else:
            raise AttributeError("Model does not support probability predictions.")

    
    
    
    
    
    
    
    def threshold_optimization(self, X_val, y_val, protected_val):
        """
        Optimize decision thresholds to improve fairness.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
    # Ensure model supports probability predictions
        if not hasattr(self.model, 'predict_proba'):
            raise AttributeError("Model does not support probability predictions.")
        
    # Handle missing values
        X_val_modified = X_val.copy()
        X_val_modified = X_val_modified.fillna(X_val_modified.median())
    
    # Convert to numpy arrays to avoid index alignment issues
        probas = self.predict_proba(X_val_modified)[:,1]  # Positive class probabilities
        y_val_array = np.array(y_val)
    
        thresholds = {}
    
    # For simplicity, optimize thresholds for gender groups only
        for gender in protected_val['gender'].unique():
        # Create mask and convert to numpy array too
            gender_mask = np.array(protected_val['gender'] == gender)
        
            best_threshold = 0.5  # Default threshold
            best_f1 = 0.0
        
        # Try different thresholds
            for threshold in np.arange(0.1, 0.9, 0.05):
                preds = (probas[gender_mask] >= threshold).astype(int)
                if len(preds) > 0 and len(y_val_array[gender_mask]) > 0:
                    f1 = f1_score(y_val_array[gender_mask], preds)
                
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
        
            thresholds[gender] = best_threshold
        
        return thresholds
    
    # def threshold_optimization(self, X_val, y_val, protected_val):
    #     """
    #     Optimize decision thresholds to improve fairness.
        
    #     Parameters:
    #     -----------
    #     X_val : pandas.DataFrame
    #         Validation features
    #     y_val : array-like
    #         Validation labels
    #     protected_val : pandas.DataFrame
    #         Protected attributes for validation data
            
    #     Returns:
    #     --------
    #     dict
    #         Optimized thresholds for each group
    #     """
    #     if self.model is None:
    #         raise ValueError("Model not trained. Call fit() first.")
            
    #     # Ensure model supports probability predictions
    #     if not hasattr(self.model, 'predict_proba'):
    #         raise AttributeError("Model does not support probability predictions.")
            
    #     thresholds = {}
    #     probas = self.predict_proba(X_val)[:,1]  # Positive class probabilities
        
    #     # For simplicity, optimize thresholds for gender groups only
    #     for gender in protected_val['gender'].unique():
    #         gender_mask = protected_val['gender'] == gender
            
    #         best_threshold = 0.5  # Default threshold
    #         best_f1 = 0.0
            
    #         # Try different thresholds
    #         for threshold in np.arange(0.1, 0.9, 0.05):
    #             preds = (probas[gender_mask] >= threshold).astype(int)
    #             f1 = f1_score(y_val[gender_mask], preds)
                
    #             if f1 > best_f1:
    #                 best_f1 = f1
    #                 best_threshold = threshold
            
    #         thresholds[gender] = best_threshold
            
    #     return thresholds
    
    # def evaluate(self, X_test, y_test, protected_test=None, optimized_thresholds=None):
    #     """
    #     Evaluate the model's performance and fairness.
        
    #     Parameters:
    #     -----------
    #     X_test : pandas.DataFrame
    #         Test features
    #     y_test : array-like
    #         Test labels
    #     protected_test : pandas.DataFrame, optional
    #         Protected attributes for test data
    #     optimized_thresholds : dict, optional
    #         Group-specific decision thresholds
            
    #     Returns:
    #     --------
    #     dict
    #         Evaluation metrics
    #     """
    #     if self.model is None:
    #         raise ValueError("Model not trained. Call fit() first.")
            
    #     # Make predictions
    #     if optimized_thresholds and hasattr(self.model, 'predict_proba'):
    #         # Get probabilities
    #         probas = self.predict_proba(X_test)[:,1]
            
    #         # Apply group-specific thresholds
    #         preds = np.zeros_like(y_test)
    #         for gender, threshold in optimized_thresholds.items():
    #             gender_mask = protected_test['gender'] == gender
    #             preds[gender_mask] = (probas[gender_mask] >= threshold).astype(int)
    #     else:
    #         preds = self.predict(X_test)
            
    #     # Standard performance metrics
    #     performance = {
    #         'accuracy': accuracy_score(y_test, preds),
    #         'precision': precision_score(y_test, preds),
    #         'recall': recall_score(y_test, preds),
    #         'f1': f1_score(y_test, preds)
    #     }
        
    #     # Fairness metrics
    #     fairness = {}
        
    #     if protected_test is not None:
    #         # Gender fairness
    #         gender_metrics = compute_fairness_metrics(y_test, preds, protected_test, 'gender')
    #         fairness['gender'] = gender_metrics
            
    #         # Calculate demographic parity difference between male and female
    #         if 'Male' in gender_metrics['Group'].values and 'Female' in gender_metrics['Group'].values:
    #             male_rate = gender_metrics.loc[gender_metrics['Group'] == 'Male', 'Selection Rate'].iloc[0]
    #             female_rate = gender_metrics.loc[gender_metrics['Group'] == 'Female', 'Selection Rate'].iloc[0]
    #             fairness['demographic_parity_diff'] = male_rate - female_rate
            
    #         # Ethnicity fairness
    #         ethnicity_metrics = compute_fairness_metrics(y_test, preds, protected_test, 'ethnicity')
    #         fairness['ethnicity'] = ethnicity_metrics
            
    #         # Group-specific disparities for different combinations
    #         for gender in protected_test['gender'].unique():
    #             for ethnicity in protected_test['ethnicity'].unique():
    #                 # Create intersectional group mask
    #                 group_mask = ((protected_test['gender'] == gender) & 
    #                               (protected_test['ethnicity'] == ethnicity))
                    
    #                 # Skip small groups
    #                 if sum(group_mask) < 50:
    #                     continue
                    
    #                 # Calculate selection rate for this group
    #                 group_selection_rate = preds[group_mask].mean()
    #                 fairness[f'{gender}_{ethnicity}_selection_rate'] = group_selection_rate
        
    #     return {'performance': performance, 'fairness': fairness}
    
    
    def evaluate(self, X_test, y_test, protected_test=None, optimized_thresholds=None):
        """
        Evaluate the model's performance and fairness.

        Parameters:
        -----------
        X_test : pandas.DataFrame
            Test features
        y_test : array-like
            Test labels
        protected_test : pandas.DataFrame, optional
            Protected attributes for test data
        optimized_thresholds : dict, optional
            Group-specific decision thresholds

        Returns:
        --------
        dict
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Handle missing values
        X_test_modified = X_test.copy()
        X_test_modified = X_test_modified.fillna(X_test_modified.median())

        # Convert to numpy arrays to avoid index alignment issues
        y_test_array = np.array(y_test)

        # Make predictions
        if optimized_thresholds and hasattr(self.model, 'predict_proba'):
            # Get probabilities
            probas = self.predict_proba(X_test_modified)[:,1]

            # Apply group-specific thresholds
            preds = np.zeros_like(y_test_array)
            for gender, threshold in optimized_thresholds.items():
                gender_mask = np.array(protected_test['gender'] == gender)
                preds[gender_mask] = (probas[gender_mask] >= threshold).astype(int)
        else:
            preds = self.predict(X_test_modified)

        # Standard performance metrics
        performance = {
            'accuracy': accuracy_score(y_test_array, preds),
            'precision': precision_score(y_test_array, preds),
            'recall': recall_score(y_test_array, preds),
            'f1': f1_score(y_test_array, preds)
        }

        # Fairness metrics
        fairness = {}

        if protected_test is not None:
            # Convert to numpy for compute_fairness_metrics
            protected_test_copy = protected_test.copy().reset_index(drop=True)

            # Gender fairness
            gender_metrics = compute_fairness_metrics(y_test_array, preds, protected_test_copy, 'gender')
            fairness['gender'] = gender_metrics

            # Calculate demographic parity difference between male and female
            if 'Male' in gender_metrics['Group'].values and 'Female' in gender_metrics['Group'].values:
                male_rate = gender_metrics.loc[gender_metrics['Group'] == 'Male', 'Selection Rate'].iloc[0]
                female_rate = gender_metrics.loc[gender_metrics['Group'] == 'Female', 'Selection Rate'].iloc[0]
                fairness['demographic_parity_diff'] = male_rate - female_rate

            # Ethnicity fairness
            ethnicity_metrics = compute_fairness_metrics(y_test_array, preds, protected_test_copy, 'ethnicity')
            fairness['ethnicity'] = ethnicity_metrics

            # Group-specific disparities for different combinations
            for gender in protected_test_copy['gender'].unique():
                for ethnicity in protected_test_copy['ethnicity'].unique():
                    # Create intersectional group mask
                    group_mask = np.array((protected_test_copy['gender'] == gender) & 
                                  (protected_test_copy['ethnicity'] == ethnicity))

                    # Skip small groups
                    if sum(group_mask) < 50:
                        continue
                    
                    # Calculate selection rate for this group
                    group_selection_rate = preds[group_mask].mean()
                    fairness[f'{gender}_{ethnicity}_selection_rate'] = group_selection_rate

        return {'performance': performance, 'fairness': fairness}
    
    
    
    
    def feature_importance(self):
        """
        Get feature importance if the model supports it.
        
        Returns:
        --------
        pandas.DataFrame
            Feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = list(self.X_train.columns) if hasattr(self, 'X_train') else None
            if not feature_names:
                # If no column names stored, use numerical indices
                feature_names = [f"Feature_{i}" for i in range(len(importances))]
            return pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)
        elif hasattr(self.model, 'coef_'):
            coefs = self.model.coef_[0]
            feature_names = list(self.X_train.columns) if hasattr(self, 'X_train') else None
            if not feature_names:
                # If no column names stored, use numerical indices
                feature_names = [f"Feature_{i}" for i in range(len(coefs))]
            return pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs}).sort_values('Coefficient', ascending=False)
        else:
            return None

# Define fairness metrics function - these need to be outside the class but not executed on import
def demographic_parity_difference(y_pred, protected_attributes, privileged_groups, unprivileged_groups):
    """
    Calculate the demographic parity difference between privileged and unprivileged groups.
    """
    # Create masks for privileged and unprivileged groups
    privileged_mask = np.ones(len(y_pred), dtype=bool)
    unprivileged_mask = np.ones(len(y_pred), dtype=bool)
    
    for attr, value in privileged_groups.items():
        privileged_mask &= (protected_attributes[attr] == value)
        
    for attr, value in unprivileged_groups.items():
        unprivileged_mask &= (protected_attributes[attr] == value)
    
    # Calculate selection rates
    privileged_selection_rate = y_pred[privileged_mask].mean()
    unprivileged_selection_rate = y_pred[unprivileged_mask].mean()
    
    return privileged_selection_rate - unprivileged_selection_rate

def equal_opportunity_difference(y_true, y_pred, protected_attributes, privileged_groups, unprivileged_groups):
    """
    Calculate the equal opportunity difference between privileged and unprivileged groups.
    Equal opportunity measures the difference in true positive rates.
    """
    # Create masks for privileged and unprivileged groups
    privileged_mask = np.ones(len(y_pred), dtype=bool)
    unprivileged_mask = np.ones(len(y_pred), dtype=bool)
    
    for attr, value in privileged_groups.items():
        privileged_mask &= (protected_attributes[attr] == value)
        
    for attr, value in unprivileged_groups.items():
        unprivileged_mask &= (protected_attributes[attr] == value)
    
    # Calculate true positive rates
    privileged_positive_mask = privileged_mask & (y_true == 1)
    unprivileged_positive_mask = unprivileged_mask & (y_true == 1)
    
    if sum(privileged_positive_mask) == 0 or sum(unprivileged_positive_mask) == 0:
        return np.nan
    
    privileged_tpr = sum(y_pred[privileged_positive_mask]) / sum(privileged_positive_mask)
    unprivileged_tpr = sum(y_pred[unprivileged_positive_mask]) / sum(unprivileged_positive_mask)
    
    return privileged_tpr - unprivileged_tpr

def disparate_impact_ratio(y_pred, protected_attributes, privileged_groups, unprivileged_groups):
    """
    Calculate the disparate impact ratio between unprivileged and privileged groups.
    """
    # Create masks for privileged and unprivileged groups
    privileged_mask = np.ones(len(y_pred), dtype=bool)
    unprivileged_mask = np.ones(len(y_pred), dtype=bool)
    
    for attr, value in privileged_groups.items():
        privileged_mask &= (protected_attributes[attr] == value)
        
    for attr, value in unprivileged_groups.items():
        unprivileged_mask &= (protected_attributes[attr] == value)
    
    # Calculate selection rates
    privileged_selection_rate = y_pred[privileged_mask].mean()
    unprivileged_selection_rate = y_pred[unprivileged_mask].mean()
    
    if privileged_selection_rate == 0:
        return np.nan
    
    return unprivileged_selection_rate / privileged_selection_rate

def compute_fairness_metrics(y_true, y_pred, protected_attributes, attribute_name, positive_label=1):
    """
    Compute various fairness metrics for different demographic groups based on a protected attribute.
    """
    metrics = []
    unique_values = protected_attributes[attribute_name].unique()
    
    for value in unique_values:
        mask = protected_attributes[attribute_name] == value
        
        # Skip if no samples for this group
        if sum(mask) == 0:
            continue
        
        # Calculate metrics
        selection_rate = y_pred[mask].mean()
        
        # True positive rate (recall for positive samples)
        positive_mask = mask & (y_true == positive_label)
        if sum(positive_mask) > 0:
            tpr = sum(y_pred[positive_mask] == positive_label) / sum(positive_mask)
        else:
            tpr = np.nan
            
        # False positive rate
        negative_mask = mask & (y_true != positive_label)
        if sum(negative_mask) > 0:
            fpr = sum(y_pred[negative_mask] == positive_label) / sum(negative_mask)
        else:
            fpr = np.nan
            
        # Precision
        if sum(y_pred[mask] == positive_label) > 0:
            precision = sum((y_pred[mask] == positive_label) & (y_true[mask] == positive_label)) / sum(y_pred[mask] == positive_label)
        else:
            precision = np.nan
            
        # Accuracy
        accuracy = sum(y_pred[mask] == y_true[mask]) / sum(mask)
        
        metrics.append({
            'Group': value,
            'Count': sum(mask),
            'Selection Rate': selection_rate,
            'True Positive Rate': tpr,
            'False Positive Rate': fpr,
            'Precision': precision,
            'Accuracy': accuracy
        })
    
    return pd.DataFrame(metrics)

# Main execution - only run this if the file is executed directly
if __name__ == "__main__":
    # Load the dataset
    try:
        df = pd.read_csv('recruitment_dataset.csv')
        X_train = pd.read_csv('recruitment_train.csv')
        X_test = pd.read_csv('recruitment_test.csv')
        # Extract target from training and test sets
        y_train = X_train.pop('truly_qualified')
        y_test = X_test.pop('truly_qualified')
        print("Successfully loaded the dataset and train/test splits")
    except FileNotFoundError:
        print("Dataset files not found. Please run the data generation script first.")
        import sys
        sys.exit(1)
        
    # Get protected attributes from the original dataset
    train_indices = df['truly_qualified'].isin(y_train)  # This is an approximation
    test_indices = ~train_indices

    # Create separate datasets with protected attributes for fairness analysis
    protected_train = df.loc[train_indices, ['gender', 'ethnicity', 'age']]
    protected_test = df.loc[test_indices, ['gender', 'ethnicity', 'age']]

    # Ensure protected attributes are properly aligned
    assert len(protected_train) == len(X_train), "Protected attributes don't match training data size"
    assert len(protected_test) == len(X_test), "Protected attributes don't match test data size"

    # Create a validation set for threshold optimization
    X_train_part, X_val, y_train_part, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, 
        stratify=y_train
    )

    # Create the corresponding protected attributes sets
    train_part_indices = y_train.index.isin(y_train_part.index)
    protected_train_part = protected_train.iloc[train_part_indices]
    protected_val = protected_train.iloc[~train_part_indices]

    # Define the models to evaluate
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

    for model in models:
        print(f"Training {model.name}...")
        
        # Train the model
        model.fit(X_train_part, y_train_part, protected_train_part)
        
        # Perform threshold optimization if applicable
        if 'Calibrated' in model.name or 'Logistic' in model.name:
            thresholds = model.threshold_optimization(X_val, y_val, protected_val)
            print(f"Optimized thresholds: {thresholds}")
        else:
            thresholds = None
        
        # Evaluate on test set
        eval_results = model.evaluate(X_test, y_test, protected_test, thresholds)
        
        # Store results
        results.append({
            'Model': model.name,
            'Performance': eval_results['performance'],
            'Fairness': eval_results['fairness']
        })
        
        # Print summary
        print(f"\nModel: {model.name}")
        print(f"Accuracy: {eval_results['performance']['accuracy']:.4f}")
        print(f"F1 Score: {eval_results['performance']['f1']:.4f}")
        
        if 'demographic_parity_diff' in eval_results['fairness']:
            print(f"Demographic Parity Difference (Male-Female): {eval_results['fairness']['demographic_parity_diff']:.4f}")
        
        print("\nGender fairness:")
        print(eval_results['fairness']['gender'][['Group', 'Selection Rate', 'True Positive Rate']])
        
        print("\n" + "-"*50 + "\n")

    # Find the best model considering both performance and fairness
    best_model_index = 0
    best_score = 0

    for i, result in enumerate(results):
        # Combine accuracy and fairness (lower demographic parity diff is better)
        perf_score = result['Performance']['f1']
        fairness_score = 1.0 - abs(result['Fairness'].get('demographic_parity_diff', 0))
        combined_score = 0.6 * perf_score + 0.4 * fairness_score
        
        if combined_score > best_score:
            best_score = combined_score
            best_model_index = i

    best_model = models[best_model_index]
    print(f"Best model: {best_model.name}")

    # Feature importance for the best model
    if best_model.feature_importance() is not None:
        print("\nFeature Importance:")
        print(best_model.feature_importance().head(10))

    # Visualization
    plt.figure(figsize=(12, 8))

    # Compare accuracy vs fairness
    accuracies = [r['Performance']['accuracy'] for r in results]
    fairness_diffs = [abs(r['Fairness'].get('demographic_parity_diff', 0)) for r in results]
    model_names = [m.name for m in models]

    plt.scatter(accuracies, fairness_diffs)
    for i, name in enumerate(model_names):
        plt.annotate(name, (accuracies[i], fairness_diffs[i]), fontsize=8)
        
    plt.xlabel('Accuracy')
    plt.ylabel('Unfairness (Absolute Demographic Parity Difference)')
    plt.title('Performance vs. Fairness Trade-off')
    plt.grid(True)
    plt.savefig('fairness_performance_tradeoff.png')

    # Create fairness comparison by gender
    plt.figure(figsize=(14, 6))
    bar_width = 0.15
    index = np.arange(len(model_names))

    # Get selection rates for males and females across models
    male_rates = []
    female_rates = []

    for result in results:
        gender_metrics = result['Fairness']['gender']
        
        male_rate = gender_metrics.loc[gender_metrics['Group'] == 'Male', 'Selection Rate'].iloc[0] if 'Male' in gender_metrics['Group'].values else 0
        female_rate = gender_metrics.loc[gender_metrics['Group'] == 'Female', 'Selection Rate'].iloc[0] if 'Female' in gender_metrics['Group'].values else 0
        
        male_rates.append(male_rate)
        female_rates.append(female_rate)

    plt.bar(index, male_rates, bar_width, label='Male')
    plt.bar(index + bar_width, female_rates, bar_width, label='Female')

    plt.xlabel('Model')
    plt.ylabel('Selection Rate')
    plt.title('Selection Rate by Gender Across Models')
    plt.xticks(index + bar_width/2, [m.name.split(' with ')[0] for m in models], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('gender_selection_rates.png')

    # Save results to CSV
    results_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': [r['Performance']['accuracy'] for r in results],
        'F1': [r['Performance']['f1'] for r in results],
        'Demographic_Parity_Diff': [r['Fairness'].get('demographic_parity_diff', np.nan) for r in results],
        'Male_Selection_Rate': male_rates,
        'Female_Selection_Rate': female_rates
    })

    results_df.to_csv('model_comparison_results.csv', index=False)
    print("\nResults saved to 'model_comparison_results.csv'")

    print("\nAnalysis complete!")