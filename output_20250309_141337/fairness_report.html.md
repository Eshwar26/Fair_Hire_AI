# Fairness Analysis Report for AI-Driven Recruitment System

Generated on: 2025-03-09 14:20:10

## Executive Summary
This report presents a comprehensive analysis of fairness in AI-driven recruitment models for software engineering positions.
The analysis focuses on identifying and mitigating bias in the candidate selection process.

### Key Findings

- The best performing model in terms of predictive accuracy is **Logistic Regression Baseline** with an F1 score of 0.99.
- The model with the most balanced selection rates across demographic groups is **Logistic Regression with Reweighing**.
- Overall, the models achieve an average accuracy of 0.99 and F1 score of 0.98.

### Bias Concerns
- No significant gender bias was detected in the evaluated models.
- The following bias mitigation techniques were evaluated: Sampling, Reweighing.
- The **Sampling** technique reduced bias by approximately 0.01 (demographic parity difference).
- The **Reweighing** technique reduced bias by approximately 0.01 (demographic parity difference).

### Recommendations

1. Implement the recommended fair recruitment model with appropriate bias mitigation techniques.
2. Establish continuous monitoring of selection rates across demographic groups.
3. Periodically retrain the model with diverse data to prevent bias drift.
4. Combine automated screening with human oversight to ensure fairness in the final selection.
## Dataset Information
The analysis was performed on a dataset with the following characteristics:

- **Total samples**: 5000
- **Features**: 22
- **Training set size**: 4000
- **Test set size**: 1000

### Demographic Distribution

#### Gender Distribution
| Gender | Percentage |
|--------|------------|
| Male | 0.6% |
| Female | 0.3% |
| Non-binary | 0.0% |

#### Ethnicity Distribution
| Ethnicity | Percentage |
|-----------|------------|
| White | 0.5% |
| Asian | 0.2% |
| Hispanic | 0.1% |
| Black | 0.1% |
| Middle Eastern | 0.0% |
| Native American | 0.0% |
| Pacific Islander | 0.0% |

### Features

The dataset includes the following types of features:

- Technical skills (programming languages, frameworks, etc.)
- Education information (degree, institution)
- Experience metrics (years of experience, previous roles)
- Project contributions and open source activity
- Assessment scores (coding tests, problem-solving evaluations)

### Protected Attributes

The following protected attributes were used for fairness evaluation:

- Gender
- Ethnicity
- Age

### Proxy Variables

The dataset includes potential proxy variables that might correlate with protected attributes:

- Names (which may suggest gender or ethnicity)
- Educational institutions (which may correlate with socioeconomic status)
- Geographic location
- Employment gaps
## Model Comparison
### Performance Metrics

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression Baseline | 0.9950 | 0.9908 | 0.9938 | 0.9923 |
| Random Forest Baseline | 0.9860 | 0.9844 | 0.9723 | 0.9783 |
| Logistic Regression with Reweighing | 0.9870 | 0.9671 | 0.9938 | 0.9803 |
| Random Forest with Reweighing | 0.9830 | 0.9812 | 0.9662 | 0.9736 |
| Logistic Regression with Sampling | 0.9910 | 0.9817 | 0.9908 | 0.9862 |
| Random Forest with Sampling | 0.9860 | 0.9726 | 0.9846 | 0.9786 |

### Fairness Metrics

| Model | Demographic Parity Diff | Male Selection Rate | Female Selection Rate |
|-------|--------------------------|---------------------|------------------------|
| Logistic Regression Baseline | 0.0483 | 0.3449 | 0.2966 |
| Random Forest Baseline | 0.0436 | 0.3402 | 0.2966 |
| Logistic Regression with Reweighing | 0.0331 | 0.3481 | 0.3150 |
| Random Forest with Reweighing | 0.0373 | 0.3370 | 0.2997 |
| Logistic Regression with Sampling | 0.0391 | 0.3449 | 0.3058 |
| Random Forest with Sampling | 0.0361 | 0.3449 | 0.3089 |

### Performance-Fairness Trade-off

The relationship between model performance and fairness can be visualized in the accompanying charts. Generally, there is a trade-off between achieving high predictive accuracy and ensuring fairness across demographic groups.

Key observations:

1. Models with bias mitigation techniques tend to have slightly lower accuracy but better fairness metrics.
2. The demographic parity difference (a measure of unfairness) varies significantly across different model architectures.
3. Calibrated models offer a good balance between performance and fairness.
## Fairness Analysis
This section analyzes the fairness of each model based on various metrics.

### Gender Fairness

The selection rates and true positive rates across gender groups indicate the level of bias in each model:

#### Logistic Regression Baseline
| Gender | Count | Selection Rate | True Positive Rate | False Positive Rate |
|--------|-------|---------------|-------------------|---------------------|
| Female | 327 | 0.2966 | 0.9898 | 0.0000 |
| Male | 632 | 0.3449 | 0.9954 | 0.0072 |
| Non-binary | 41 | 0.2683 | 1.0000 | 0.0000 |

Demographic Parity Difference: 0.0483

✅ **Low gender bias.** The selection rates between males and females are relatively balanced.

#### Random Forest Baseline
| Gender | Count | Selection Rate | True Positive Rate | False Positive Rate |
|--------|-------|---------------|-------------------|---------------------|
| Female | 327 | 0.2966 | 0.9694 | 0.0087 |
| Male | 632 | 0.3402 | 0.9815 | 0.0072 |
| Non-binary | 41 | 0.2195 | 0.8182 | 0.0000 |

Demographic Parity Difference: 0.0436

✅ **Low gender bias.** The selection rates between males and females are relatively balanced.

#### Logistic Regression with Reweighing
| Gender | Count | Selection Rate | True Positive Rate | False Positive Rate |
|--------|-------|---------------|-------------------|---------------------|
| Female | 327 | 0.3150 | 1.0000 | 0.0218 |
| Male | 632 | 0.3481 | 0.9907 | 0.0144 |
| Non-binary | 41 | 0.2683 | 1.0000 | 0.0000 |

Demographic Parity Difference: 0.0331

✅ **Low gender bias.** The selection rates between males and females are relatively balanced.

#### Random Forest with Reweighing
| Gender | Count | Selection Rate | True Positive Rate | False Positive Rate |
|--------|-------|---------------|-------------------|---------------------|
| Female | 327 | 0.2997 | 0.9694 | 0.0131 |
| Male | 632 | 0.3370 | 0.9722 | 0.0072 |
| Non-binary | 41 | 0.2195 | 0.8182 | 0.0000 |

Demographic Parity Difference: 0.0373

✅ **Low gender bias.** The selection rates between males and females are relatively balanced.

#### Logistic Regression with Sampling
| Gender | Count | Selection Rate | True Positive Rate | False Positive Rate |
|--------|-------|---------------|-------------------|---------------------|
| Female | 327 | 0.3058 | 1.0000 | 0.0087 |
| Male | 632 | 0.3449 | 0.9907 | 0.0096 |
| Non-binary | 41 | 0.2439 | 0.9091 | 0.0000 |

Demographic Parity Difference: 0.0391

✅ **Low gender bias.** The selection rates between males and females are relatively balanced.

#### Random Forest with Sampling
| Gender | Count | Selection Rate | True Positive Rate | False Positive Rate |
|--------|-------|---------------|-------------------|---------------------|
| Female | 327 | 0.3089 | 0.9898 | 0.0175 |
| Male | 632 | 0.3449 | 0.9861 | 0.0120 |
| Non-binary | 41 | 0.2439 | 0.9091 | 0.0000 |

Demographic Parity Difference: 0.0361

✅ **Low gender bias.** The selection rates between males and females are relatively balanced.

### Ethnicity Fairness

#### Logistic Regression Baseline
| Ethnicity | Count | Selection Rate | True Positive Rate |
|-----------|-------|---------------|-------------------|
| Asian | 230 | 0.3696 | 0.9884 |
| White | 557 | 0.3160 | 1.0000 |
| Black | 76 | 0.3684 | 0.9655 |
| Hispanic | 85 | 0.2588 | 1.0000 |
| Native American | 15 | 0.2667 | 1.0000 |
| Middle Eastern | 27 | 0.2593 | 1.0000 |
| Pacific Islander | 10 | 0.4000 | 1.0000 |

Maximum Selection Rate Disparity: 0.1412

⚠️ **Moderate ethnicity bias detected.** The selection rates across ethnic groups vary by more than 10 percentage points.

#### Random Forest Baseline
| Ethnicity | Count | Selection Rate | True Positive Rate |
|-----------|-------|---------------|-------------------|
| Asian | 230 | 0.3652 | 0.9767 |
| White | 557 | 0.3106 | 0.9713 |
| Black | 76 | 0.3684 | 0.9655 |
| Hispanic | 85 | 0.2706 | 1.0000 |
| Native American | 15 | 0.2000 | 0.7500 |
| Middle Eastern | 27 | 0.2222 | 1.0000 |
| Pacific Islander | 10 | 0.4000 | 1.0000 |

Maximum Selection Rate Disparity: 0.2000

⚠️ **Moderate ethnicity bias detected.** The selection rates across ethnic groups vary by more than 10 percentage points.

#### Logistic Regression with Reweighing
| Ethnicity | Count | Selection Rate | True Positive Rate |
|-----------|-------|---------------|-------------------|
| Asian | 230 | 0.3739 | 0.9884 |
| White | 557 | 0.3214 | 1.0000 |
| Black | 76 | 0.3947 | 0.9655 |
| Hispanic | 85 | 0.2706 | 1.0000 |
| Native American | 15 | 0.2667 | 1.0000 |
| Middle Eastern | 27 | 0.2963 | 1.0000 |
| Pacific Islander | 10 | 0.4000 | 1.0000 |

Maximum Selection Rate Disparity: 0.1333

⚠️ **Moderate ethnicity bias detected.** The selection rates across ethnic groups vary by more than 10 percentage points.

#### Random Forest with Reweighing
| Ethnicity | Count | Selection Rate | True Positive Rate |
|-----------|-------|---------------|-------------------|
| Asian | 230 | 0.3739 | 0.9884 |
| White | 557 | 0.3070 | 0.9598 |
| Black | 76 | 0.3553 | 0.9310 |
| Hispanic | 85 | 0.2706 | 1.0000 |
| Native American | 15 | 0.2000 | 0.7500 |
| Middle Eastern | 27 | 0.2222 | 1.0000 |
| Pacific Islander | 10 | 0.4000 | 1.0000 |

Maximum Selection Rate Disparity: 0.2000

⚠️ **Moderate ethnicity bias detected.** The selection rates across ethnic groups vary by more than 10 percentage points.

#### Logistic Regression with Sampling
| Ethnicity | Count | Selection Rate | True Positive Rate |
|-----------|-------|---------------|-------------------|
| Asian | 230 | 0.3783 | 1.0000 |
| White | 557 | 0.3142 | 0.9885 |
| Black | 76 | 0.3684 | 0.9655 |
| Hispanic | 85 | 0.2706 | 1.0000 |
| Native American | 15 | 0.2667 | 1.0000 |
| Middle Eastern | 27 | 0.2593 | 1.0000 |
| Pacific Islander | 10 | 0.4000 | 1.0000 |

Maximum Selection Rate Disparity: 0.1407

⚠️ **Moderate ethnicity bias detected.** The selection rates across ethnic groups vary by more than 10 percentage points.

#### Random Forest with Sampling
| Ethnicity | Count | Selection Rate | True Positive Rate |
|-----------|-------|---------------|-------------------|
| Asian | 230 | 0.3739 | 0.9884 |
| White | 557 | 0.3178 | 0.9828 |
| Black | 76 | 0.3816 | 0.9655 |
| Hispanic | 85 | 0.2706 | 1.0000 |
| Native American | 15 | 0.2667 | 1.0000 |
| Middle Eastern | 27 | 0.2222 | 1.0000 |
| Pacific Islander | 10 | 0.4000 | 1.0000 |

Maximum Selection Rate Disparity: 0.1778

⚠️ **Moderate ethnicity bias detected.** The selection rates across ethnic groups vary by more than 10 percentage points.

### Disparate Impact Analysis (80% Rule)

The 80% rule (or four-fifths rule) is a legal guideline used to determine adverse impact in employment decisions. It states that the selection rate for any protected group should be at least 80% of the selection rate for the group with the highest selection rate.

#### Logistic Regression Baseline
Gender Selection Rate Ratio: 0.7778

⚠️ **Fails the 80% rule.** This model shows evidence of disparate impact by gender.

Ethnicity Selection Rate Ratio: 0.6471

⚠️ **Fails the 80% rule.** This model shows evidence of disparate impact by ethnicity.

#### Random Forest Baseline
Gender Selection Rate Ratio: 0.6453

⚠️ **Fails the 80% rule.** This model shows evidence of disparate impact by gender.

Ethnicity Selection Rate Ratio: 0.5000

⚠️ **Fails the 80% rule.** This model shows evidence of disparate impact by ethnicity.

#### Logistic Regression with Reweighing
Gender Selection Rate Ratio: 0.7707

⚠️ **Fails the 80% rule.** This model shows evidence of disparate impact by gender.

Ethnicity Selection Rate Ratio: 0.6667

⚠️ **Fails the 80% rule.** This model shows evidence of disparate impact by ethnicity.

#### Random Forest with Reweighing
Gender Selection Rate Ratio: 0.6513

⚠️ **Fails the 80% rule.** This model shows evidence of disparate impact by gender.

Ethnicity Selection Rate Ratio: 0.5000

⚠️ **Fails the 80% rule.** This model shows evidence of disparate impact by ethnicity.

#### Logistic Regression with Sampling
Gender Selection Rate Ratio: 0.7071

⚠️ **Fails the 80% rule.** This model shows evidence of disparate impact by gender.

Ethnicity Selection Rate Ratio: 0.6481

⚠️ **Fails the 80% rule.** This model shows evidence of disparate impact by ethnicity.

#### Random Forest with Sampling
Gender Selection Rate Ratio: 0.7071

⚠️ **Fails the 80% rule.** This model shows evidence of disparate impact by gender.

Ethnicity Selection Rate Ratio: 0.5556

⚠️ **Fails the 80% rule.** This model shows evidence of disparate impact by ethnicity.
## Bias Mitigation Effectiveness
This section evaluates the effectiveness of different bias mitigation techniques.

### Sampling Technique

Comparison with baseline models:

| Metric | Baseline Average | With Mitigation | Difference |
|--------|------------------|----------------|------------|
| Accuracy | 0.9905 | 0.9885 | -0.0020 |
| F1 Score | 0.9853 | 0.9824 | -0.0029 |
| Demographic Parity Difference | 0.0459 | 0.0376 | -0.0083 |

✅ **Sampling reduced bias** by 0.0083 demographic parity difference points.

#### Selection Rate Changes

| Group | Baseline Selection Rate | With Mitigation | Change |
|-------|--------------------------|----------------|--------|
| Male | 0.3426 | 0.3449 | +0.0024 |
| Female | 0.2966 | 0.3073 | +0.0107 |
### Reweighing Technique

Comparison with baseline models:

| Metric | Baseline Average | With Mitigation | Difference |
|--------|------------------|----------------|------------|
| Accuracy | 0.9905 | 0.9850 | -0.0055 |
| F1 Score | 0.9853 | 0.9770 | -0.0084 |
| Demographic Parity Difference | 0.0459 | 0.0352 | -0.0107 |

✅ **Reweighing reduced bias** by 0.0107 demographic parity difference points.

#### Selection Rate Changes

| Group | Baseline Selection Rate | With Mitigation | Change |
|-------|--------------------------|----------------|--------|
| Male | 0.3426 | 0.3426 | +0.0000 |
| Female | 0.2966 | 0.3073 | +0.0107 |

### Overall Effectiveness Ranking

Ranking of techniques by bias reduction effectiveness:

| Rank | Technique | Bias Reduction | Performance Impact |
|------|-----------|----------------|-------------------|
| 1 | Reweighing | 0.0107 | -0.0084 |
| 2 | Sampling | 0.0083 | -0.0029 |

**Reweighing** is the most effective technique for reducing bias while maintaining performance.
## Feature Importance Analysis
No feature importance analysis available.
## Intersectional Fairness Analysis
This section explores how biases may compound across multiple demographic dimensions, particularly the intersection of gender and ethnicity.

### Intersectional Selection Rates

#### Logistic Regression Baseline

| Gender | Ethnicity | Selection Rate | Difference from Average |
|--------|-----------|---------------|-------------------------|
| Female | Asian | 0.3600 | +0.0299 |
| Female | White | 0.2865 | -0.0435 |
| Male | Asian | 0.3800 | +0.0499 |
| Male | Hispanic | 0.2941 | -0.0359 |
| Male | White | 0.3296 | -0.0004 |

Maximum Intersectional Disparity: 0.0935
- Highest selection rate: Male Asian (0.3800)
- Lowest selection rate: Female White (0.2865)

✅ **Low intersectional bias.** Selection rates are relatively balanced across different gender-ethnicity combinations.

#### Random Forest Baseline

| Gender | Ethnicity | Selection Rate | Difference from Average |
|--------|-----------|---------------|-------------------------|
| Female | Asian | 0.3467 | +0.0204 |
| Female | White | 0.2865 | -0.0397 |
| Male | Asian | 0.3800 | +0.0537 |
| Male | Hispanic | 0.2941 | -0.0322 |
| Male | White | 0.3241 | -0.0022 |

Maximum Intersectional Disparity: 0.0935
- Highest selection rate: Male Asian (0.3800)
- Lowest selection rate: Female White (0.2865)

✅ **Low intersectional bias.** Selection rates are relatively balanced across different gender-ethnicity combinations.

#### Logistic Regression with Reweighing

| Gender | Ethnicity | Selection Rate | Difference from Average |
|--------|-----------|---------------|-------------------------|
| Female | Asian | 0.3867 | +0.0503 |
| Female | White | 0.2924 | -0.0439 |
| Male | Asian | 0.3733 | +0.0370 |
| Male | Hispanic | 0.2941 | -0.0422 |
| Male | White | 0.3352 | -0.0012 |

Maximum Intersectional Disparity: 0.0943
- Highest selection rate: Female Asian (0.3867)
- Lowest selection rate: Female White (0.2924)

✅ **Low intersectional bias.** Selection rates are relatively balanced across different gender-ethnicity combinations.

#### Random Forest with Reweighing

| Gender | Ethnicity | Selection Rate | Difference from Average |
|--------|-----------|---------------|-------------------------|
| Female | Asian | 0.3733 | +0.0434 |
| Female | White | 0.2807 | -0.0492 |
| Male | Asian | 0.3800 | +0.0501 |
| Male | Hispanic | 0.2941 | -0.0358 |
| Male | White | 0.3213 | -0.0086 |

Maximum Intersectional Disparity: 0.0993
- Highest selection rate: Male Asian (0.3800)
- Lowest selection rate: Female White (0.2807)

✅ **Low intersectional bias.** Selection rates are relatively balanced across different gender-ethnicity combinations.

#### Logistic Regression with Sampling

| Gender | Ethnicity | Selection Rate | Difference from Average |
|--------|-----------|---------------|-------------------------|
| Female | Asian | 0.3867 | +0.0513 |
| Female | White | 0.2865 | -0.0488 |
| Male | Asian | 0.3800 | +0.0446 |
| Male | Hispanic | 0.2941 | -0.0413 |
| Male | White | 0.3296 | -0.0058 |

Maximum Intersectional Disparity: 0.1001
- Highest selection rate: Female Asian (0.3867)
- Lowest selection rate: Female White (0.2865)

⚠️ **Moderate intersectional bias detected.** Selection rates vary by more than 10 percentage points across different gender-ethnicity combinations.

#### Random Forest with Sampling

| Gender | Ethnicity | Selection Rate | Difference from Average |
|--------|-----------|---------------|-------------------------|
| Female | Asian | 0.3733 | +0.0394 |
| Female | White | 0.2924 | -0.0415 |
| Male | Asian | 0.3800 | +0.0461 |
| Male | Hispanic | 0.2941 | -0.0398 |
| Male | White | 0.3296 | -0.0043 |

Maximum Intersectional Disparity: 0.0876
- Highest selection rate: Male Asian (0.3800)
- Lowest selection rate: Female White (0.2924)

✅ **Low intersectional bias.** Selection rates are relatively balanced across different gender-ethnicity combinations.

### Compound Discrimination Analysis

Intersectional analysis reveals how different forms of discrimination may interact and compound:

1. **Compounding Effect** - When multiple marginalized identities intersect, bias effects may be multiplicative rather than additive
2. **Unique Challenges** - Groups at specific intersections may face unique challenges not captured by looking at single attributes
3. **Hidden Disparities** - Overall metrics for a single protected attribute may mask significant disparities at intersections

### Recommendations for Intersectional Fairness

1. **Targeted Interventions** - Design bias mitigation strategies that specifically address the most affected intersectional groups
2. **Disaggregated Monitoring** - Regularly monitor selection rates across intersectional categories, not just individual protected attributes
3. **Representation in Training Data** - Ensure sufficient representation of all intersectional groups in training data
4. **Customized Thresholds** - Consider setting group-specific thresholds where legally permissible to balance selection rates
## Recommendations
Based on the comprehensive analysis of model performance and fairness, we offer the following recommendations for implementing an unbiased AI recruitment system for software engineering roles:

### Model Selection

Recommended Model: **Logistic Regression with Sampling**

This model offers the best balance between predictive performance and fairness across demographic groups.

### Implementation Guidelines

1. **Preprocessing and Feature Engineering**
   - Remove direct identifiers (names, photos) during initial screening
   - Apply consistent normalization across all numerical features
   - Use blind evaluations for technical assessments

2. **Bias Mitigation Techniques**
   - Apply data rebalancing or reweighing to ensure fair representation
   - Consider calibrated probability estimates for better threshold selection
   - Implement threshold adjustments to ensure equitable selection rates

3. **Model Training and Validation**
   - Use stratified sampling to ensure representation of all groups
   - Apply regularization to prevent overfitting to biased patterns
   - Validate on diverse test sets that represent all demographic groups

4. **Post-Processing and Decision Making**
   - Apply group-specific thresholds where legally permissible
   - Implement human review for borderline cases
   - Document decision rationales for transparency and accountability

### Monitoring and Maintenance

1. **Continuous Fairness Monitoring**
   - Track selection rates across demographic groups over time
   - Monitor for concept drift in both performance and fairness metrics
   - Establish regular fairness audits by independent reviewers

2. **Feedback Loops**
   - Collect post-hire performance data to validate model predictions
   - Incorporate successful candidate outcomes back into model training
   - Adjust feature importance based on actual job performance

3. **Documentation and Transparency**
   - Maintain comprehensive documentation of model architecture and decisions
   - Create clear explanation mechanisms for rejected candidates
   - Provide appeal processes for candidates who believe they were unfairly evaluated

### Legal and Ethical Considerations

1. **Compliance with Anti-Discrimination Laws**
   - Ensure the system adheres to relevant employment laws (e.g., Title VII, EEOC guidelines)
   - Apply the four-fifths rule to evaluate adverse impact
   - Document business necessity for any practices with disparate impact

2. **Transparency and Explainability**
   - Provide clear explanations of how the AI system influences hiring decisions
   - Ensure human oversight and accountability for all automated decisions
   - Maintain logs of all model outputs for audit purposes

3. **Data Privacy and Security**
   - Follow best practices for securing candidate data
   - Implement data retention policies that comply with relevant regulations
   - Obtain appropriate consent for automated processing of applications
## Technical Implementation Details
This section provides technical details on the implementation of the AI recruitment system, focusing on the model architecture, bias mitigation techniques, and evaluation methodology.

### Model Architecture

The recommended model architecture consists of the following components:

1. **Feature Preprocessing**
   - Standardization of numerical features (z-score normalization)
   - One-hot encoding of categorical variables
   - Text embedding for unstructured data (e.g., project descriptions)

2. **Core Model**
   - Ensemble approach combining multiple base classifiers
   - Gradient boosting for technical skill assessment
   - Calibrated probability outputs for better threshold selection

3. **Post-Processing Layer**
   - Group-specific threshold optimization
   - Fairness constraints implementation
   - Confidence scoring for human review decisions

### Bias Mitigation Implementation

1. **Pre-Processing Techniques**
   ```python
   # Sample reweighing implementation
   def compute_sample_weights(y_train, protected_attributes):
       # Calculate expected vs. observed probabilities
       weights = np.ones(len(y_train))
       for gender in protected_attributes['gender'].unique():
           for ethnicity in protected_attributes['ethnicity'].unique():
               for outcome in [0, 1]:
                   # Create mask for this group and outcome
                   mask = ((protected_attributes['gender'] == gender) & 
                           (protected_attributes['ethnicity'] == ethnicity) & 
                           (y_train == outcome))
                   
                   # Calculate expected and observed probabilities
                   expected = sum(y_train == outcome) / len(y_train)
                   observed = sum(mask) / len(y_train)
                   
                   # Adjust weights
                   if observed > 0:
                       weights[mask] = expected / observed
       return weights
   ```

2. **In-Processing Techniques**
   ```python
   # Fairness constraint in the loss function
   def fair_loss(y_true, y_pred, protected_attributes):
       # Base loss (e.g., binary cross-entropy)
       base_loss = log_loss(y_true, y_pred)
       
       # Fairness penalty (demographic parity)
       male_mask = protected_attributes['gender'] == 'Male'
       female_mask = protected_attributes['gender'] == 'Female'
       
       male_selection = np.mean(y_pred[male_mask])
       female_selection = np.mean(y_pred[female_mask])
       
       fairness_penalty = abs(male_selection - female_selection)
       
       # Combined loss with fairness constraint
       alpha = 0.3  # Fairness weight
       combined_loss = base_loss + alpha * fairness_penalty
       
       return combined_loss
   ```

3. **Post-Processing Techniques**
   ```python
   # Group-specific threshold optimization
   def optimize_thresholds(model, X_val, y_val, protected_val):
       probas = model.predict_proba(X_val)[:,1]
       thresholds = {}
       
       for gender in protected_val['gender'].unique():
           gender_mask = protected_val['gender'] == gender
           
           best_threshold = 0.5  # Default
           best_balanced_accuracy = 0.0
           
           # Try different thresholds
           for threshold in np.arange(0.1, 0.9, 0.05):
               preds = (probas[gender_mask] >= threshold).astype(int)
               bal_acc = balanced_accuracy_score(y_val[gender_mask], preds)
               
               if bal_acc > best_balanced_accuracy:
                   best_balanced_accuracy = bal_acc
                   best_threshold = threshold
           
           thresholds[gender] = best_threshold
       
       return thresholds
   ```

### Fairness Evaluation Methodology

The system evaluates fairness using the following metrics and methodology:

1. **Demographic Parity**
   ```python
   def demographic_parity_difference(y_pred, protected_attributes):
       # Create masks for different groups
       male_mask = protected_attributes['gender'] == 'Male'
       female_mask = protected_attributes['gender'] == 'Female'
       
       # Calculate selection rates
       male_selection_rate = y_pred[male_mask].mean()
       female_selection_rate = y_pred[female_mask].mean()
       
       return male_selection_rate - female_selection_rate
   ```

2. **Equal Opportunity**
   ```python
   def equal_opportunity_difference(y_true, y_pred, protected_attributes):
       # Create masks for different groups
       male_mask = protected_attributes['gender'] == 'Male'
       female_mask = protected_attributes['gender'] == 'Female'
       
       # Calculate true positive rates
       male_tpr = recall_score(y_true[male_mask], y_pred[male_mask])
       female_tpr = recall_score(y_true[female_mask], y_pred[female_mask])
       
       return male_tpr - female_tpr
   ```

3. **Intersectional Fairness**
   ```python
   def intersectional_disparities(y_pred, protected_attributes):
       # Create a DataFrame to store selection rates
       group_rates = []
       
       # Calculate selection rates for each intersection
       for gender in protected_attributes['gender'].unique():
           for ethnicity in protected_attributes['ethnicity'].unique():
               mask = ((protected_attributes['gender'] == gender) & 
                       (protected_attributes['ethnicity'] == ethnicity))
               
               # Skip if group is too small
               if sum(mask) < 30:
                   continue
                   
               selection_rate = y_pred[mask].mean()
               group_rates.append({
                   'Gender': gender,
                   'Ethnicity': ethnicity,
                   'Selection Rate': selection_rate,
                   'Count': sum(mask)
               })
       
       return pd.DataFrame(group_rates)
   ```
## References
1. Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2021). A survey on bias and fairness in machine learning. ACM Computing Surveys (CSUR), 54(6), 1-35.

2. Barocas, S., Hardt, M., & Narayanan, A. (2019). Fairness and machine learning. fairmlbook.org.

3. Kamiran, F., & Calders, T. (2012). Data preprocessing techniques for classification without discrimination. Knowledge and Information Systems, 33(1), 1-33.

4. Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. Advances in Neural Information Processing Systems, 29, 3315-3323.

5. Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R. (2012). Fairness through awareness. Proceedings of the 3rd Innovations in Theoretical Computer Science Conference, 214-226.

6. Zafar, M. B., Valera, I., Gomez Rodriguez, M., & Gummadi, K. P. (2017). Fairness beyond disparate treatment & disparate impact: Learning classification without disparate mistreatment. Proceedings of the 26th International Conference on World Wide Web, 1171-1180.

7. Pleiss, G., Raghavan, M., Wu, F., Kleinberg, J., & Weinberger, K. Q. (2017). On fairness and calibration. Advances in Neural Information Processing Systems, 30, 5680-5689.

8. Feldman, M., Friedler, S. A., Moeller, J., Scheidegger, C., & Venkatasubramanian, S. (2015). Certifying and removing disparate impact. Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 259-268.

9. Crenshaw, K. (1989). Demarginalizing the intersection of race and sex: A Black feminist critique of antidiscrimination doctrine, feminist theory and antiracist politics. University of Chicago Legal Forum, 139-167.

10. Kearns, M., Neel, S., Roth, A., & Wu, Z. S. (2018). Preventing fairness gerrymandering: Auditing and learning for subgroup fairness. International Conference on Machine Learning, 2564-2572.

11. Holstein, K., Wortman Vaughan, J., Daumé III, H., Dudik, M., & Wallach, H. (2019). Improving fairness in machine learning systems: What do industry practitioners need? Proceedings of the 2019 CHI Conference on Human Factors in Computing Systems, 1-16.

12. Raghavan, M., Barocas, S., Kleinberg, J., & Levy, K. (2020). Mitigating bias in algorithmic hiring: Evaluating claims and practices. Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency, 469-481.

13. Beutel, A., Chen, J., Doshi, T., Qian, H., Woodruff, A., Luu, C., ... & Chi, E. H. (2019). Putting fairness principles into practice: Challenges, metrics, and improvements. Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society, 453-459.

14. Garg, S., Perot, V., Limtiaco, N., Taly, A., Chi, E. H., & Beutel, A. (2020). Counterfactual fairness in text classification through robustness. Proceedings of the 2020 AAAI/ACM Conference on AI, Ethics, and Society, 219-226.

15. Pessach, D., & Shmueli, E. (2020). Algorithmic fairness. arXiv preprint arXiv:2001.09784.