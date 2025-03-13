**Project Overview**
This machine learning project predicts whether a Pokemon is legendary based on its base statistics. Using a Random Forest classifier, the model achieves 96.27% accuracy in distinguishing between legendary and non-legendary Pokemon.

**Dataset**
The dataset contains information on 801 Pokemon across multiple generations, including:

Base stats (HP, Attack, Defense, Special Attack, Special Defense, Speed)
Type information
Physical characteristics (height, weight)
Other attributes (capture rate, base happiness, etc.)
Only 8.74% of Pokemon in the dataset are legendary, making this an imbalanced classification problem.

**Features Used**
The model uses seven key statistical features:

HP
Attack
Defense
Special Attack
Special Defense
Speed
Base Total (sum of all base stats)

**Model Performance**
Accuracy: 96.27%
Precision for non-legendary Pokemon: 98%
Precision for legendary Pokemon: 79%
Recall for non-legendary Pokemon: 98%
Recall for legendary Pokemon: 79%

**Key Findings**
Base Total is the most important feature for predicting legendary status, which aligns with the game mechanics where legendary Pokemon typically have higher overall stats.
Legendary Pokemon have significantly higher base stats across all categories compared to non-legendary Pokemon.
Despite the imbalanced dataset (only 8.74% legendary Pokemon), the model performs well on both classes.

**Technical Implementation**
Data Preprocessing: Handled missing values and scaled features using StandardScaler
Model: Random Forest Classifier with optimized parameters
Evaluation: Used stratified train-test split to maintain class distribution
Visualization: Created plots for feature importance, confusion matrix, and stat distributions

**Future Improvements**
Incorporate Pokemon types as features
Experiment with other algorithms (XGBoost, Neural Networks)
Implement hyperparameter tuning

Add a simple web interface for making predictions
Expand to predict other Pokemon characteristics (e.g., type, generation)

**Skills Demonstrated**
Data preprocessing and cleaning
Handling imbalanced datasets
Feature selection and engineering
Machine learning model training and evaluation
Data visualization
Python programming (pandas, scikit-learn, matplotlib)
Project organization and documentation
