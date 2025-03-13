from sklearn.ensemble import RandomForestClassifier
import joblib
import os


def train_model(X_train, y_train):
    """Train a Random Forest classifier."""
    # Create and train the model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, scaler, features, output_dir='models'):
    """Save the trained model and scaler."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    joblib.dump(model, os.path.join(output_dir, 'pokemon_legendary_model.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

    # Save feature names for reference
    with open(os.path.join(output_dir, 'features.txt'), 'w') as f:
        for feature in features:
            f.write(f"{feature}\n")