import os
from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_model, save_model
from src.model_evaluation import evaluate_model, plot_evaluation_results
from src.visualisation import plot_feature_importance, plot_stat_distributions


def main():
    # Set paths
    data_path = os.path.join('data', 'Pokemon Project.csv')

    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data(data_path)
    X_train, X_test, y_train, y_test, scaler, features = preprocess_data(df)

    # Train model
    print("Training model...")
    model = train_model(X_train, y_train)

    # Save model
    print("Saving model...")
    save_model(model, scaler, features)

    # Evaluate model
    print("Evaluating model...")
    evaluation_results = evaluate_model(model, X_test, y_test)

    # Print results
    print(f"\nModel Accuracy: {evaluation_results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(evaluation_results['report'])

    # Create visualisations
    print("Creating visualisations...")
    plot_evaluation_results(evaluation_results)
    plot_feature_importance(model, features)
    plot_stat_distributions(df)

    print("\nCheck the 'results' directory for visualisations.")


if __name__ == "__main__":
    main()