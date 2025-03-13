from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics."""
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    return {
        'accuracy': accuracy,
        'report': report,
        'conf_matrix': conf_matrix,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_prob': y_prob
    }


def plot_evaluation_results(evaluation_results, output_dir='results'):
    """Plot and save evaluation metrics."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

        # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        evaluation_results['conf_matrix'],
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Non-Legendary', 'Legendary'],
        yticklabels=['Non-Legendary', 'Legendary']
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(
        evaluation_results['fpr'],
        evaluation_results['tpr'],
        color='darkorange',
        lw=2,
        label=f'ROC curve (area = {evaluation_results["roc_auc"]:.2f})'
    )
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))

    # Plot feature importance
    return plt