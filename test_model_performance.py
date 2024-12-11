import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import NoReturn, Dict, Any
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import model_evaluation

def generate_plot(fig) -> str:
    """Convert matplotlib figure to base64 string"""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    return img_str

def create_confusion_matrix_plot(conf_matrix: np.ndarray) -> str:
    """Generate confusion matrix heatmap"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    return generate_plot(plt.gcf())

def create_feature_importance_plot(model: RandomForestClassifier, features: list) -> str:
    """Generate feature importance plot"""
    importances = pd.Series(model.feature_importances_, index=features)
    plt.figure(figsize=(10, 6))
    importances.sort_values().plot(kind='barh')
    plt.title('Feature Importance')
    return generate_plot(plt.gcf())

def generate_html_report(
    performance_data: Dict[str, Any],
    conf_matrix_img: str,
    feature_importance_img: str
) -> str:
    """Generate detailed HTML report"""
    return f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Model Performance Report - {performance_data['timestamp']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }}
                .full-width {{ grid-column: 1 / -1; }}
                .card {{ 
                    border: 1px solid #ddd; 
                    padding: 20px; 
                    border-radius: 5px;
                    background-color: white;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                table {{ 
                    width: 100%; 
                    border-collapse: collapse; 
                    margin: 10px 0; 
                }}
                th, td {{ 
                    padding: 12px; 
                    text-align: left; 
                    border-bottom: 1px solid #ddd; 
                }}
                th {{ background-color: #f8f9fa; }}
                .metric-value {{ 
                    font-size: 24px; 
                    font-weight: bold; 
                    color: #2c3e50; 
                }}
                .plot-container {{ text-align: center; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Model Performance Report</h1>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <div class="container">
                <div class="card">
                    <h2>Dataset Overview</h2>
                    <table>
                        <tr>
                            <td>Training Set Size:</td>
                            <td class="metric-value">{performance_data['train_size']}</td>
                        </tr>
                        <tr>
                            <td>Test Set Size:</td>
                            <td class="metric-value">{performance_data['test_size']}</td>
                        </tr>
                    </table>
                </div>

                <div class="card">
                    <h2>Model Performance</h2>
                    <table>
                        <tr>
                            <td>Accuracy:</td>
                            <td class="metric-value">
                                {performance_data['classification_metrics']['accuracy']:.4f}
                            </td>
                        </tr>
                        <tr>
                            <td>Macro Avg F1-Score:</td>
                            <td class="metric-value">
                                {performance_data['classification_metrics']['macro avg']['f1-score']:.4f}
                            </td>
                        </tr>
                    </table>
                </div>

                <div class="card full-width">
                    <h2>Classification Report</h2>
                    <table>
                        <tr>
                            <th>Class</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>F1-Score</th>
                            <th>Support</th>
                        </tr>
                        {generate_classification_table(performance_data['classification_metrics'])}
                    </table>
                </div>

                <div class="card">
                    <h2>Confusion Matrix</h2>
                    <div class="plot-container">
                        <img src="data:image/png;base64,{conf_matrix_img}" alt="Confusion Matrix">
                    </div>
                </div>

                <div class="card">
                    <h2>Feature Importance</h2>
                    <div class="plot-container">
                        <img src="data:image/png;base64,{feature_importance_img}" alt="Feature Importance">
                    </div>
                </div>
            </div>
        </body>
    </html>
    """

def generate_classification_table(metrics: Dict) -> str:
    """Generate HTML table rows for classification report"""
    rows = []
    for class_name, values in metrics.items():
        if isinstance(values, dict) and class_name not in ['macro avg', 'weighted avg']:
            rows.append(f"""
                <tr>
                    <td>{class_name}</td>
                    <td>{values['precision']:.4f}</td>
                    <td>{values['recall']:.4f}</td>
                    <td>{values['f1-score']:.4f}</td>
                    <td>{values['support']}</td>
                </tr>
            """)
    return '\n'.join(rows)

def test_model_performance() -> NoReturn:
    """Tests model performance and generates comprehensive reports."""
    try:
        reports_dir = Path("performance_reports")
        data_path = Path("/content/AI-ML-Testing/data/iris.csv")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        reports_dir.mkdir(exist_ok=True)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at {data_path}")
            
        data = pd.read_csv(data_path)
        if data.empty:
            raise ValueError("Dataset is empty")
            
        X = data[['feature']]
        y = data['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Generate performance metrics
        classification_metrics = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Generate plots
        conf_matrix_img = create_confusion_matrix_plot(conf_matrix)
        feature_importance_img = create_feature_importance_plot(model, X.columns)
        
        # Prepare performance data
        performance_data = {
            "timestamp": timestamp,
            "model_type": "RandomForestClassifier",
            "model_params": model.get_params(),
            "classification_metrics": classification_metrics,
            "confusion_matrix": conf_matrix.tolist(),
            "train_size": len(X_train),
            "test_size": len(X_test)
        }
        
        # Generate and save reports
        html_content = generate_html_report(
            performance_data,
            conf_matrix_img,
            feature_importance_img
        )
        
        html_path = reports_dir / f"performance_report_{timestamp}.html"
        html_path.write_text(html_content)
        
        json_path = reports_dir / f"performance_metrics_{timestamp}.json"
        json_path.write_text(json.dumps(performance_data, indent=4))
        
    except Exception as e:
        print(f"Error in model performance testing: {str(e)}")

if __name__ == "__main__":
    test_model_performance()