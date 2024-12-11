import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import NoReturn
import json
from sklearn.model_selection import train_test_split
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import TrainTestFeatureDrift

def test_data_validation() -> NoReturn:
    """
    Validates dataset using TrainTestFeatureDrift check and generates reports.
    """
    try:
        # Setup paths
        reports_dir = Path("validation_reports")
        data_path = Path("/content/AI-ML-Testing/data/iris.csv")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create reports directory
        reports_dir.mkdir(exist_ok=True)
        
        # Validate data file exists
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at {data_path}")
        
        # Load and prepare data
        data = pd.read_csv(data_path)
        if data.empty:
            raise ValueError("Dataset is empty")
            
        data.columns = ['feature', 'label']
        
        # Split data
        train_data, test_data = train_test_split(
            data, test_size=0.2, random_state=42
        )
        
        if len(train_data) == 0 or len(test_data) == 0:
            raise ValueError("Train/test split resulted in empty datasets")
        
        # Create datasets for validation
        train_dataset = Dataset(train_data, label='label', cat_features=[])
        test_dataset = Dataset(test_data, label='label', cat_features=[])
        
        # Run drift check
        check = TrainTestFeatureDrift()
        result = check.run(train_dataset=train_dataset, test_dataset=test_dataset)
        
        # Generate HTML report
        html_content = f"""
        <html>
            <head>
                <title>Data Validation Report - {timestamp}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ background-color: #f8f9fa; padding: 20px; }}
                    .content {{ margin-top: 20px; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Data Validation Report</h1>
                    <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                </div>
                <div class="content">
                    <h2>Dataset Overview</h2>
                    <p>Train set size: {len(train_data)}</p>
                    <p>Test set size: {len(test_data)}</p>
                    <h2>Drift Analysis</h2>
                    <p>Feature drift score: {result.value.get('feature', 'N/A')}</p>
                </div>
            </body>
        </html>
        """
        
        # Save reports
        report_path = reports_dir / f"drift_report_{timestamp}.html"
        report_path.write_text(html_content)
        
        # Save JSON results
        json_results = {
            "timestamp": timestamp,
            "drift_scores": result.value,
            "train_size": len(train_data),
            "test_size": len(test_data),
            "train_mean": float(train_data['feature'].mean()),
            "test_mean": float(test_data['feature'].mean())
        }
        
        json_path = reports_dir / f"drift_results_{timestamp}.json"
        json_path.write_text(json.dumps(json_results, indent=4))
        
        # Save visualization
        viz_path = reports_dir / f"drift_visualization_{timestamp}.html"
        result.save_as_html(str(viz_path))
        
        print(f"Reports saved in {reports_dir} directory")
        
    except FileNotFoundError as e:
        print(f"File error: {str(e)}")
    except ValueError as e:
        print(f"Data validation error: {str(e)}")
    except Exception as e:
        print(f"Error in data validation: {str(e)}")

if __name__ == "__main__":
    test_data_validation()