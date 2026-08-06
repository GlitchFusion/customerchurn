churn_prediction/
│
├── .env                                 # Environment variables (optional)
├── .gitignore
├── README.md                            # Setup & run instructions
├── requirements.txt                     # All dependencies
├── main.py                              # Entry point to run everything
│
├── config/                              # Configuration layer
│   └── configs.py                       # Centralised params (paths, hyperparams, seeds)
│
├── src/                                 # Core source code (modular)
│   ├── data/
│   │   ├── loader.py                    # Raw data loading
│   │   └── preprocessor.py              # Cleaning, encoding, scaling, feature creation
│   │
│   ├── models/
│   │   └── custom_logistic.py           # NumPy LogisticRegression class
│   │
│   ├── training/
│   │   └── trainer.py                   # Orchestrates training (fit, save model)
│   │
│   ├── evaluation/
│   │   ├── metrics.py                   # Accuracy, Precision, Recall, F1, AUC
│   │   ├── visualizer.py                # ROC, Confusion Matrix, Feature Importance plots, and other plots
│   │   └── benchmark.py                 # Compares custom vs. sklearn model
│   │
│   └── utils/
│       ├── logger.py                    # Sets up logging (to file and console)
│       └── io_helpers.py                # Save/load models, preprocessors (pickle/joblib)
│
├── tests/                               # Unit & integration tests
│   ├── test_preprocessor.py
│   ├── test_custom_logistic.py
│   └── test_integration.py
│
├── deployment/                          # Web service for predictions
│   ├── app.py                           # Flask/FastAPI entry point
│   ├── static/                          # (if you add a CSS/JS frontend)
│   └── templates/                       # (if using Flask HTML templates)
│
├── data/                                # All datasets
│   ├── raw/
│   │   └── Telco-Customer-Churn-Data.csv.csv              # Place your original file here
│   ├── processed/
│   │   └── churn_processed.csv          # Saved after preprocessing
│
├── models/                              # Persisted models + preprocessors
│   ├── custom_model.pkl
│   ├── sklearn_model.pkl
│   └── preprocessor.pkl                 # Fit encoders/scalers for deployment
│
├── reports/                             # Analytical outputs
│   ├── figures/
│   │   ├── roc_curve.png
│   │   ├── confusion_matrix.png
│   │   └── feature_importance.png
│   └── final_report.md                  # Your written analysis
│
└── logs/                                # Runtime logs
    └── training.log