config/	          - stores the configuration file that controls all the settings for the project
data/	          - holds the dataset used for training and testing the model
development/	  - contains the web application files for the user interface
docs/	          - has documentation files explaining the project
logs/	          - stores training logs that show what happened during model training
models/	          - saves the trained models and preprocessor after training is complete
reports/	      - contains all the plots and visualizations generated during the project
src/	          - contains all the source code for the project
tests/	          - holds unit tests to check if the code works correctly
.gitignore	      - tells git which files and folders to ignore when uploading to github
create_dirs.py	  - creates all the required folders automatically
main.py	          - the main file that runs the entire pipeline from start to finish


CONFIG FOLDER
configs.py	    - central file that stores all paths, hyperparameters, and settings for the project


DATA FOLDER
raw/	        - contains the original dataset file
processed/	    - stores the cleaned and preprocessed data ready for the model



DEVELOPMENT FOLDER
static/	        - stores CSS files for styling the web pages
style.css	    - dark theme styles for the web application
templates/	    - stores HTML templates for the web pages
base.html	    - the main layout template used by all pages
index.html	    - the prediction page where users enter customer details
logs.html	    - the page that shows training logs
plots.html	    - the page that displays all visualizations
app.py	        - the flask application that runs the web server and handles predictions


DOCS FOLDER
file_structure.md	    - explains the project file structure
problem_statement.md	- describes the problem the project solves


LOGS FOLDER
training.log	    - the log file that records everything that happens during training


MODELS FOLDER
custom_model.pkl	- the trained custom logistic regression model saved for later use
sklearn_model.pkl	- the trained scikit-learn model saved for comparison
preprocessor.pkl	- the saved data preprocessor pipeline used for transforming new data
metrics.json	    - stores the performance metrics from training

REPORTS FOLDER
figures/	        - stores all the plots generated during the project



SRC FOLDER (Source Code)

SRC/DATA
eda.py	           -  performs exploratory data analysis and generates plots to understand the data
loader.py	       - loads the raw dataset from the CSV file
preprocessor.py	   - cleans the data, creates new features, encodes categories, and scales numbers


SRC/EVALUATION
benchmark.py	    - compares my custom model against scikit-learn to verify correctness
metrics.py	        - computes accuracy, precision, recall, F1 score, and ROC-AUC
visualizer.py	    - generates all the plots like confusion matrices and ROC curves

SRC/MODELS
custom_logistic.py	    - the logistic regression model built from scratch using NumPy


SRC/TRAINING
File/Folder	What It Does
trainer.py	orchestrates the training process, fits the model, and saves it


SRC/UTILS
io_helpers.py	    - handles saving and loading models and preprocessors
logger.py	        - sets up logging to both the console and the log file



TESTS FOLDER
test_preprocessor.py	    - tests if the preprocessor works correctly
test_custom_logistic.py	    - tests if the custom logistic model works correctly
test_integration.py	        - tests if everything works together as expected