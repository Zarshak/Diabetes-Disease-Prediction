DIABETES DISEASE PREDICTION USING MACHINE LEARNING
Beginner's guide - Windows laptop

WHAT IS INCLUDED?
- app.py: the website the user interacts with
- train_model.py: cleans data, trains four algorithms, compares them and saves the selected model
- data/diabetes.csv: dataset will be downloaded here the first time training runs
- models/: trained model files are saved here
- model_comparison.csv: comparison of four models after training
- requirements.txt: Python libraries needed

BEFORE STARTING
1. Use a Windows laptop or college computer.
2. Install Python 3.10 or newer from https://www.python.org/downloads/
3. During installation, tick the box "Add Python to PATH".
4. Extract this ZIP folder. Do not run files while they are still inside the ZIP.
5. Connect to the internet for the first setup and dataset download.

EASIEST WAY TO RUN
1. Open the extracted folder.
2. Double-click run_project.bat.
3. Wait while Python creates an environment and installs libraries. This may take several minutes.
4. The program downloads the dataset, trains Logistic Regression, Decision Tree, Random Forest and KNN, and saves a model.
5. A browser tab should open with the Streamlit application. If it does not, look at the black Command Prompt window for a Local URL, usually http://localhost:8501, and open it in Chrome.
6. Enter sample values and press Predict.
7. Keep the black Command Prompt window open while demonstrating the website. Close it only after the demo.

MANUAL WAY (IF THE BAT FILE DOES NOT WORK)
Open Command Prompt inside the extracted folder and run these commands one by one:
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py

SAMPLE VALUES FOR A DEMO ONLY
Pregnancies: 2
Glucose: 120
Blood pressure: 70
Skin thickness: 20
Insulin: 80
BMI: 28
Diabetes pedigree function: 0.5
Age: 30
These are example inputs, not a real person's health record. Do not claim the output is a medical diagnosis.

IMPORTANT
- The first run requires internet access to download the public CSV dataset and Python packages.
- If the college computer has no internet, ask the lab assistant to allow installation/download beforehand.
- Never claim accuracy before running train_model.py. Actual metrics are written to model_comparison.csv.
- The Pima dataset is limited and represents a specific study population. The model is an academic demonstration, not a clinical diagnostic tool.
