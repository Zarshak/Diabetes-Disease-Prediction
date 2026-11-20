@echo off
echo STEP 1: Creating virtual environment (first run only)
python -m venv .venv
call .venv\Scripts\activate
echo STEP 2: Installing required libraries
python -m pip install --upgrade pip
pip install -r requirements.txt
echo STEP 3: Training and comparing the machine-learning models
python train_model.py
if errorlevel 1 (
  echo Model training failed. Read the error above.
  pause
  exit /b 1
)
echo STEP 4: Starting the Streamlit website
python -m streamlit run app.py
pause
