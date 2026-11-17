@echo off
echo ========================================================
echo Starting Kisan Mandi Intelligence (SIH26132)
echo AI Farmer Market & Price Discovery Platform
echo ========================================================
echo.

REM Check if dependencies are installed
python -m pip install -r requirements.txt

REM Launch Streamlit app
python -m streamlit run app.py --server.port 8501

pause
