@echo off
title Wind Turbine Blade Defect + Severity - Demo
cd /d "%~dp0"
echo ======================================================================
echo  WIND TURBINE BLADE DEFECT TYPE + SEVERITY  (model: proposed_multitask_cbam)
echo ======================================================================
echo.
echo [1/2] Terminal inference on test image 4.jpg (crack, severity 5)
python inference/predict_single.py --image "data/figshare/WT blade defect dataset/JPEGImages/4.jpg"
echo.
echo [2/2] Opening the demo page in your browser (close this window to stop it)
python -m streamlit run app/demo_app.py --browser.gatherUsageStats false
