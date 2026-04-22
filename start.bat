@echo off
echo Starting AI Courtroom v2.0...

echo 1. Setting up Backend...
cd backend
python -m venv venv
call venv\Scripts\activate.bat
pip install -r requirements.txt
cd ..
alembic upgrade head
cd backend

echo 2. Starting Backend Server...
start "AI Courtroom - Backend" cmd /k "call venv\Scripts\activate.bat & uvicorn main:app --reload --port 8000"

cd ..

echo 3. Setting up Frontend...
call npm install

echo 4. Starting Frontend Server...
start "AI Courtroom - Frontend" cmd /k "npm run dev"

echo.
echo ========================================================
echo The application is starting!
echo Two new terminal windows have been opened.
echo.
echo Frontend will be available at: http://localhost:3000
echo Backend API will be available at: http://localhost:8000
echo ========================================================
echo.
