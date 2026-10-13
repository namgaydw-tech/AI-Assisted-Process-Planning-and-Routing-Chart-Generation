# FIRST RUN IN VS CODE (Windows PowerShell)

## 0. Unzip to your preferred folder

Recommended: `D:\PS04_Manufacturing_AI` or `%USERPROFILE%\Desktop\PS04_Manufacturing_AI`.
Open VS Code: File > Open Folder > choose `PS04_Manufacturing_AI`.

## 1. Backend terminal

```powershell
cd D:\PS04_Manufacturing_AI\backend
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

If 3.11 is not installed, use `py -3.12` or `py -3` for the *core* baseline.

## 2. Frontend terminal

```powershell
cd D:\PS04_Manufacturing_AI\frontend
npm install
npm run dev
```

Browse http://127.0.0.1:5173 and click `Generate process route`.

## 3. Validate

In a separate backend terminal:

```powershell
cd D:\PS04_Manufacturing_AI\backend
.\.venv\Scripts\python.exe -m pytest -q
```

Frontend:

```powershell
cd D:\PS04_Manufacturing_AI\frontend
npm run build
```

### Notes

- Run the backend and frontend terminals simultaneously.
- `http://127.0.0.1:8000/docs` contains the interactive FastAPI tester.
- STEP upload requires optional CadQuery installation. If absent the app will explain the missing dependency; manual routing remains fully functional.
- No CAD-feature recognition model or language model is trained yet. Acquire real datasets and expert route labels first.
- See `README.md` and `docs/RESEARCH_PLAN.md` before claiming research results.
