# Run AEDISIGHT locally

These steps use the existing FastAPI application and its checked-in processed dataset and trained model artifacts. Use Python 3.12. The commands work in a VS Code integrated terminal or a regular terminal.

## Open the project in VS Code

Open the project folder, then choose **Terminal → New Terminal**. In the terminal, confirm its current directory is the project root. After creating the virtual environment below, open the Command Palette (`Ctrl+Shift+P`), select **Python: Select Interpreter**, and choose the interpreter inside `.venv`.

Keep any existing `.env` file and database. If `.env` does not exist, copy `.env.example` using the platform-specific command below. Set `APP_ENV=development`, `DATABASE_URL=sqlite:///./dengue.db`, and `COOKIE_SECURE=false` for local development. Replace the example `SESSION_SECRET` with a fresh random value of at least 32 characters. Generate one with:

```sh
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Edit `.env` locally to set the generated value. Never commit `.env` or use the example secret.

## Linux terminal or VS Code terminal (Bash)

From the project root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
test -f .env || cp .env.example .env
```

Set the local values and a fresh session secret in `.env`, then start the server:

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

## Windows PowerShell or VS Code PowerShell terminal

From the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Set the local values and a fresh session secret in `.env`, then start the server:

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

If PowerShell blocks virtual-environment activation, allow it for this terminal only and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Windows Command Prompt (cmd.exe)

From the project root:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if not exist .env copy .env.example .env
```

Set the local values and a fresh session secret in `.env`, then start the server:

```bat
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

## Open and stop the application

Visit <http://127.0.0.1:8000>. Leave the terminal running while you use the app. Press `Ctrl+C` in that terminal to stop it. If port 8000 is already in use, change the command to `--port 8001` and visit <http://127.0.0.1:8001>.

The project includes a local SQLite database file where supplied and will not reset it during these steps. For a fresh checkout, the SQLite database is created as needed. The processed dataset and trained horizon artifacts are included, so model training is not required to start the server.

## Forecast data status

The bundled dengue dataset ends on 2025-12-29. With the current freshness policy, live/current forecasting is disabled until recent weekly data is supplied. The dashboard can still demonstrate held-out historical replay; label it as historical replay, not live surveillance.
