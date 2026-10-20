# 🚀 My First API (FastAPI)

A minimal FastAPI app with two GET endpoints: a welcome message and a personalised greeting.
Good as a starting template for learning FastAPI.

## Endpoints

| Method | Path | Description | Example response |
|---|---|---|---|
| GET | `/` | Welcome message | `{"message": "Welcome to my FastAPI app!"}` |
| GET | `/greet/{name}` | Greets the given name | `{"name": "Melvin", "message": "Hello, Melvin!"}` |

## Requirements

- Python 3.9+

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS

pip install -r requirements.txt
```

## Run

```bash
python -m uvicorn main:app --reload
```

`--reload` restarts the server automatically when you edit `main.py`, so use it during development only.

The server starts at **http://127.0.0.1:8000**.

## Try it

**Browser**
- http://127.0.0.1:8000/
- http://127.0.0.1:8000/greet/Melvin

**Interactive docs** (generated automatically by FastAPI)
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

**Command line**

```bash
curl http://127.0.0.1:8000/
curl http://127.0.0.1:8000/greet/Melvin
```

PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/greet/Melvin
```

## Run on a different port or open to your network

```bash
python -m uvicorn main:app --reload --port 8080
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## Project layout

```
.
├── main.py
├── requirements.txt
└── README.md
```

## Next steps

- Add a POST endpoint with a Pydantic model for the request body
- Add query parameters, e.g. `/greet/{name}?lang=ta`
- Write tests with `fastapi.testclient.TestClient` and `pytest`
