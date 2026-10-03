"""Run: python -m uvicorn main:app --reload"""
from fastapi import FastAPI

app = FastAPI(title="My First API")


@app.get("/")
def welcome():
    return {"message": "Welcome to my FastAPI app!"}


@app.get("/greet/{name}")
def greet(name: str):
    return {"name": name, "message": f"Hello, {name}!"}