from fastapi import FastAPI

from ai import test

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/aitesting/")
async def aitesting():
    return {"message": test.test()}

