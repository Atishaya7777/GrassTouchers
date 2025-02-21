from fastapi import FastAPI

app = FastAPI()


@app.get("/aitesting")
async def root():
    return {"message": "Hello World"}
