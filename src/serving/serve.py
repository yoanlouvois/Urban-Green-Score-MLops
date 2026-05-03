import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "serving.app:app",
        host="0.0.0.0",
        port=8080,
        workers=1,
    )