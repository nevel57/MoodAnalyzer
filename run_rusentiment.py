import uvicorn

if __name__ == "__main__":
    print("Starting RuSentiment API")
    print("Docs: http://localhost:8000/docs")
    uvicorn.run(
        "app.main_rusentiment:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )