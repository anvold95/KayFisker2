from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI()

# Serve static HTML (optional)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return HTMLResponse(open("static/index.html", encoding="utf-8").read())

# Example API route
@app.get("/ping")
def ping():
    return {"message": "pong"}

# If you want to run locally for testing:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860)
