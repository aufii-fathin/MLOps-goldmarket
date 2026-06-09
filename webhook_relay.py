from fastapi import FastAPI, Request
import httpx
import os

app = FastAPI()

GITHUB_TOKEN = os.getenv("GITHUB_PAT")
GITHUB_URL = "https://api.github.com/repos/aufii-fathin/MLOps-goldmarket/dispatches"

@app.post("/relay")
async def relay(request: Request):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            GITHUB_URL,
            json={"event_type": "retraining-triggered"},
            headers={
                "Authorization": f"Bearer {GITHUB_TOKEN}",
                "Accept": "application/vnd.github.v3+json"
            }
        )
    return {"status": response.status_code}