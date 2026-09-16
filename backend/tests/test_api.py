from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_logs():
    response = client.get("/logs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_update_simulation():
    response = client.post(
        "/simulate",
        json={"dataset_name": "DEAP", "artifacts": ["Ocular (Blink)"]}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_auth_login():
    response = client.post(
        "/auth/token",
        json={"username": "admin", "password": "admin"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["role"] == "Admin"
    
def test_auth_invalid_login():
    response = client.post(
        "/auth/token",
        json={"username": "wrong", "password": "wrong"}
    )
    assert response.status_code == 401
