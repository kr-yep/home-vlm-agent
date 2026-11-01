from fastapi.testclient import TestClient
from PIL import Image

from home_vlm.pipeline import SeenObject
from home_vlm.server import create_app


class FakeSource:
    def __init__(self, objects):
        self.objects = objects

    def snapshot(self):
        return Image.new("RGB", (64, 64)), self.objects, b"jpeg"

    def largest_crop(self):
        return Image.new("RGB", (32, 32)) if self.objects else None


class FakeChatter:
    def __init__(self):
        self.seen = None

    def chat(self, question, scene, images=()):
        self.seen = (question, scene, len(images))
        return "我看到一個馬克杯。"


MUG = SeenObject(track_id=1, yolo_name="cup", yolo_confidence=0.31, box=(1, 2, 30, 40), name="馬克杯", detail="白色")


def test_objects_lists_vlm_names_with_yolo_hint():
    client = TestClient(create_app(FakeSource([MUG]), FakeChatter()))
    body = client.get("/api/objects").json()
    assert body["objects"][0]["name"] == "馬克杯"
    assert body["objects"][0]["yolo_name"] == "cup"


def test_chat_passes_scene_text_and_one_crop_to_the_model():
    chatter = FakeChatter()
    client = TestClient(create_app(FakeSource([MUG]), chatter))
    response = client.post("/api/chat", json={"question": "你看到什麼？"})
    assert response.status_code == 200
    assert response.json()["answer"] == "我看到一個馬克杯。"
    question, scene, images = chatter.seen
    assert "馬克杯" in scene and "cup" in scene
    assert images == 1


def test_chat_with_empty_scene_sends_no_image():
    chatter = FakeChatter()
    client = TestClient(create_app(FakeSource([]), chatter))
    client.post("/api/chat", json={"question": "有東西嗎"})
    assert chatter.seen[1] == "（畫面中沒有偵測到物品）"
    assert chatter.seen[2] == 0


def test_chat_rejects_empty_question():
    client = TestClient(create_app(FakeSource([]), FakeChatter()))
    assert client.post("/api/chat", json={"question": ""}).status_code == 422


def test_objects_reports_a_dead_frame_loop():
    source = FakeSource([])
    source.error = "ModuleNotFoundError: No module named 'lap'"
    body = TestClient(create_app(source, FakeChatter())).get("/api/objects").json()
    assert body["error"].startswith("ModuleNotFoundError")
