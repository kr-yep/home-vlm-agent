import io

from fastapi.testclient import TestClient
from PIL import Image

from home_vlm import server
from home_vlm.pipeline import BrowserCamera, SeenObject
from home_vlm.server import create_app


class FakeSource:
    def __init__(self, objects):
        self.objects = objects

    def snapshot(self):
        return Image.new("RGB", (64, 64)), self.objects, b"jpeg"


class FakeChatter:
    def __init__(self):
        self.seen = None

    def chat(self, question, scene, images=()):
        self.seen = (question, scene, len(images))
        return "我看到一個馬克杯。"


MUG = SeenObject(track_id=1, yolo_name="cup", yolo_confidence=0.31, box=(1, 2, 30, 40), name="馬克杯", detail="白色")
UNNAMED = SeenObject(track_id=2, yolo_name="hindu temple", yolo_confidence=0.9, box=(1, 2, 30, 40), name=None, detail=None)


def test_objects_lists_vlm_names_with_yolo_hint():
    client = TestClient(create_app(FakeSource([MUG]), FakeChatter()))
    body = client.get("/api/objects").json()
    assert body["objects"][0]["name"] == "馬克杯"
    assert body["objects"][0]["yolo_name"] == "cup"


def test_chat_passes_named_objects_and_the_frame_to_the_model():
    chatter = FakeChatter()
    client = TestClient(create_app(FakeSource([MUG, UNNAMED]), chatter))
    response = client.post("/api/chat", json={"question": "你看到什麼？"})
    assert response.status_code == 200
    assert response.json()["answer"] == "我看到一個馬克杯。"
    question, scene, images = chatter.seen
    assert "馬克杯" in scene and "1 個物體還沒辨識" in scene and "hindu temple" not in scene
    assert images == 1


def test_chat_with_nothing_detected_still_sends_the_frame():
    chatter = FakeChatter()
    client = TestClient(create_app(FakeSource([]), chatter))
    client.post("/api/chat", json={"question": "有東西嗎"})
    assert chatter.seen[1] == "（畫面中沒有偵測到物品）"
    assert chatter.seen[2] == 1


def test_chat_rejects_empty_question():
    client = TestClient(create_app(FakeSource([]), FakeChatter()))
    assert client.post("/api/chat", json={"question": ""}).status_code == 422


def test_objects_reports_a_dead_frame_loop():
    source = FakeSource([])
    source.error = "ModuleNotFoundError: No module named 'lap'"
    body = TestClient(create_app(source, FakeChatter())).get("/api/objects").json()
    assert body["error"].startswith("ModuleNotFoundError")


def test_objects_says_whether_frames_are_arriving():
    source = FakeSource([])
    client = TestClient(create_app(source, FakeChatter()))
    assert client.get("/api/objects").json()["receiving"] is True
    source.snapshot = lambda: (None, [], None)
    assert client.get("/api/objects").json()["receiving"] is False



def jpeg_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), "red").save(buffer, format="JPEG")
    return buffer.getvalue()


def test_page_offers_the_camera_only_when_the_source_is_the_browser():
    assert "const live=true" in TestClient(create_app(FakeSource([]), FakeChatter(), BrowserCamera())).get("/").text
    assert "const live=false" in TestClient(create_app(FakeSource([]), FakeChatter())).get("/").text


def test_posted_frame_reaches_the_camera():
    camera = BrowserCamera()
    client = TestClient(create_app(FakeSource([]), FakeChatter(), camera))
    assert client.post("/api/frame", content=jpeg_bytes()).status_code == 204
    ok, frame = camera.read()
    assert ok and frame.shape == (24, 32, 3)


def test_frame_that_is_not_an_image_is_refused():
    client = TestClient(create_app(FakeSource([]), FakeChatter(), BrowserCamera()))
    assert client.post("/api/frame", content=b"not a jpeg").status_code == 400
    assert client.post("/api/frame", content=b"").status_code == 400


def test_stop_turns_away_whoever_is_sending_frames_for_a_moment(monkeypatch):
    client = TestClient(create_app(FakeSource([]), FakeChatter(), BrowserCamera()))
    assert client.post("/api/stop").status_code == 204
    assert client.post("/api/frame", content=jpeg_bytes()).status_code == 410
    assert client.get("/api/objects").json()["receiving"] is False
    monkeypatch.setattr(server, "BROWSER_IDLE_S", 0.0)
    client.post("/api/stop")
    assert client.post("/api/frame", content=jpeg_bytes()).status_code == 204


def test_no_frame_endpoint_without_a_browser_camera():
    client = TestClient(create_app(FakeSource([]), FakeChatter()))
    assert client.post("/api/frame", content=jpeg_bytes()).status_code in (404, 405)
