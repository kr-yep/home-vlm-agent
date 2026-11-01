from PIL import Image

from home_vlm.vlm import encode_image, parse_label


def test_parse_label_reads_json_and_converts_to_taiwan_traditional():
    label = parse_label('{"name": "马克杯", "name_en": "mug", "detail": "白色陶瓷", "confident": true}')
    assert label.name == "馬克杯"
    assert label.name_en == "mug"
    assert label.detail == "白色陶瓷"
    assert label.confident is True


def test_parse_label_finds_json_inside_fenced_text():
    label = parse_label('```json\n{"name": "遙控器", "name_en": "remote"}\n```')
    assert label.name == "遙控器"
    assert label.confident is False


def test_parse_label_falls_back_to_plain_text():
    label = parse_label("一支黑色手機\n其他說明")
    assert label.name == "一支黑色手機"
    assert label.name_en == ""


def test_encode_image_upscales_small_crops_and_caps_large_ones():
    import base64, io

    def decoded_size(url):
        return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).size

    assert min(decoded_size(encode_image(Image.new("RGB", (40, 80))))) == 224
    assert max(decoded_size(encode_image(Image.new("RGB", (2000, 1000))))) == 512
