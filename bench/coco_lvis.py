"""Which LVIS categories each COCO class (what YOLO knows) stands for.

Hand-made; `prepare_lvis.py` refuses to run if a name here is not an LVIS
category, so typos cannot silently shrink the COCO split.
"""

COCO_TO_LVIS: dict[str, set[str]] = {
    "person": {"person"},
    "bicycle": {"bicycle"},
    "car": {"car_(automobile)", "police_cruiser", "race_car", "cab_(taxi)", "jeep", "minivan"},
    "motorcycle": {"motorcycle", "motor_scooter"},
    "airplane": {"airplane", "jet_plane"},
    "bus": {"bus_(vehicle)", "school_bus"},
    "train": {"train_(railroad_vehicle)", "passenger_car_(part_of_a_train)", "railcar_(part_of_a_train)"},
    "truck": {"truck", "pickup_truck", "trailer_truck", "garbage_truck", "tow_truck", "fire_engine"},
    "boat": {"boat", "kayak", "canoe", "yacht", "ferry", "houseboat"},
    "traffic light": {"traffic_light"},
    "fire hydrant": {"fireplug"},
    "stop sign": {"stop_sign"},
    "parking meter": {"parking_meter"},
    "bench": {"bench"},
    "bird": {"bird", "seabird", "duck", "pigeon", "parrot", "goose", "gull"},
    "cat": {"cat", "kitten"},
    "dog": {"dog", "puppy"},
    "horse": {"horse", "pony"},
    "sheep": {"sheep", "lamb_(animal)"},
    "cow": {"cow", "calf", "bull"},
    "elephant": {"elephant"},
    "bear": {"bear", "polar_bear", "grizzly"},
    "zebra": {"zebra"},
    "giraffe": {"giraffe"},
    "backpack": {"backpack"},
    "umbrella": {"umbrella"},
    "handbag": {"handbag", "shoulder_bag", "clutch_bag"},
    "tie": {"necktie", "bow-tie"},
    "suitcase": {"suitcase"},
    "frisbee": {"frisbee"},
    "skis": {"ski"},
    "snowboard": {"snowboard"},
    "sports ball": {"ball", "soccer_ball", "tennis_ball", "baseball", "basketball", "football_(American)", "volleyball", "softball", "beachball"},
    "kite": {"kite"},
    "baseball bat": {"baseball_bat"},
    "baseball glove": {"baseball_glove"},
    "skateboard": {"skateboard"},
    "surfboard": {"surfboard"},
    "tennis racket": {"tennis_racket"},
    "bottle": {"bottle", "beer_bottle", "water_bottle", "wine_bottle"},
    "wine glass": {"wineglass"},
    "cup": {"cup", "mug", "Dixie_cup", "teacup"},
    "fork": {"fork"},
    "knife": {"knife", "steak_knife"},
    "spoon": {"spoon", "wooden_spoon"},
    "bowl": {"bowl", "soup_bowl", "sugar_bowl"},
    "banana": {"banana"},
    "apple": {"apple"},
    "sandwich": {"sandwich", "hamburger"},
    "orange": {"orange_(fruit)", "mandarin_orange"},
    "broccoli": {"broccoli"},
    "carrot": {"carrot"},
    "hot dog": set(),  # LVIS has no hot dog category
    "pizza": {"pizza"},
    "donut": {"doughnut"},
    "cake": {"cake", "cupcake", "wedding_cake"},
    "chair": {"chair", "armchair", "folding_chair", "highchair", "rocking_chair", "deck_chair", "chaise_longue"},
    "couch": {"sofa", "loveseat", "sofa_bed"},
    "potted plant": {"flowerpot"},
    "bed": {"bed", "bunk_bed"},
    "dining table": {"dining_table", "table", "kitchen_table", "coffee_table"},
    "toilet": {"toilet"},
    "tv": {"television_set", "monitor_(computer_equipment) computer_monitor"},
    "laptop": {"laptop_computer"},
    "mouse": {"mouse_(computer_equipment)"},
    "remote": {"remote_control"},
    "keyboard": {"computer_keyboard"},
    "cell phone": {"cellular_telephone"},
    "microwave": {"microwave_oven"},
    "oven": {"oven", "stove"},
    "toaster": {"toaster", "toaster_oven"},
    "sink": {"sink"},
    "refrigerator": {"refrigerator"},
    "book": {"book", "notebook", "booklet", "hardback_book", "paperback_book", "comic_book"},
    "clock": {"clock", "wall_clock", "alarm_clock", "clock_tower"},
    "vase": {"vase"},
    "scissors": {"scissors"},
    "teddy bear": {"teddy_bear"},
    "hair drier": {"hair_dryer"},
    "toothbrush": {"toothbrush"},
}

# Images that contain one of these are treated as indoor/home scenes.
INDOOR_ANCHORS = {
    "dining_table", "sofa", "bed", "television_set", "refrigerator", "oven",
    "sink", "toilet", "microwave_oven", "bookcase", "kitchen_table", "armchair",
}


def coco_split(lvis_name: str) -> str:
    return "coco" if any(lvis_name in names for names in COCO_TO_LVIS.values()) else "non_coco"


def yolo_correct(yolo_class: str | None, lvis_name: str) -> bool:
    return yolo_class is not None and lvis_name in COCO_TO_LVIS.get(yolo_class, set())
