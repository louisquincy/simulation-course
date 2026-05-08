from rng import MultiKongGen

_ITEMS_RAW = [
    ("Naruto — Seventh Hokage",           0.0020, "legendary"),
    ("Sasuke — Shadow Avenger",           0.0015, "legendary"),
    ("Sakura — Cherry Blossom",           0.0200, "epic"),
    ("Kakashi — Copy Ninja",              0.0150, "epic"),
    ("Kunai Blade",                       0.0800, "rare"),
    ("Shuriken Kill Effect",              0.0700, "rare"),
    ("Konoha Recall Effect",              0.0600, "rare"),
    ("Ramen Coupon",                      0.2500, "common"),
    ("Ninja Scroll Fragment",             0.2000, "common"),
    ("Konoha Emblem Border",              0.1500, "common"),
    ("Training Log Card",                 0.1515, "common"),
]

YES_PROB = 0.5

# NAMES   = [item[0] for item in _ITEMS_RAW]
# WEIGHTS = [item[1] for item in _ITEMS_RAW]
# RARITY  = {item[0]: item[2] for item in _ITEMS_RAW}

# def gacha_pull_once(rng: MultiKongGen) -> str:
#     u = rng.anabios()
#     cum = 0.0
#     for i in range(len(NAMES)):
#         cum += WEIGHTS[i]
#         if u < cum:
#             return NAMES[i]
#     return NAMES[-1]

def yes_no_once(rng: MultiKongGen) -> str:
    return "Yes" if rng.anabios() < YES_PROB else "No"

_raw_weights = [item[1] for item in _ITEMS_RAW]
_total_weight = sum(_raw_weights)
NAMES   = [item[0] for item in _ITEMS_RAW]
WEIGHTS = [w / _total_weight for w in _raw_weights]
RARITY  = {item[0]: item[2] for item in _ITEMS_RAW}

def gacha_pull_once(rng: MultiKongGen) -> str:
    """Вернуть название предмета согласно нормированным весам."""
    u = rng.anabios()
    cum = 0.0
    for name, w in zip(NAMES, WEIGHTS):
        cum += w
        if u < cum:
            return name
    return NAMES[-1]