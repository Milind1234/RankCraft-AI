"""
Pillar 1: Data Science Engine
Handles e-commerce catalog synthesis, realistic user interaction logs,
and tabular feature engineering.
"""
import json
import random
import math
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from src.config import (
    CATALOG_PATH,
    SEARCH_LOGS_PATH,
    FEATURE_METADATA_PATH,
    FEATURE_NAMES,
    RELEVANCE_LABELS
)

# Seed for reproducible high-quality catalog generation
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CATEGORIES = {
    "Footwear & Athletics": {
        "sub_categories": ["Running Shoes", "Trail Running Shoes", "Basketball Shoes", "Cross Trainers", "Walking Shoes", "Insoles & Socks"],
        "brands": ["Nike", "Adidas", "Hoka", "Brooks", "Asics", "New Balance", "Saucony", "On Cloud", "Salomon", "Under Armour"],
        "price_range": (45.0, 240.0),
        "typical_discount_chance": 0.55
    },
    "Audio & Acoustics": {
        "sub_categories": ["Over-Ear Headphones", "Wireless Earbuds", "Noise Cancelling", "Bluetooth Speakers", "Studio Monitors", "Microphones"],
        "brands": ["Sony", "Bose", "Apple", "Sennheiser", "Anker Soundcore", "JBL", "Audio-Technica", "Beats", "Marshall", "Shure"],
        "price_range": (35.0, 399.0),
        "typical_discount_chance": 0.45
    },
    "Computing & Ergonomics": {
        "sub_categories": ["Mechanical Keyboards", "Ergonomic Mice", "Laptops & Tablets", "4K Monitors", "USB-C Docks", "Monitor Arms", "Webcams"],
        "brands": ["Logitech", "Keychron", "Apple", "Dell", "Razer", "Corsair", "Anker", "LG", "CalDigit", "Ergotron"],
        "price_range": (29.0, 550.0),
        "typical_discount_chance": 0.40
    },
    "Office Furniture & Wellness": {
        "sub_categories": ["Ergonomic Chairs", "Standing Desks", "Desk Converters", "Lumbar Support", "Footrests", "Desk Mats"],
        "brands": ["Herman Miller", "Steelcase", "Autonomous", "FLEXISPOT", "Uplift", "Branch", "Vari", "Everlasting Comfort"],
        "price_range": (25.0, 850.0),
        "typical_discount_chance": 0.35
    },
    "Apparel & Outdoor": {
        "sub_categories": ["Running Jackets", "Hoodies & Sweatshirts", "Moisture-Wicking Tees", "Trail Pants", "Thermal Baselayers", "Backpacks"],
        "brands": ["Patagonia", "Arc'teryx", "Nike", "Lululemon", "The North Face", "Columbia", "Gymshark", "Smartwool", "Osprey"],
        "price_range": (22.0, 320.0),
        "typical_discount_chance": 0.50
    },
    "Fitness Gear & Hydration": {
        "sub_categories": ["Insulated Water Bottles", "Fitness Smartwatches", "Yoga Mats", "Resistance Bands", "Foam Rollers", "Gym Bags"],
        "brands": ["Hydro Flask", "Garmin", "Yeti", "Apple", "Manduka", "Lululemon", "Theragun", "Iron Gym", "Fitbit", "Stanley"],
        "price_range": (18.0, 450.0),
        "typical_discount_chance": 0.45
    }
}

PRODUCT_TEMPLATES = [
    # Footwear templates
    {
        "category": "Footwear & Athletics",
        "templates": [
            ("{brand} {model} Lightweight Running Shoes with Responsive Arch Support",
             "Designed for high-mileage road runners who need dynamic arch stability, lightweight cushioning, and breathable engineered mesh.",
             ["Engineered mesh breathable upper", "Dual-density responsive foam midsole", "Targeted medial arch support arch-band", "Abrasion-resistant rubber outsole", "Weight: 8.4 oz"]),
            ("{brand} {model} Waterproof Trail Running Shoes with Vibram Megagrip",
             "Conquer technical trails, mud, and rain with seam-sealed waterproof protection and aggressive multidirectional traction lugs.",
             ["GORE-TEX waterproof breathable membrane", "Vibram Megagrip lugged outsole", "Protective rock plate underfoot", "Reinforced toe cap shield", "Quick-lace retention system"]),
            ("{brand} {model} Max-Cushion Road Marathon Racing Shoes",
             "Engineered for ultimate energy return and effortless forward roll during half and full marathons.",
             ["Full-length carbon fiber propulsion plate", "Supercritical nitrogen-infused foam", "Ultralight single-layer mesh", "Segmented crash pad", "Drop: 8mm"]),
            ("{brand} {model} Cushion Orthotic Walking Shoes with Extra Wide Toe Box",
             "Relieve plantar fasciitis and foot fatigue with orthopedic arch support and ergonomic anatomical footbed.",
             ["Podiatrist-designed anatomical arch support", "Wide toe box eliminates pinching", "Shock-absorbing gel heel cup", "Non-slip certified outsole", "Padded collar"]),
            ("{brand} {model} Zero Drop Minimalist Barefoot Cross Training Shoes",
             "Natural foot movement and ground feedback for deadlifts, squats, sprint training, and functional fitness.",
             ["Zero-drop heel-to-toe offset", "Wide anatomical toe splay design", "Sticky rubber flexible outsole", "Breathable vegan microfiber", "Removable 2mm insole"]),
        ],
        "models": ["Pegasus Air 40", "Ghost Max 16", "Clifton 9", "Cloudmonster 2", "Speedcross 6", "Fresh Foam 1080", "Endorphin Speed 4", "Kayano 31", "Metaspeed Sky", "Infinity Run 4"]
    },
    # Audio templates
    {
        "category": "Audio & Acoustics",
        "templates": [
            ("{brand} {model} Wireless Over-Ear Active Noise Cancelling Headphones",
             "Industry-leading adaptive noise cancellation, studio-grade 40mm acoustic drivers, and 35-hour battery life for immersive travel and focus.",
             ["Multi-microphone hybrid ANC", "High-Resolution Audio certified LDAC/AAC", "35-hour battery with fast USB-C charge", "Memory foam plush earcups", "Multipoint Bluetooth 5.3"]),
            ("{brand} {model} True Wireless Earbuds with Transparency Mode and Mic",
             "Compact ergonomic earbuds featuring crystal-clear beamforming call microphones, active noise cancellation, and IPX7 sweat resistance.",
             ["Active Noise Cancellation with Transparency", "6 beamforming microphones with wind reduction", "IPX7 waterproof and sweatproof", "Wireless Qi charging case", "32-hour total playtime"]),
            ("{brand} {model} Portable Waterproof Bluetooth Speaker with Deep Punchy Bass",
             "Take powerful 360-degree sound and booming bass anywhere with rugged shockproof waterproof construction and 20-hour battery life.",
             ["Dual passive radiators for deep bass", "IP67 dust and waterproof rating", "20-hour continuous playback", "PartyBoost pairing support", "Durable rugged fabric shell"]),
            ("{brand} {model} USB-C Studio Condenser Microphone for Podcasting & Streaming",
             "Broadcast-quality audio capture with cardioid pickup pattern, zero-latency headphone monitoring, and built-in pop filter.",
             ["24-bit/96kHz professional studio sampling", "Cardioid and omnidirectional polar patterns", "Zero-latency 3.5mm monitor jack", "Internal shock mount & pop filter", "Tap-to-mute capacitive sensor"]),
        ],
        "models": ["QuietComfort Ultra", "WH-1000XM5", "AirPods Max Pro", "Momentum 4", "Space Q45", "Live 770NC", "Flip 6 Pro", "Boom 3 Max", "Yeti Studio X", "Wave Neo"]
    },
    # Computing & Ergonomics
    {
        "category": "Computing & Ergonomics",
        "templates": [
            ("{brand} {model} Wireless Ergonomic Mechanical Keyboard with Mac Layout",
             "Split ergonomic key layout, tactile hot-swappable mechanical switches, dedicated Mac/Win keys, and multi-device Bluetooth switching.",
             ["Split curved ergonomic keyframe", "Hot-swappable Gateron silent tactile switches", "Triple mode: Bluetooth 5.1 / 2.4G / USB-C", "Cushioned memory foam wrist rest", "Full Mac & Windows compatibility"]),
            ("{brand} {model} Advanced Wireless Vertical Ergonomic Mouse with Thumb Rest",
             "Promote a natural handshake posture that reduces forearm strain and wrist pressure by 57% during long working hours.",
             ["Natural 57-degree vertical handshake angle", "4000 DPI high-precision sensor with glass tracking", "Thumb scroll wheel and customizable buttons", "USB-C rechargeable 70-day battery", "Multi-computer Flow cross-control"]),
            ("{brand} {model} 27-inch 4K UHD IPS Ergonomic Eye-Care Monitor",
             "Stunning 3840x2160 resolution with 99% sRGB color accuracy, 90W USB-C power delivery, and fully adjustable pivot stand.",
             ["4K UHD IPS panel (3840 x 2160)", "USB-C 90W single-cable laptop charging", "TÜV certified flicker-free low blue light", "Height, tilt, swivel, pivot adjustable stand", "Built-in dual stereo speakers"]),
            ("{brand} {model} 14-in-1 Thunderbolt 4 Universal Laptop Docking Station",
             "Transform your laptop into an ultimate workstation with dual 4K/single 8K display support, 100W PD charging, and Gigabit Ethernet.",
             ["Dual 4K @ 60Hz or Single 8K display support", "100W Power Delivery charging to host", "Thunderbolt 4 / USB4 40Gbps bandwidth", "Gigabit Ethernet, SD/TF card slots, audio", "Solid aluminum alloy heat dissipation"]),
        ],
        "models": ["Master MX Ergo", "Pro Wireless K8", "UltraDock Elite", "ViewSonic Pro 4K", "ErgoSplit Q11", "Craft Touch Pro", "Vertical Lift 3", "ThunderHub 4X", "Precision Pro 27"]
    },
    # Office Furniture & Wellness
    {
        "category": "Office Furniture & Wellness",
        "templates": [
            ("{brand} {model} Ergonomic Mesh Office Chair with Dynamic Lumbar Support",
             "Engineered for all-day comfort with breathable Pellicle mesh, 4D adjustable armrests, and forward/recline tilt posture lock.",
             ["Self-adjusting dynamic lumbar curve", "Breathable responsive elastomeric mesh", "4D armrests (height, angle, depth, width)", "Synchronous tilt mechanism with 4 tilt locks", "Weight capacity: 330 lbs"]),
            ("{brand} {model} Dual-Motor Electric Standing Desk with Memory Presets",
             "Smooth, ultra-quiet height adjustment from 25 to 50 inches with anti-collision sensor and solid eco-friendly wood desktop.",
             ["Dual high-torque whisper-quiet motors (<45dB)", "4 programmable digital memory height presets", "Smart anti-collision safety bounce system", "Heavy-duty steel 3-stage lifting columns", "Weight capacity: 275 lbs"]),
            ("{brand} {model} Dual Monitor Heavy Duty Gas Spring Aluminum Arm",
             "Reclaim desk space with fully articulating gas spring arms supporting dual screens from 17 to 34 inches with integrated cable channels.",
             ["Gas spring counterbalance tension system", "Supports dual monitors up to 22 lbs each", "360-degree rotation, 90-degree tilt, 180 swivel", "C-clamp and grommet mounting hardware included", "Hidden cable management channels"]),
        ],
        "models": ["Aeron Pro Elite", "Gesture Comfort", "ProLine Dual Desk", "ActiveCurve Ergonomist", "ErgoFlex Matrix", "Zenith MotorDesk", "Freedom Executive", "AeroMesh 360"]
    },
    # Apparel & Outdoor
    {
        "category": "Apparel & Outdoor",
        "templates": [
            ("{brand} {model} Breathable Lightweight Packable Running Rain Jacket",
             "Ultralight windproof and water-repellent jacket with underarm laser-cut ventilation and reflective 360 safety strips for night runs.",
             ["DWR water-repellent windproof ripstop fabric", "Laser-cut zoned breathability vents", "Packs into its own zippered chest pocket", "360-degree high-visibility reflective elements", "Weight: 4.6 oz"]),
            ("{brand} {model} Heavyweight Organic Cotton Oversized Hoodie",
             "Premium 450 GSM combed organic French terry cotton with structured double-layered hood, ribbed side gussets, and relaxed drape.",
             ["100% GOTS certified organic heavyweight cotton (450 GSM)", "Pre-shrunk double-layered architectural hood", "Reinforced rib side panels for mobility", "Kangaroo pocket with hidden phone sleeve", "Garment dyed for vintage soft texture"]),
            ("{brand} {model} Quick-Dry Trail Running Hiking Pants with Stretch",
             "Durable 4-way stretch ripstop pants with articulated knees, water-resistant finish, and secure zippered cargo pockets.",
             ["4-way stretch abrasion-resistant nylon blend", "DWR moisture and dirt repellent coating", "Articulated knees for unrestricted stride", "5 zippered pockets including phone holster", "Adjustable integrated webbing belt"]),
            ("{brand} {model} Merino Wool Anti-Blister Cushioned Running Socks (3-Pack)",
             "Seamless toe closure and high-density terry heel cushioning that wicks sweat and prevents hotspots on ultra-distance runs.",
             ["Fine gauge Merino wool blend naturally resists odor", "Targeted anatomical arch compression band", "Seamless toe prevents rubbing and blisters", "Reinforced heel and toe wear zones", "Breathable mesh ventilation top"]),
        ],
        "models": ["Torrentshell Trail", "AeroShield Wind", "EcoHeavyweight Terry", "Beta LT Ultralight", "SpeedVent Pro", "Alpinist Stretch", "SwiftGrip Merino", "EnduroShield 3"]
    },
    # Fitness Gear & Hydration
    {
        "category": "Fitness Gear & Hydration",
        "templates": [
            ("{brand} {model} Insulated Stainless Steel Water Bottle with Straw Lid 32oz",
             "Double-wall vacuum insulation keeps ice water frosty for 24 hours or coffee hot for 12 hours without condensation or metallic taste.",
             ["Pro-grade 18/8 food-safe stainless steel", "Double-wall vacuum insulation (24h cold / 12h hot)", "Leak-proof flex straw lid with carry handle", "BPA-free and dishwasher safe", "Fits standard cup holders and bottle cages"]),
            ("{brand} {model} GPS Multisport Fitness Smartwatch with Heart Rate & SpO2",
             "Track 120+ sport profiles, heart rate variability, VO2 max, sleep recovery scores, and offline route navigation with 14-day battery.",
             ["Always-on Sunlight-readable AMOLED display", "Wrist-based Gen 4 heart rate and pulse oximeter", "Dual-frequency multi-GNSS satellite positioning", "Up to 14 days battery life in smartwatch mode", "5 ATM water resistance (swim proof)"]),
            ("{brand} {model} Extra Thick 6mm Non-Slip Alignment Yoga Mat",
             "High-density eco-friendly TPE foam provides joint cushioning, laser-etched alignment guides, and sweat-activated grip.",
             ["6mm high-density cushioning protects knees and spine", "Laser-engraved body alignment lines", "Textured dual-sided non-slip surface", "Eco-friendly recyclable odorless TPE material", "Includes carrying strap and sling bag"]),
            ("{brand} {model} Heavy Duty Fabric Resistance Loop Bands Set (5 Resistance Levels)",
             "Non-rolling thick fabric bands with inner anti-slip latex grips for glute activation, strength training, and physical therapy.",
             ["Durable fabric blend that never rolls, pinches, or snaps", "Inner non-slip grip strips keep band in place", "5 progressive resistance levels (15 to 65 lbs)", "Reinforced multi-row stitching", "Includes breathable mesh carry pouch and workout guide"]),
        ],
        "models": ["HydroPro 32", "Forerunner 265", "Summit Cold 36", "UltraFit Tracker", "AuraGrip Eco", "PowerLoop Pro", "Endurance Pulse", "HydraChug 40"]
    }
]

# Query bank for training and evaluating LTR
SAMPLE_QUERIES = [
    # Query, Target Category, Brand Intent, Attribute Intent, Max Price
    ("lightweight running shoes with arch support", "Footwear & Athletics", None, ["lightweight", "running", "arch support"], None),
    ("waterproof trail running shoes under 120", "Footwear & Athletics", None, ["waterproof", "trail", "running"], 120.0),
    ("nike lightweight running sneakers", "Footwear & Athletics", "Nike", ["lightweight", "running"], None),
    ("orthopedic walking shoes wide toe box", "Footwear & Athletics", None, ["orthopedic", "walking", "wide toe"], None),
    ("zero drop barefoot cross training shoes", "Footwear & Athletics", None, ["zero drop", "barefoot", "training"], None),
    ("noise cancelling headphones for travel", "Audio & Acoustics", None, ["noise cancelling", "headphones", "wireless"], None),
    ("sony wireless noise cancelling headphones", "Audio & Acoustics", "Sony", ["wireless", "noise cancelling", "headphones"], None),
    ("portable waterproof bluetooth speaker deep bass", "Audio & Acoustics", None, ["portable", "waterproof", "bluetooth", "speaker", "bass"], None),
    ("true wireless earbuds with transparency mode", "Audio & Acoustics", None, ["wireless", "earbuds", "transparency"], None),
    ("usb-c studio condenser microphone podcasting", "Audio & Acoustics", None, ["usb-c", "microphone", "podcasting"], None),
    ("wireless ergonomic keyboard for mac", "Computing & Ergonomics", None, ["wireless", "ergonomic", "keyboard", "mac"], None),
    ("vertical ergonomic mouse wrist pain", "Computing & Ergonomics", None, ["vertical", "ergonomic", "mouse"], None),
    ("27 inch 4k monitor usb-c charging", "Computing & Ergonomics", None, ["4k", "monitor", "usb-c"], None),
    ("thunderbolt 4 docking station dual 4k", "Computing & Ergonomics", None, ["thunderbolt 4", "docking station", "dual 4k"], None),
    ("ergonomic mesh office chair lumbar support", "Office Furniture & Wellness", None, ["ergonomic", "mesh", "office chair", "lumbar support"], None),
    ("dual motor electric standing desk", "Office Furniture & Wellness", None, ["standing desk", "electric", "dual motor"], None),
    ("dual monitor arm gas spring desk mount", "Office Furniture & Wellness", None, ["monitor arm", "gas spring", "desk mount"], None),
    ("packable waterproof running rain jacket", "Apparel & Outdoor", None, ["waterproof", "running", "jacket", "packable"], None),
    ("heavyweight organic cotton oversized hoodie", "Apparel & Outdoor", None, ["heavyweight", "organic cotton", "hoodie", "oversized"], None),
    ("quick dry trail hiking pants stretch", "Apparel & Outdoor", None, ["quick dry", "hiking pants", "stretch"], None),
    ("merino wool running socks anti blister", "Apparel & Outdoor", None, ["merino wool", "running socks", "anti blister"], None),
    ("insulated stainless steel water bottle 32oz", "Fitness Gear & Hydration", None, ["insulated", "stainless steel", "water bottle", "32oz"], None),
    ("gps multisport fitness smartwatch vo2 max", "Fitness Gear & Hydration", None, ["gps", "fitness smartwatch", "heart rate"], None),
    ("thick non slip alignment yoga mat", "Fitness Gear & Hydration", None, ["thick", "non slip", "yoga mat"], None),
    ("heavy duty fabric resistance bands set", "Fitness Gear & Hydration", None, ["fabric resistance bands", "heavy duty"], None),
    # Typo variations to exercise NLP query cleaning
    ("lighweight runing shoes with arch suport", "Footwear & Athletics", None, ["lightweight", "running", "arch support"], None),
    ("nikke lightweight runing shoes", "Footwear & Athletics", "Nike", ["lightweight", "running"], None),
    ("wirless nois cancelling hedphones", "Audio & Acoustics", None, ["wireless", "noise cancelling", "headphones"], None),
    ("sonny blutooth noise cancelling", "Audio & Acoustics", "Sony", ["bluetooth", "noise cancelling"], None),
    ("ergonmic keybord mac wireless", "Computing & Ergonomics", None, ["ergonomic", "keyboard", "mac", "wireless"], None),
    ("standing desk electrick dual motor", "Office Furniture & Wellness", None, ["standing desk", "electric"], None),
    ("organik cotton hoodie oversize", "Apparel & Outdoor", None, ["organic cotton", "hoodie", "oversize"], None),
    ("water botle insulated straw 32 oz", "Fitness Gear & Hydration", None, ["water bottle", "insulated"], None)
]


class DataEngine:
    """Manages e-commerce catalog, session logs, and tabular feature engineering."""

    def __init__(self):
        self.catalog: List[Dict[str, Any]] = []
        self.category_stats: Dict[str, Dict[str, float]] = {}
        self.search_logs: List[Dict[str, Any]] = []
        self.catalog_by_id: Dict[str, Dict[str, Any]] = {}

    def generate_or_load_catalog(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Generates or loads realistic e-commerce product catalog."""
        if CATALOG_PATH.exists() and not force_refresh:
            with open(CATALOG_PATH, "r", encoding="utf-8") as f:
                self.catalog = json.load(f)
            self.catalog_by_id = {p["id"]: p for p in self.catalog}
            self._compute_category_stats()
            return self.catalog

        print("[DataEngine] Synthesizing realistic 1,200+ product catalog...")
        products = []
        prod_idx = 1

        for cat_info in PRODUCT_TEMPLATES:
            category = cat_info["category"]
            cat_meta = CATEGORIES[category]
            brands = cat_meta["brands"]
            min_price, max_price = cat_meta["price_range"]
            discount_chance = cat_meta["typical_discount_chance"]

            for template_title, template_desc, template_feats in cat_info["templates"]:
                for brand in brands:
                    for model in cat_info["models"]:
                        # Introduce realistic variations
                        base_price = round(random.uniform(min_price, max_price), 2)
                        has_discount = random.random() < discount_chance
                        if has_discount:
                            discount_pct = round(random.uniform(0.10, 0.45), 2)
                            original_price = round(base_price / (1.0 - discount_pct), 2)
                        else:
                            discount_pct = 0.0
                            original_price = base_price

                        title = template_title.format(brand=brand, model=model)
                        desc = f"{brand} presents {model}. {template_desc}"

                        # Realistic behavioral distributions
                        rating = round(random.betavariate(16, 2) * 2.0 + 3.0, 1)  # skewed towards 4.0 - 4.9
                        rating = min(5.0, max(3.0, rating))
                        review_count = int(random.lognormvariate(5.5, 1.2)) + 12

                        # Historical impressions, clicks, conversions
                        # Higher rating & discount correlates with higher CTR
                        latent_ctr_base = 0.035 + (rating - 3.5) * 0.015 + discount_pct * 0.02
                        impressions = random.randint(5000, 150000)
                        clicks = int(impressions * max(0.01, min(0.12, random.gauss(latent_ctr_base, 0.008))))
                        cvr_base = 0.08 + (rating - 3.5) * 0.04
                        purchases = int(clicks * max(0.02, min(0.25, random.gauss(cvr_base, 0.015))))

                        return_rate = round(max(0.01, min(0.15, random.betavariate(2, 20))), 3)
                        freshness_days = random.randint(5, 360)
                        is_prime = random.random() < 0.85

                        # Visual image tag placeholder for luxury frontend
                        sub_cat = cat_meta["sub_categories"][prod_idx % len(cat_meta["sub_categories"])]
                        image_tag = sub_cat.lower().replace(" ", "_")

                        product = {
                            "id": f"prod_{prod_idx:04d}",
                            "title": title,
                            "brand": brand,
                            "category": category,
                            "sub_category": sub_cat,
                            "price": base_price,
                            "original_price": original_price,
                            "discount_pct": discount_pct,
                            "rating": rating,
                            "review_count": review_count,
                            "features": template_feats,
                            "description": desc,
                            "historical_impressions": impressions,
                            "historical_clicks": clicks,
                            "historical_purchases": purchases,
                            "return_rate": return_rate,
                            "freshness_days": freshness_days,
                            "is_prime_eligible": is_prime,
                            "image_tag": image_tag,
                            "semantic_text": f"{brand} {title}. Category: {category}. Features: {'; '.join(template_feats)}. {desc}"
                        }
                        products.append(product)
                        prod_idx += 1

        self.catalog = products
        self.catalog_by_id = {p["id"]: p for p in self.catalog}
        print(f"[DataEngine] Generated {len(self.catalog)} products.")

        # Save to disk
        CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CATALOG_PATH, "w", encoding="utf-8") as f:
            json.dump(self.catalog, f, indent=2)

        self._compute_category_stats()
        return self.catalog

    def _compute_category_stats(self):
        """Computes statistical baselines per category for feature engineering."""
        cat_prices = {}
        for p in self.catalog:
            cat = p["category"]
            cat_prices.setdefault(cat, []).append(p["price"])

        self.category_stats = {}
        for cat, prices in cat_prices.items():
            arr = np.array(prices)
            self.category_stats[cat] = {
                "mean_price": float(np.mean(arr)),
                "median_price": float(np.median(arr)),
                "std_price": float(np.std(arr) + 1e-5),
                "min_price": float(np.min(arr)),
                "max_price": float(np.max(arr))
            }

        FEATURE_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(FEATURE_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "category_stats": self.category_stats,
                "feature_names": FEATURE_NAMES
            }, f, indent=2)

    def extract_tabular_features(
        self,
        product: Dict[str, Any],
        query_nlp: Dict[str, Any],
        semantic_score: float,
        bm25_score: float
    ) -> Dict[str, float]:
        """
        Pillar 1: Tabular feature engineering combining catalog metadata,
        historical behavioral signals, NLP query match, and deep semantic scores.
        """
        cat = product.get("category", "")
        cat_stats = self.category_stats.get(cat, {
            "mean_price": 100.0,
            "median_price": 85.0,
            "std_price": 40.0,
            "min_price": 20.0,
            "max_price": 300.0
        })

        price = product.get("price", 50.0)
        mean_p = cat_stats["mean_price"]
        med_p = cat_stats["median_price"]
        min_p = cat_stats["min_price"]
        max_p = cat_stats["max_price"]

        price_norm = (price - min_p) / (max_p - min_p + 1e-5)
        price_diff_mean = (price - mean_p) / (mean_p + 1e-5)
        price_diff_median = (price - med_p) / (med_p + 1e-5)
        discount_pct = product.get("discount_pct", 0.0)
        rating = product.get("rating", 4.0)
        review_count = product.get("review_count", 50)
        log_reviews = math.log1p(review_count)

        # Smoothed empirical Bayes historical CTR & CVR
        impr = product.get("historical_impressions", 1000)
        clicks = product.get("historical_clicks", 40)
        purchases = product.get("historical_purchases", 4)
        smoothed_ctr = (clicks + 5.0) / (impr + 150.0)
        smoothed_cvr = (purchases + 1.0) / (clicks + 15.0)

        # NLP match signals
        product_brand = product.get("brand", "").lower()
        query_brand = (query_nlp.get("detected_brand") or "").lower()
        brand_match = 1.0 if (query_brand and query_brand in product_brand) else 0.0

        product_cat = cat.lower()
        query_cat = (query_nlp.get("detected_category") or "").lower()
        cat_match = 1.0 if (query_cat and (query_cat in product_cat or product_cat in query_cat)) else 0.0

        # Substring / exact query match in title
        raw_query = query_nlp.get("cleaned_query", "").lower()
        title_lower = product.get("title", "").lower()
        exact_in_title = 1.0 if (raw_query and raw_query in title_lower) else 0.0

        # Jaccard overlap between query tokens and product title
        q_tokens = set(raw_query.split())
        t_tokens = set(title_lower.split())
        jaccard = len(q_tokens & t_tokens) / (len(q_tokens | t_tokens) + 1e-5)

        is_prime = 1.0 if product.get("is_prime_eligible", True) else 0.0
        freshness_days = product.get("freshness_days", 60)
        freshness_score = math.exp(-0.005 * freshness_days)
        return_rate = product.get("return_rate", 0.05)

        features = {
            "semantic_similarity": float(semantic_score),
            "bm25_score": float(bm25_score),
            "jaccard_overlap": float(jaccard),
            "brand_match": float(brand_match),
            "category_match": float(cat_match),
            "exact_query_in_title": float(exact_in_title),
            "price_normalized": float(price_norm),
            "price_diff_to_cat_mean": float(price_diff_mean),
            "price_diff_to_cat_median": float(price_diff_median),
            "discount_pct": float(discount_pct),
            "rating": float(rating),
            "log_review_count": float(log_reviews),
            "historical_ctr": float(smoothed_ctr),
            "historical_cvr": float(smoothed_cvr),
            "is_prime_eligible": float(is_prime),
            "freshness_score": float(freshness_score),
            "return_rate": float(return_rate)
        }
        return features

    def generate_search_logs(
        self,
        nlp_cleaner,
        deep_engine,
        bm25_retriever,
        force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generates realistic search interaction session logs across diverse queries.
        Each session contains candidate items with multi-level relevance ground truth:
          0 = Impression (not clicked)
          1 = Clicked
          2 = Added to Cart
          3 = Purchased
        """
        if SEARCH_LOGS_PATH.exists() and not force_refresh:
            with open(SEARCH_LOGS_PATH, "r", encoding="utf-8") as f:
                self.search_logs = json.load(f)
            return self.search_logs

        print(f"[DataEngine] Generating realistic search interaction sessions across {len(SAMPLE_QUERIES)} queries...")
        sessions = []

        for q_idx, query_tuple in enumerate(SAMPLE_QUERIES):
            raw_query, target_cat, brand_target, target_attrs, max_price = query_tuple
            cleaned_query, nlp_info = nlp_cleaner.clean_and_extract_intent(raw_query)

            # Retrieve top 80 candidates using BM25 + category filtering to simulate candidate retrieval stage
            bm25_candidates = bm25_retriever.retrieve(cleaned_query, top_k=80)
            candidate_ids = [c["id"] for c in bm25_candidates]

            # Ensure some products from target category are included
            target_cat_prods = [p for p in self.catalog if p["category"] == target_cat]
            sampled_target = random.sample(target_cat_prods, min(25, len(target_cat_prods)))
            for sp in sampled_target:
                if sp["id"] not in candidate_ids:
                    candidate_ids.append(sp["id"])

            candidate_products = [self.catalog_by_id[cid] for cid in candidate_ids if cid in self.catalog_by_id]

            # Compute semantic embeddings score for candidates
            semantic_scores = deep_engine.compute_similarity(cleaned_query, candidate_products)

            session_records = []
            for prod, sem_score in zip(candidate_products, semantic_scores):
                bm25_score = next((c["bm25_score"] for c in bm25_candidates if c["id"] == prod["id"]), 0.0)

                # Determine industrial multi-level ground truth label based on:
                # 1. Semantic match to query
                # 2. Category correctness
                # 3. Brand match (if query targeted brand)
                # 4. Attribute coverage in product title/features
                # 5. Price & rating satisfaction
                relevance_grade = 0
                cat_match = prod["category"] == target_cat
                brand_match = brand_target.lower() in prod["brand"].lower() if brand_target else True
                title_lower = prod["title"].lower()
                desc_lower = prod["description"].lower()

                attr_matches = sum(1 for attr in target_attrs if attr in title_lower or attr in desc_lower)
                attr_ratio = attr_matches / max(1, len(target_attrs))

                price_ok = True
                if max_price and prod["price"] > max_price:
                    price_ok = False

                if cat_match and brand_match and attr_ratio >= 0.70 and sem_score > 0.45 and price_ok:
                    # High relevance candidate: potential click/cart/purchase
                    p_click = 0.85
                    p_cart = 0.55 if prod["rating"] >= 4.2 else 0.30
                    p_purchase = 0.45 if prod["discount_pct"] > 0.10 and prod["rating"] >= 4.4 else 0.20

                    if random.random() < p_purchase:
                        relevance_grade = 3  # Purchase
                    elif random.random() < p_cart:
                        relevance_grade = 2  # Cart
                    elif random.random() < p_click:
                        relevance_grade = 1  # Click
                    else:
                        relevance_grade = 0
                elif cat_match and (attr_ratio >= 0.35 or sem_score > 0.35):
                    # Medium relevance: occasional click
                    relevance_grade = 1 if random.random() < 0.40 else 0
                else:
                    # Irrelevant or poor match: mostly skipped impression
                    relevance_grade = 1 if random.random() < 0.05 else 0

                feat_dict = self.extract_tabular_features(prod, nlp_info, sem_score, bm25_score)

                session_records.append({
                    "product_id": prod["id"],
                    "relevance_grade": relevance_grade,
                    "features": feat_dict,
                    "semantic_score": float(sem_score),
                    "bm25_score": float(bm25_score)
                })

            sessions.append({
                "session_id": f"sess_{q_idx:03d}",
                "query": raw_query,
                "cleaned_query": cleaned_query,
                "nlp_info": nlp_info,
                "records": session_records
            })

        self.search_logs = sessions
        with open(SEARCH_LOGS_PATH, "w", encoding="utf-8") as f:
            json.dump(self.search_logs, f, indent=2)
        print(f"[DataEngine] Generated and saved {len(self.search_logs)} search sessions.")
        return self.search_logs
