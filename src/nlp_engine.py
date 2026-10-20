"""
Pillar 2: NLP & Query Intelligence Engine
Handles search query cleaning, typo correction, intent extraction
(Brand vs Attribute vs Category vs Price constraints), text normalization,
and BM25 lexical candidate retrieval.
"""
import re
from typing import List, Dict, Any, Tuple, Optional
from rank_bm25 import BM25Okapi

# Known e-commerce vocabulary mapping for typo correction
COMMON_TYPOS = {
    "runing": "running",
    "lighweight": "lightweight",
    "lightweigth": "lightweight",
    "suport": "support",
    "archsuport": "arch support",
    "wirless": "wireless",
    "wireles": "wireless",
    "nois": "noise",
    "canselling": "cancelling",
    "canceling": "cancelling",
    "hedphones": "headphones",
    "headfones": "headphones",
    "blutooth": "bluetooth",
    "bleutooth": "bluetooth",
    "keybord": "keyboard",
    "keybaord": "keyboard",
    "ergonmic": "ergonomic",
    "ergonimic": "ergonomic",
    "nikke": "nike",
    "sonny": "sony",
    "aple": "apple",
    "electrick": "electric",
    "organik": "organic",
    "botle": "bottle",
    "watter": "water",
    "pant": "pants",
    "moniter": "monitor",
    "staning": "standing",
    "shooe": "shoes",
    "snicker": "sneakers"
}

KNOWN_BRANDS = [
    "nike", "adidas", "hoka", "brooks", "asics", "new balance", "saucony",
    "on cloud", "salomon", "under armour", "sony", "bose", "apple",
    "sennheiser", "anker soundcore", "anker", "jbl", "audio-technica", "beats",
    "marshall", "shure", "logitech", "keychron", "dell", "razer", "corsair",
    "lg", "caldigit", "ergotron", "herman miller", "steelcase", "autonomous",
    "flexispot", "uplift", "branch", "vari", "everlasting comfort", "patagonia",
    "arc'teryx", "lululemon", "the north face", "columbia", "gymshark",
    "smartwool", "osprey", "hydro flask", "garmin", "yeti", "manduka",
    "theragun", "fitbit", "stanley"
]

CATEGORY_KEYWORDS = {
    "Footwear & Athletics": ["shoe", "shoes", "sneaker", "sneakers", "running", "trail runner", "athletic", "insoles", "footwear", "boots", "socks"],
    "Audio & Acoustics": ["headphone", "headphones", "earbuds", "earphone", "audio", "speaker", "speakers", "microphone", "soundcore", "noise cancelling"],
    "Computing & Ergonomics": ["keyboard", "mouse", "monitor", "dock", "docking", "usb-c", "webcam", "laptop", "computing", "trackpad"],
    "Office Furniture & Wellness": ["chair", "desk", "standing desk", "converter", "lumbar", "arm", "office", "ergonomic chair"],
    "Apparel & Outdoor": ["jacket", "hoodie", "pants", "shirt", "tee", "sweatshirt", "baselayer", "backpack", "running jacket"],
    "Fitness Gear & Hydration": ["bottle", "water bottle", "smartwatch", "watch", "yoga", "yoga mat", "resistance bands", "fitness", "hydration"]
}

ATTRIBUTE_KEYWORDS = [
    "lightweight", "arch support", "waterproof", "gore-tex", "cushion",
    "breathable", "noise cancelling", "wireless", "bluetooth", "mechanical",
    "ergonomic", "split", "vertical", "4k", "thunderbolt", "dual motor",
    "electric", "mesh", "packable", "organic cotton", "oversized", "quick dry",
    "merino wool", "anti blister", "insulated", "stainless steel", "32oz",
    "gps", "heart rate", "non slip", "heavy duty", "zero drop", "wide toe"
]


class NLPEngine:
    """Provides typo correction, intent extraction, and text normalization."""

    def __init__(self):
        self.brands_lower = sorted(KNOWN_BRANDS, key=len, reverse=True)

    def normalize_text(self, text: str) -> str:
        """Standardizes text: lowercased, stripped, punctuation simplified."""
        if not text:
            return ""
        text = text.lower()
        # Keep alphanumeric, hyphen, and spaces
        text = re.sub(r"[^a-z0-9\s\-\$]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def correct_typos(self, query: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        Pillar 2: Typo correction for e-commerce search queries.
        Returns corrected query and list of corrections made.
        """
        tokens = query.lower().split()
        corrections = []
        corrected_tokens = []

        for token in tokens:
            clean_token = re.sub(r"[^a-z0-9]", "", token)
            if clean_token in COMMON_TYPOS:
                replacement = COMMON_TYPOS[clean_token]
                corrections.append({"original": token, "corrected": replacement})
                corrected_tokens.append(replacement)
            else:
                corrected_tokens.append(token)

        corrected_query = " ".join(corrected_tokens)
        return corrected_query, corrections

    def extract_intent(self, query: str) -> Dict[str, Any]:
        """
        Pillar 2: Intent extraction identifying Brand, Category,
        Attributes, and Price constraints.
        """
        normalized_q = self.normalize_text(query)
        detected_brand: Optional[str] = None
        detected_category: Optional[str] = None
        detected_attributes: List[str] = []
        max_price_constraint: Optional[float] = None

        # 1. Price constraint extraction (e.g., 'under 100', 'below $50', '< 80')
        price_patterns = [
            r"under\s+\$?(\d+(?:\.\d+)?)",
            r"below\s+\$?(\d+(?:\.\d+)?)",
            r"less\s+than\s+\$?(\d+(?:\.\d+)?)",
            r"<\s*\$?(\d+(?:\.\d+)?)"
        ]
        for pattern in price_patterns:
            m = re.search(pattern, normalized_q)
            if m:
                try:
                    max_price_constraint = float(m.group(1))
                    break
                except ValueError:
                    pass

        # 2. Brand identification
        for brand in self.brands_lower:
            pattern = rf"\b{re.escape(brand)}\b"
            if re.search(pattern, normalized_q):
                detected_brand = brand.title()
                break

        # 3. Category detection
        cat_scores = {}
        for category, keywords in CATEGORY_KEYWORDS.items():
            score = 0
            for kw in keywords:
                if rf" {kw} " in f" {normalized_q} " or normalized_q.startswith(kw) or normalized_q.endswith(kw):
                    score += len(kw.split()) + 1
            if score > 0:
                cat_scores[category] = score

        if cat_scores:
            detected_category = max(cat_scores.items(), key=lambda x: x[1])[0]

        # 4. Attribute extraction
        for attr in ATTRIBUTE_KEYWORDS:
            if attr in normalized_q:
                detected_attributes.append(attr)

        return {
            "query": query,
            "detected_brand": detected_brand,
            "detected_category": detected_category,
            "detected_attributes": detected_attributes,
            "max_price_constraint": max_price_constraint
        }

    def clean_and_extract_intent(self, raw_query: str) -> Tuple[str, Dict[str, Any]]:
        """Executes full NLP pipeline: cleaning, typo fixing, and intent extraction."""
        normalized = self.normalize_text(raw_query)
        corrected_query, corrections = self.correct_typos(normalized)
        intent = self.extract_intent(corrected_query)
        intent["raw_query"] = raw_query
        intent["cleaned_query"] = corrected_query
        intent["typo_corrections"] = corrections
        return corrected_query, intent


class BM25Retriever:
    """Stage 1: Lexical Candidate Retrieval baseline using BM25Okapi."""

    def __init__(self, catalog: List[Dict[str, Any]]):
        self.catalog = catalog
        self.corpus = [
            f"{p['title']} {p['brand']} {p['category']} {' '.join(p.get('features', []))} {p.get('description', '')}".lower().split()
            for p in catalog
        ]
        self.bm25 = BM25Okapi(self.corpus)

    def retrieve(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """Retrieves top_k candidates with raw BM25 scores."""
        query_tokens = query.lower().split()
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        candidates = []
        for idx in top_indices:
            score = float(scores[idx])
            prod = dict(self.catalog[idx])
            prod["bm25_score"] = score
            candidates.append(prod)

        return candidates
