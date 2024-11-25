ATTRIBUTES_TO_EXTRACT_PROMPT = """
Attributes to extract:

- type: one of ["top", "bottom", "dress", "jacket", "shoes", "sunglasses", "bag", "hat", "scarf", "belt"]
- color: primary color(s) of the item, e.g., "red", "blue", "black", "white", "gray", "green", "yellow", "pink", "purple", "orange", "brown", "beige", "multi-color"
- style: one or more of ["casual", "formal", "sporty", "business", "party", "beach"]
- season: one or more of ["spring", "summer", "autumn", "winter", "all-season"]
- occasion: one or more of ["casual", "business", "party", "wedding", "sports", "travel", "work", "school", "evening", "holiday"]
- price: one of ["low", "medium", "high"]
- material: one or more of ["cotton", "denim", "leather", "silk", "wool", "linen", "polyester", "nylon", "cashmere", "satin", "lace", "fur", "suede", "velvet", "canvas"]
- fit: one of ["slim", "regular", "loose", "oversized", "skinny", "relaxed"]
- gender: one of ["men", "women", "unisex"]
- age_group: one of ["adult", "teen", "child"]

Please provide the attributes in the following JSON format:

{
    "type": "...",
    "color": "...",
    "style": ["...", "..."],
    "season": ["...", "..."],
    "occasion": ["...", "..."],
    "price": "...",
    "material": ["...", "..."],
    "fit": "...",
    "gender": "...",
    "age_group": "..."
}

Ensure that all attribute values are selected from the provided options.

"""

STYLE_TO_EXTRACT_PROMPT = """




"""