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

### For product attribute extraction from raw images of data sheets
DEFAULT_SYS_PROMPT = """Review the product description and images carefully. Your main task is to write a product title, product description and determine the accurate
selection(s) for a series of attributes which are highlighted by being placed within ###. These attributes are
identified as keys, and for each key, you will have a set of possible options (values) to choose from. To achieve
precise results, adhere to the following guidelines:

1- You are provided with the product data sheet / specification sheet / seller raw data. Get an understanding of the entire product. 
2- Focus on relevant product areas for each attribute, like the neck for 'neckline', to choose the best matching
   option and pay attention to small details in the images.
3- Use all provided images to get a thorough understanding of the product from different views—front, back, and
   side. Select and utilize the image that offers the best view for identifying the value of each attribute accurately.
   For example for the attribute 'back exposure,' ensure to use the image showing the back of the product to determine
   the correct value.
4- Conduct a final review of all selected attribute values against the images and product description to confirm
   their accuracy before finalizing the response. 
5. Ensure you have the product details, size and material&care information extracted. If not available, just leave that field blank.
Your task is to use the product description and images meticulously to identify the attribute and its value for each attribute
based on the visual evidence provided. Remove any digit and percentage signs from the final output.Compile findings in
JSON format, attributing attributes as keys and values as a single string seperated by comma. Example output:

  {"product_details": { "Product Title": "Mens Athletic Jacket",
    "Product Summary": "A timeless staple, this <brand name> hoodie is in soft, brushed-back fleece for classic comfort.",
    "Product Description": "Fit & Details" : <Fit and detail description that you can infer>, "Material": <material description that you can infer>",
    "Brand": "XYZ"},
    "attributes" : {"Neck Style": "V Neck",
    "Fit": "Slim, Fitted" ,
    "Sleeve Length": "Long",
    "Fabric Type": "Cotton, Cotton Blend",
    "Pattern":"Striped"}
    }
 """


attributes = {
    "sleeve length": ["full sleeve", "short sleeve", "sleeveless"],
    "activity": ["leisure", "sports", "yoga", "formal"],
    "back exposure": ["partial coverage", "full coverage"],
    "fit": [
        "tight",
        "unstructured",
        "oversize",
        "structured",
        "classic",
        "loose",
        "slim",
    ],
    "sleeve fit": [
        "fitted sleeve",
        "oversized sleeve",
        "relaxed sleeve",
        "tight sleeve",
    ],
    "closure": ["front button", "pullover"],
    "pocket details": [
        "patch pocket",
        "kangaroo pocket",
        "seam pocket",
        "slant pocket",
    ],
    "hem details": ["curved hem", "asymmetric hem", "elastic hem"],
    "length": ["at knee", "at waist", "at hips"],
    "pattern": ["striped", "plain", "logo", "floral", "abstract"],
}

DEFAULT_USER_PROMPT_TEMPLATE = """"Your task is to examine the provided image
to give a suitabe product title, product summary in less than 2 lines and a section wise product description and identify specific features mentioned in the description. For each attribute listed between the '###' markers,
with attributes as keys and their potential values, you are to assign the correct value(s) based on the images
and description. Here are the steps to follow:

    Come up with a suitable product title, description. 
    Review all provided images of the product, observing it from front, back, and side views to get a
    comprehensive understanding.
    For each attribute listed:
        1- If the attribute is clearly seen in any of the images, select the corresponding value that best matches.
        2- If the attribute cannot be determined from the images and text , choose the 'not present' option.
        3- If the attribute does not apply to the product (e.g., 'sleeve length' for a pair of pants),
           select 'not relevant'.

Here are the attributes you need to evaluate:

Attributes:
###
{}
###

Use the information from both the description and the images to accurately assign values for each attribute."
"""

DEFAULT_USER_PROMPT = DEFAULT_USER_PROMPT_TEMPLATE.format(attributes)