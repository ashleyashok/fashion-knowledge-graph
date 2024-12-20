# app/main.py

import os
import sys
import uuid
import base64

import numpy as np
import pandas as pd
import requests
import streamlit as st
from loguru import logger
from PIL import Image
from io import BytesIO

# Add the src directory to the sys.path if necessary
sys.path.append("src")

from src.database.graph_database import GraphDatabaseHandler
from src.database.vector_database import VectorDatabase
from src.inference.recommender import Recommender
from src.inference.product_attributes import AttributeExtractionModel

# Function to set the page title dynamically
def set_page_title(title):
    st.markdown(f"""
        <script>
            var originalTitle = document.title;
            document.title = '{title}';
            window.addEventListener('blur', function() {{
                document.title = 'Come back!';
            }});
            window.addEventListener('focus', function() {{
                document.title = '{title}';
            }});
        </script>
        """, unsafe_allow_html=True)

# Initialize databases and recommender in session_state
if 'graph_db' not in st.session_state:
    # Initialize graph database handler
    st.session_state['graph_db'] = GraphDatabaseHandler(
        uri=os.getenv("NEO4J_URI"),
        user=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD"),
    )

if 'vector_db_image' not in st.session_state:
    # Initialize vector databases
    st.session_state['vector_db_image'] = VectorDatabase(
        api_key=os.getenv("PINECONE_API_KEY"),
        host=os.getenv("PINECONE_HOST_IMAGE"),
        index_name="catalog-clothes",
    )

if 'vector_db_style' not in st.session_state:
    st.session_state['vector_db_style'] = VectorDatabase(
        api_key=os.getenv("PINECONE_API_KEY"),
        host=os.getenv("PINECONE_HOST_STYLE"),
        index_name="catalog-style-description",
    )

if 'recommender' not in st.session_state:
    # Initialize Recommender
    st.session_state['recommender'] = Recommender(
        graph_db=st.session_state['graph_db'],
        catalog_csv_path="output/data/catalog_combined.csv",
        vector_db_image=st.session_state['vector_db_image'],
        vector_db_style=st.session_state['vector_db_style'],
    )

# Load catalog data
if 'catalog_df' not in st.session_state:
    catalog_df = pd.read_csv("output/data/catalog_combined.csv")
    catalog_df["product_id"] = catalog_df["product_id"].astype(str)
    st.session_state['catalog_df'] = catalog_df

# Set initial page configuration without the specific page title
st.set_page_config(
    page_title="Catalog Enrichment",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    /* Main layout adjustments */
    .main {
        background-color: #ffffff;
        padding: 20px;
    }
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
        padding: 20px;
    }
    /* Logo in sidebar */
    .sidebar .logo {
        text-align: center;
        margin-bottom: 20px;
    }
    /* Header styling */
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #333333;
        text-align: center;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .title img {
        margin-right: 15px;
    }
    /* Product attributes display */
    .attributes-display {
        background-color: #fafafa;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .attributes-display h3 {
        margin-top: 0;
    }
    .attribute-item {
        margin-bottom: 10px;
    }
    .attribute-label {
        font-weight: bold;
        color: #555555;
    }
    /* Button styling */
    .stButton>button {
        background-color: #0072C6;
        color: white;
        border-radius: 5px;
    }
    /* Single-line text input */
    .stTextInput>div>div>input {
        height: auto;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar with logo and tool name
with st.sidebar:
    st.markdown(
        f"""
        <div class="logo">
            <img src="data:image/png;base64,{base64.b64encode(open('temp_images/tiger_logo.png', 'rb').read()).decode()}" width="150" />
        </div>
        <h2 style="text-align: center;">Catalog Enrichment</h2>
        """,
        unsafe_allow_html=True,
    )
    # Create a radio button to select between the functionalities
    option = st.radio(
        "Choose a feature",
        (
            "Product Attribute Extraction",
            "Complete the Look",
            "Style Match: Upload Your Outfit",
            "Style Match: Describe Your Outfit",
        ),
    )

# Set the page title based on the selected option
set_page_title(option)

# Update the main header based on the selected option
st.markdown(
    f"""
    <div class="title">
        {option}
    </div>
    """,
    unsafe_allow_html=True,
)

def display_attributes(attributes):
    for key, value in attributes.items():
        if isinstance(value, (list, np.ndarray)):
            # Remove nulls from the list
            value = [str(v) for v in value if not pd.isnull(v)]
            if not value:
                continue  # Skip if the list is empty after removing nulls
            value = ", ".join(value)
        else:
            if pd.isnull(value):
                continue  # Skip if the scalar value is null
        st.write(f"<span class='attribute-label'>{key.capitalize()}:</span> {value}", unsafe_allow_html=True)

def display_recommendations(products):
    num_products = len(products)
    cols = st.columns(5)
    for idx, rec in enumerate(products):
        with cols[idx % 5]:
            st.markdown('<div class="product-card">', unsafe_allow_html=True)
            if rec.get("image_path"):
                st.image(
                    rec["image_path"],
                    width=150,
                    caption=f"Product ID: {rec.get('product_id', 'N/A')}",
                )
            else:
                st.write(f"Product ID: {rec.get('product_id', 'N/A')}")
                st.write("Image not available.")
            if "clip_similarity_score" in rec:
                st.write(f"**Clip Similarity Score:** {rec['clip_similarity_score']:.2f}")
            if "style_similarity_score" in rec:
                st.write(f"**Style Similarity Score:** {rec['style_similarity_score']:.2f}")
            # Add subheading "Attributes"
            st.markdown("**Attributes:**")
            # Safely get attributes and make a copy
            attributes = rec.get("attributes", {}).copy()
            # Include 'product_id' in attributes
            attributes['product_id'] = rec.get('product_id', 'N/A')
            # Sort attributes alphabetically, with 'product_id' first
            sorted_attributes = dict(sorted(attributes.items(), key=lambda x: (0, x[0]) if x[0] == 'product_id' else (1, x[0].lower())))
            # Display attributes
            for key, value in sorted_attributes.items():
                if isinstance(value, (list, np.ndarray)):
                    value = ', '.join(map(str, value))
                st.write(f"<span class='attribute-label'>{key.capitalize()}:</span> {value}", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

def find_similar_outfit(
    image_path_or_url: str,
    image_id: str = "",
    similarity_threshold: float = 0.75,
    top_k=1,
    visualize=False,
) -> None:
    matched_products, segmented_filepaths = recommender.get_outfit_from_image(
        image_path_or_url,
        similarity_threshold=similarity_threshold,
        top_k=top_k,
        visualize=visualize,
        image_id=image_id,
    )
    if segmented_filepaths:
        st.subheader("Segmented Items")
        num_images = len(segmented_filepaths)
        img_cols = st.columns(num_images)
        for idx, img_file in enumerate(segmented_filepaths):
            segmented_name = img_file.split("_")[-1].split(".")[0]
            with img_cols[idx % num_images]:
                st.image(img_file, width=150)
                st.write(segmented_name)
    else:
        st.info("No segmented items found.")

    if matched_products:
        st.subheader("Matched Products from Catalog")
        display_recommendations(matched_products)
    else:
        st.info("No matching products found for the image.")

def display_attributes_readonly(attributes):
    st.markdown('<div class="attributes-display">', unsafe_allow_html=True)
    st.markdown("<h3>Extracted Attributes</h3>", unsafe_allow_html=True)
    for key, value in attributes.items():
        if isinstance(value, list):
            value = ', '.join(map(str, value))
        st.markdown(
            f"<div class='attribute-item'><span class='attribute-label'>{key}:</span> {value}</div>",
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

def display_attributes_editable(attributes):
    edited_attributes = {}
    for key, value in attributes.items():
        if isinstance(value, list):
            value_str = ', '.join(map(str, value))
            new_value = st.text_input(f"{key}", value=value_str, key=f"edit_{key}")
            new_value_list = [v.strip() for v in new_value.split(',')]
            edited_attributes[key] = new_value_list
        else:
            new_value = st.text_input(f"{key}", value=str(value), key=f"edit_{key}")
            edited_attributes[key] = new_value
    return edited_attributes

# Define a function to set edit_mode to True
def enter_edit_mode():
    st.session_state['edit_mode'] = True

if option == "Product Attribute Extraction":
    # Product Attribute Extraction functionality
    st.subheader("Extract Attributes from an Image")

    # Initialize session state variables if they don't exist
    if 'image_url' not in st.session_state:
        st.session_state['image_url'] = ''
    if 'image' not in st.session_state:
        st.session_state['image'] = None
    if 'attributes' not in st.session_state:
        st.session_state['attributes'] = None
    if 'edited_attributes' not in st.session_state:
        st.session_state['edited_attributes'] = None
    if 'edit_mode' not in st.session_state:
        st.session_state['edit_mode'] = False

    # Use session state for image_url
    image_url = st.text_input("Enter the URL of your image:", value=st.session_state['image_url'])
    if st.button("Submit"):
        if image_url:
            try:
                # Fetch the image from the URL
                response = requests.get(image_url)
                response.raise_for_status()  # Raise an HTTPError for bad responses
                image = Image.open(BytesIO(response.content))

                # Save the image and URL in session state
                st.session_state['image'] = image
                st.session_state['image_url'] = image_url

                # Extract attributes
                with st.spinner("Generating output..."):
                    model = AttributeExtractionModel()
                    attributes = model.extract_attributes(image_url)
                    st.session_state['attributes'] = attributes
                    st.session_state['edited_attributes'] = attributes.copy()
                    st.session_state['edit_mode'] = False
                    st.success("Attributes extracted successfully!")
            except Exception as e:
                st.error(f"Error loading image or extracting attributes: {e}")

    # If image and attributes are available, display them
    if st.session_state['image'] and st.session_state['attributes']:
        # Retrieve attributes from session state
        attributes = st.session_state['attributes']
        edited_attributes = st.session_state['edited_attributes']

        # Create two columns
        left_column, right_column = st.columns([1, 1])

        # Display the image in the left column
        with left_column:
            st.image(
                st.session_state['image'],
                caption="Product Image",
                use_container_width=True,
            )

        # Display attributes in the right column
        with right_column:
            if not st.session_state['edit_mode']:
                # Display product details
                product_details = attributes.get('product_details', {})
                if product_details:
                    st.markdown('<div class="attributes-display">', unsafe_allow_html=True)
                    st.markdown("<h3>Product Details</h3>", unsafe_allow_html=True)
                    for key, value in product_details.items():
                        if isinstance(value, list):
                            value = ', '.join(map(str, value))
                        st.markdown(
                            f"<div class='attribute-item'><span class='attribute-label'>{key}:</span> {value}</div>",
                            unsafe_allow_html=True,
                        )
                    st.markdown('</div>', unsafe_allow_html=True)

                # Display attributes under 'Attributes' section
                attributes_only = attributes.get('attributes', {})
                if attributes_only:
                    st.markdown('<div class="attributes-display">', unsafe_allow_html=True)
                    st.markdown("<h3>Attributes</h3>", unsafe_allow_html=True)
                    for key, value in attributes_only.items():
                        if isinstance(value, list):
                            value = ', '.join(map(str, value))
                        st.markdown(
                            f"<div class='attribute-item'><span class='attribute-label'>{key}:</span> {value}</div>",
                            unsafe_allow_html=True,
                        )
                    st.markdown('</div>', unsafe_allow_html=True)

                # Edit button with a callback to enter edit mode
                st.button("Edit Attributes", on_click=enter_edit_mode)
            else:
                # Editable form
                st.markdown("### Edit Attributes")
                with st.form(key='attributes_form'):
                    # Combine product_details and attributes for editing
                    combined_attributes = {**edited_attributes.get('product_details', {}), **edited_attributes.get('attributes', {})}
                    updated_attributes = display_attributes_editable(combined_attributes)
                    # Submit button
                    submit_attributes = st.form_submit_button(label='Save Changes')

                if submit_attributes:
                    # Update the attributes in session state
                    attributes = st.session_state.get('attributes')
                    if not attributes:
                        st.error("Attributes data is missing.")
                    else:
                        # Split the updated attributes back into 'product_details' and 'attributes'
                        keys_product_details = attributes.get('product_details', {}).keys()
                        keys_attributes = attributes.get('attributes', {}).keys()
                        st.session_state['edited_attributes']['product_details'] = {k: updated_attributes.get(k, '') for k in keys_product_details}
                        st.session_state['edited_attributes']['attributes'] = {k: updated_attributes.get(k, '') for k in keys_attributes}

                        st.session_state['attributes'] = st.session_state['edited_attributes'].copy()
                        st.session_state['edit_mode'] = False
                        st.success("Attributes updated!")
    else:
        st.info("Please enter an image URL and click Submit.")

elif option == "Complete the Look":
    # Complete the Look functionality
    st.subheader("Select a Product")
    recommender = st.session_state['recommender']
    catalog_df = st.session_state['catalog_df']
    # Create a selection box for products with images
    product_options = catalog_df[["product_id", "image_path"]]
    product_options["display"] = product_options.apply(
        lambda row: f"ID: {row['product_id']}", axis=1
    )
    selected_product_display = st.selectbox(
        "Choose a product:", product_options["display"].tolist()
    )
    selected_product_id = product_options.loc[
        product_options["display"] == selected_product_display, "product_id"
    ].values[0]

    if selected_product_id:
        result = recommender.get_recommendations(
            selected_product_id, threshold=1, top_k=5
        )
        selected_product = result["selected_product"]
        worn_with_products = result["worn_with"]
        complemented_products = result["complemented"]

        # Remove duplicates
        unique_products = {item["product_id"]: item for item in worn_with_products}
        worn_with_products = list(unique_products.values())

        unique_products = {item["product_id"]: item for item in complemented_products}
        complemented_products = list(unique_products.values())

        # Display selected product
        st.subheader("Selected Product")
        with st.container():
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(
                    selected_product["image_path"],
                    width=250,
                    caption=f"Product ID: {selected_product['product_id']}",
                )
            with col2:
                st.markdown("<h3>Attributes</h3>", unsafe_allow_html=True)
                attributes = selected_product["attributes"]
                display_attributes(attributes)

        # Display outfit ideas
        st.subheader("Outfit Ideas")
        if worn_with_products or complemented_products:
            for rec in worn_with_products + complemented_products:
                st.markdown(f"<h4>Outfit with Product ID: {rec['product_id']}</h4>", unsafe_allow_html=True)
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.image(
                        selected_product["image_path"],
                        width=250,
                        caption=f"Selected Product ID: {selected_product['product_id']}",
                    )
                with col2:
                    st.image(
                        rec["image_path"],
                        width=250,
                        caption=f"Recommended Product ID: {rec['product_id']}",
                    )
                # Display social media images if available
                if rec.get("images"):
                    rec_images = list(set(rec["images"]))
                    st.write("**Social Media Images:**")
                    num_images = len(rec_images)
                    img_cols = st.columns(num_images)
                    for idx, img_file in enumerate(rec_images):
                        img_path = os.path.join("dataset/DeepFashion", img_file)
                        if os.path.exists(img_path):
                            with img_cols[idx % num_images]:
                                st.image(img_path, width=150)
                        else:
                            st.write(f"Image not found: {img_file}")
                else:
                    st.write("No social media images available.")
                st.write("---")
        else:
            st.info("No outfit ideas found.")
    else:
        st.write("Please select a product to get recommendations.")

elif option == "Style Match: Upload Your Outfit":
    # Style Match: Upload Your Outfit functionality
    st.subheader("Upload an Image of Your Outfit")

    # Option to select input method
    input_option = st.radio("Select input method:", ("Upload Image", "Enter Image URL"))
    recommender = st.session_state['recommender']
    if input_option == "Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image of your outfit", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            # Save the uploaded image to a temporary location
            temp_dir = "temp_images/user_uploaded"
            image_id = "user_uploaded_image_" + str(uuid.uuid4())
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            temp_image_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.image(temp_image_path, caption="Uploaded Image", use_container_width=True)

            # Process the image and get matching products
            with st.spinner("Processing image..."):
                find_similar_outfit(
                    temp_image_path,
                    image_id=image_id,
                    visualize=True,
                    similarity_threshold=0.7,
                )

    elif input_option == "Enter Image URL":
        image_url = st.text_input("Enter the URL of the image:")
        if image_url:
            # Create a unique image id prefix using uuid
            image_id = "user_uploaded_image_" + str(uuid.uuid4())
            st.image(image_url, caption="Input Image", use_container_width=True)
            find_similar_outfit(
                image_url, image_id=image_id, visualize=True, similarity_threshold=0.74
            )

elif option == "Style Match: Describe Your Outfit":
    # Style Match: Describe Your Outfit functionality
    st.subheader("Describe Your Outfit")
    outfit_description = st.text_input("Enter a description of your outfit:")
    recommender = st.session_state['recommender']
    if outfit_description:
        # Process the text and get matching products
        matched_products = recommender.get_outfit_from_text(
            text=outfit_description, top_k=5, style_text_similarity_threshold=0.7
        )
        if matched_products:
            st.subheader("Matched Products from Catalog")
            display_recommendations(matched_products)
        else:
            st.info("No matching products found for the description.")

