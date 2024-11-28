# app/main.py

import os
import sys
import uuid

import numpy as np
import pandas as pd
import requests
import streamlit as st
from loguru import logger
from PIL import Image
from io import BytesIO
import base64

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

# Initialize graph database handler
graph_db = GraphDatabaseHandler(
    uri=os.getenv("NEO4J_URI"),
    user=os.getenv("NEO4J_USERNAME", "neo4j"),
    password=os.getenv("NEO4J_PASSWORD"),
)

# Initialize vector databases
vector_db_image = VectorDatabase(
    api_key=os.getenv("PINECONE_API_KEY"),
    host=os.getenv("PINECONE_HOST_IMAGE"),
    index_name="catalog-clothes",
)
vector_db_style = VectorDatabase(
    api_key=os.getenv("PINECONE_API_KEY"),
    host=os.getenv("PINECONE_HOST_STYLE"),
    index_name="catalog-style-description",
)

# Initialize Recommender
recommender = Recommender(
    graph_db=graph_db,
    catalog_csv_path="output/data/catalog_combined.csv",
    vector_db_image=vector_db_image,
    vector_db_style=vector_db_style,
)

# Load catalog data
catalog_df = pd.read_csv("output/data/catalog_combined.csv")
catalog_df["product_id"] = catalog_df["product_id"].astype(str)

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
    /* Product card styling */
    .product-card {
        background-color: #fafafa;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
    }
    .product-image {
        border-radius: 10px;
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
            "Product Recommendations",
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
        <img src="data:image/png;base64,{base64.b64encode(open('temp_images/tiger_logo.png', 'rb').read()).decode()}" width="60" />
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
            if rec["image_path"]:
                st.image(
                    rec["image_path"],
                    width=150,
                    caption=f"Product ID: {rec['product_id']}",
                )
            else:
                st.write(f"Product ID: {rec['product_id']}")
                st.write("Image not available.")
            if "clip_similarity_score" in rec:
                st.write(f"**Clip Similarity Score:** {rec['clip_similarity_score']:.2f}")
            if "style_similarity_score" in rec:
                st.write(f"**Style Similarity Score:** {rec['style_similarity_score']:.2f}")
            st.write("**Attributes:**")
            attributes = rec["attributes"]
            display_attributes(attributes)
            st.markdown("</div>", unsafe_allow_html=True)

def find_similar_outfit(
    image_path_or_url: str,
    image_id: str = "",
    similarity_threshold: float = 0.75,
    top_k=5,
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

if option == "Product Recommendations":
    # Product Recommendations functionality
    st.subheader("Select a Product")
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
            find_similar_outfit(
                temp_image_path,
                image_id=image_id,
                visualize=True,
                similarity_threshold=0.74,
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

elif option == "Product Attribute Extraction":
    # Product Attribute Extraction functionality
    st.subheader("Extract Attributes from an Image")
    with st.form(key="image_form"):
        image_url = st.text_input("Enter the URL of your image:", "")
        submit_button = st.form_submit_button(label="Submit")
    if submit_button and image_url:
        try:
            # Fetch the image from the URL
            response = requests.get(image_url)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            image = Image.open(BytesIO(response.content))

            # Create two columns below the input box
            left_column, _, right_column = st.columns([3, 1.2, 3])

            # Display the image in the left column
            with left_column:
                st.image(
                    image,
                    caption="Product Image",
                    use_container_width=True,
                    width=300,
                )

            # Display "Hello World" in the right column
            with right_column:
                try:
                    with st.spinner("Generating output..."):
                        model = AttributeExtractionModel()
                        attributes = model.extract_attributes(image_url)
                        product_details = attributes["product_details"]
                        attributes_str = attributes["attributes"]

                    st.markdown(
                        """
                            <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100%; text-align: center;">
                            """,
                        unsafe_allow_html=True,
                    )

                    # Add dictionary content inside the div
                    for key, value in product_details.items():
                        st.markdown(
                            f"<div style='margin-bottom: 10px;'><b>{key}:</b><br>{value}</div>",
                            unsafe_allow_html=True,
                        )

                    # Close the div
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Display the second key (details) as a JSON object
                    st.json(attributes_str)

                except Exception as e:
                    st.error(f"Model failed to generate: {e}")
        except Exception as e:
            st.error(f"Error loading image: {e}")