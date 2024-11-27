import os
import base64
from typing import Dict, Any
from PIL import Image
from io import BytesIO
import json
from loguru import logger
from src.utils.prompts import (
    ATTRIBUTES_TO_EXTRACT_PROMPT,
    STYLE_TO_EXTRACT_PROMPT,
    STYLE_TO_EXTRACT_FROM_TEXT_PROMPT,
)
from openai import AzureOpenAI


class AttributeExtractionModel:
    def __init__(self):
        """
        Initialize the AttributeExtractionModel.
        """
        # Initialize Azure OpenAI client
        self.llm_client = AzureOpenAI(
            api_key=os.getenv("TIGER_AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("TIGER_AZURE_OPENAI_URL"),
            api_version="2024-08-01-preview",
        )

    def encode_image(self, image: Image.Image) -> str:
        """
        Encode a PIL Image to a base64 string.

        Parameters
        ----------
        image : PIL.Image.Image
            The image to encode.

        Returns
        -------
        str
            The base64-encoded image string.
        """
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def extract_attributes(self, image: Image.Image, label: str) -> Dict[str, Any]:
        """
        Extract attributes from the image using the LLM.

        Parameters
        ----------
        image : PIL.Image.Image
            The image to analyze.
        label : str
            The label or description of the image.

        Returns
        -------
        Dict[str, Any]
            The extracted attributes.
        """
        image_base64 = self.encode_image(image)

        prompt = [
            {
                "role": "system",
                "content": "You are an AI assistant that can analyze images and extract attributes from them and return the attributes in JSON format.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this image and extract the following attributes, ensuring that the output JSON follows the specified format and uses only the provided options for each attribute value.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                    {"type": "text", "text": ATTRIBUTES_TO_EXTRACT_PROMPT},
                ],
            },
        ]

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=prompt,
            )

            assistant_message = response.choices[0].message.content
            logger.debug(f"GPT-4o response: {assistant_message}")

            attributes = json.loads(assistant_message)
            return attributes
        except Exception as e:
            logger.error(f"Error extracting attributes: {e}")
            # Return default attributes if extraction fails
            attributes = {
                "type": "unknown",
                "color": "unknown",
                "style": ["unknown"],
                "season": ["unknown"],
                "occasion": ["unknown"],
                "price": "unknown",
                "material": ["unknown"],
                "pattern": "unknown",
                "fit": "unknown",
                "gender": "unknown",
                "age_group": "unknown",
            }
            return attributes

    def extract_style_description_from_image(
        self, image: Image.Image, label: str
    ) -> str:
        """
        Extract a concise style description from the image.

        Parameters
        ----------
        image : PIL.Image.Image
            The image to analyze.
        label : str
            The label or description of the image.

        Returns
        -------
        str
            The style description.
        """
        image_base64 = self.encode_image(image)

        prompt = [
            {
                "role": "system",
                "content": "You are a fashion expert AI assistant that analyzes images and describes the style attributes of garments in a concise sentence.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please analyze the following image and provide a style description as instructed.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                    {"type": "text", "text": STYLE_TO_EXTRACT_PROMPT},
                ],
            },
        ]

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "text"},
                messages=prompt,
            )

            assistant_message = response.choices[0].message.content.strip()
            logger.debug(f"Style Description: {assistant_message}")

            return assistant_message
        except Exception as e:
            logger.error(f"Error extracting style description: {e}")
            return ""

    def extract_style_description_from_text(self, text: str) -> str:
        """
        Extract a style description from a search query using GPT-4o.

        Parameters
        ----------
        text : str
            Search query input by the user.

        Returns
        -------
        str
            The style description generated by GPT-4.
        """
        prompt = [
            {
                "role": "system",
                "content": "You are a fashion expert AI assistant that interprets search queries and extracts the essence of the style in a concise, factual sentence.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please interpret the following search query and provide a style description as instructed.",
                    },
                    {
                        "type": "text",
                        "text": f"Search Query: \"{text}\"",
                    },
                    {"type": "text", "text": STYLE_TO_EXTRACT_FROM_TEXT_PROMPT},
                ],
            },
        ]

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "text"},
                messages=prompt,
                temperature=0.5,
                max_tokens=150,
            )

            assistant_message = response.choices[0].message.content.strip()
            logger.debug(f"Style Description from text: {assistant_message}")
            return assistant_message
        except Exception as e:
            logger.error(f"Error extracting style description from text: {e}")
            return ""


if __name__ == "__main__":
    # Test the AttributeExtractionModel
    attribute_model = AttributeExtractionModel()

    # Test style description extraction from text
    text = "something to wear on a night out"
    style_description_text = attribute_model.extract_style_description_from_text(text)
    logger.info(f"Style Description from Text: {style_description_text}")
