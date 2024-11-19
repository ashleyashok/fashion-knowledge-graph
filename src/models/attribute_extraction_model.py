# src/models/attribute_extraction_model.py

import os
import base64
from typing import Dict, Any
from PIL import Image
from io import BytesIO
import json
from loguru import logger
from src.utils.prompts import ATTRIBUTES_TO_EXTRACT_PROMPT
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
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
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
