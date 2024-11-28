import os
from typing import Dict, Any
import json
from loguru import logger
from openai import AzureOpenAI
from src.utils.prompts import DEFAULT_SYS_PROMPT, DEFAULT_USER_PROMPT
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()


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

    def extract_attributes(self, url: str) -> Dict[str, Any]:
        """
        Extract attributes from the image using the LLM.
        Parameters
        ----------
        url : str
            The link to the image to analyze.
        Returns
        -------
        Dict[str, Any]
            The extracted attributes.
        """

        prompt = [
            {
                "role": "system",
                "content": DEFAULT_SYS_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": DEFAULT_USER_PROMPT,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": url},
                    },
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


if __name__ == "__main__":
    model = AttributeExtractionModel()
    attributes = model.extract_attributes(
        url="https://www.textileblog.com/wp-content/uploads/2023/05/spec-sheet-of-garment.jpg"
    )
    print(attributes)
