## function that transforms non ascii and unicode characters to ascii
## This is a good article on encoding and decoding characters in python: https://realpython.com/python-encodings-guide/

def transform_non_ascii_to_ascii(text):
  """Transforms non-ASCII characters in a string to their ASCII equivalents using HTML entities.

  Args:
    text: The input string.

  Returns:
    The string with non-ASCII characters replaced by HTML entities.
    Returns the original text if it's None or empty.
    If text is a float, converts it to string first.
  """
  if not text:
      return text
  # Convert float to string before applying html.escape
  if isinstance(text, float):
      text = str(text)
  return html.escape(text)



## example usage
df['message_ascii'] = df['message'].apply(transform_non_ascii_to_ascii)

------------------------------------------------------------
## Another function that uses the chardet library in Python
import chardet

def decode_non_english(encoded_string):
    """Decodes non-English characters in a string."""

    # Check if the input is already a string (decoded) or is None
    if isinstance(encoded_string, str) or encoded_string is None:
        return encoded_string  # Return as is if it's already decoded or None

    # Check if the input is a float and convert to string if necessary
    if isinstance(encoded_string, float):
        encoded_string = str(encoded_string)

    # Check if the string is actually encoded (contains non-ASCII characters)
    try:
        encoded_string.encode('ascii')  # Try encoding to ASCII
        # If successful, it's already ASCII and doesn't need decoding
        return encoded_string
    except UnicodeEncodeError:
        # If it fails, it needs decoding
        encoding = chardet.detect(encoded_string.encode())['encoding']
        decoded_string = encoded_string.decode(encoding, errors='ignore')  # Handle potential decoding errors
        return decoded_string


## example usage
df['message_decoded'] = df['message'].apply(encoded_string)


==========================================
## you can also use Pydantic to do this --> note this will leave empty strings if a row value cant be decoded. 
from typing import Union
from pydantic import BaseModel, field_validator ## V2 pydantic field_validator

## Pydantic class to decode string
class DecodedString(BaseModel):
    text: str

    @field_validator('text', mode="before")
    def decode_if_needed(cls, v):
        """Decodes non-English characters if necessary."""
        if isinstance(v, bytes):
            encoding = chardet.detect(v)['encoding']
            try:
                return v.decode(encoding)
            except UnicodeDecodeError:
                # Fallback to UTF-8 if detected encoding fails
                return v.decode('utf-8', errors='ignore')
        return v

def decode_non_english_pydantic(text: Union[str, bytes]) -> str:
    """Decodes non-English characters using Pydantic."""
    # Replace nan with an empty string before passing to Pydantic
    if pd.isna(text):  # Use pd.isna to check for nan
        text = ''
    return DecodedString(text=text).text

# Example usage
df['message_decoded_pydantic'] = df['message'].apply(decode_non_english_pydantic)
df['message_decoded_pydantic'].head())


------------------------------------------
## Another approach using Pydantic ---> however this will keep the nan values if they cant be decoded from non-english characters
from typing import Union
from pydantic import BaseModel, field_validator, ValidationError

class DecodedString(BaseModel):
    text: str

    @field_validator('text', mode="before")
    def decode_if_needed(cls, v):
        """Decodes non-English characters if necessary."""
        if isinstance(v, bytes):
            encoding = chardet.detect(v)['encoding']
            try:
                return v.decode(encoding)
            except UnicodeDecodeError:
                # If decoding fails, return nan
                return float('nan')  
        return v

def decode_non_english_pydantic(text: Union[str, bytes]) -> str:
    """Decodes non-English characters using Pydantic."""
    
    try:
        # Attempt to decode using the Pydantic model
        decoded_text = DecodedString(text=text).text  
    except ValidationError:
        # If validation fails (e.g., decoding error), return nan
        return float('nan') 
    
    return decoded_text 

# Example usage
df['message_decoded_pydantic'] = df['message'].apply(decode_non_english_pydantic)
df['message_decoded_pydantic'].head()

