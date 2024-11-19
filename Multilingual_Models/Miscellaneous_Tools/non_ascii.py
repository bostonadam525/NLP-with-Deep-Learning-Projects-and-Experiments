## function that transforms non ascii and unicode characters to ascii

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

    # Detect the encoding
    encoding = chardet.detect(encoded_string)['encoding']

    # Decode the string using the detected encoding
    decoded_string = encoded_string.decode(encoding)

    return decoded_string

# Example usage
encoded_text = b'\xc3\xa9cole'
decoded_text = decode_non_english(encoded_text)

print(decoded_text)  # Output: école
