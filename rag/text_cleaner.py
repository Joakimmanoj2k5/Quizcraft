import re

def clean_text(text: str) -> str:
    """
    Cleans extracted text without destroying meaningful information.
    Handles excessive whitespace, repeated blank lines, and unnecessary control characters.
    Preserves headings, bullet points, numbering, and important punctuation.
    """
    if not text:
        return ""
        
    # Remove null bytes and other strange control characters (but keep newlines, tabs)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # Replace multiple horizontal spaces/tabs with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Replace 3 or more newlines with exactly 2 newlines (preserve paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Attempt to fix broken line wrapping within a paragraph.
    # If a line ends with a word and the next line starts with a lower case letter,
    # it's likely a broken sentence.
    # We do this carefully: match newline not preceded by punctuation or newline,
    # and not followed by a newline, bullet, or uppercase letter.
    # We'll just do a very conservative replacement:
    # A newline surrounded by regular text characters (no punctuation at end of line 1, no bullet on line 2)
    # is often a wrapping artifact.
    # Instead of full complex regex, a simple approach: 
    # replace single newlines that are between lower/upper case letters with a space.
    # E.g., "The quick brown \nfox jumps" -> "The quick brown fox jumps"
    text = re.sub(r'(?<=[a-zA-Z,])[ \t]*\n[ \t]*(?=[a-z])', ' ', text)
    
    # Strip leading/trailing whitespace
    return text.strip()
