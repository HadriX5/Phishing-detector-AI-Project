def repair_encoding(text):
    """
    Attempt to repair encoding corruption in the input text.
    - If `text` is str we replace known CP1252-unicode artefacts with their
      raw single-byte equivalents, then try Latin-1 -> UTF-8 re-decode.

    Returns:
        Repaired string when possible, otherwise the original input.
    """
    if not isinstance(text, str):
        return text

    # Dictionary mapping Windows-1252 mis-encodings to their byte values
    cp1252_fix = {
        '\u20AC': '\x80', '\u201A': '\x82', '\u0192': '\x83', '\u201E': '\x84',
        '\u2026': '\x85', '\u2020': '\x86', '\u2021': '\x87', '\u02C6': '\x88',
        '\u2030': '\x89', '\u0160': '\x8A', '\u2039': '\x8B', '\u0152': '\x8C',
        '\u017D': '\x8E', '\u2018': '\x91', '\u2019': '\x92', '\u201C': '\x93',
        '\u201D': '\x94', '\u2022': '\x95', '\u2013': '\x96', '\u2014': '\x97',
        '\u02DC': '\x98', '\u2122': '\x99', '\u0161': '\x9A', '\u203A': '\x9B',
        '\u0153': '\x9C', '\u017E': '\x9E', '\u0178': '\x9F'
    }
    
    # Replace Windows-1252 chars with their raw byte equivalents
    for char, byte_char in cp1252_fix.items():
        text = text.replace(char, byte_char)
        
    # Encode to Latin-1 and decode as UTF-8
    try:
        return text.encode('latin-1').decode('utf-8')
        
    except UnicodeError:
        return text
