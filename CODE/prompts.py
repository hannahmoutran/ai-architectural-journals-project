"""
Southern Architect Prompts Module

This module contains all prompt components for the Southern Architect archival project.
Centralizes prompt management and eliminates duplication.
"""

class SouthernArchitectPrompts:
    """Container for all Southern Architect text analysis prompts."""
    
    # Core prompt components
    COLLECTION_DESCRIPTION = """
You are an archivist at the University of Texas at Austin, cataloging a collection of issues of The Southern Architect (1892-1931). You are working from OCR text that may contain errors, so use your judgment to decipher meaning. This metadata is for architectural historians and architectural history students. Architectural styles, movements, trends, and other historically important information should be prioritized.
"""
    
    OCR_CLEANING_INSTRUCTIONS = """
Before analyzing the content, please clean the OCR text according to these guidelines:
1. Correct misspellings that are clearly due to OCR errors.
2. Fix obvious punctuation errors.
3. Correct word splits (e.g., "architec ture" should be "architecture").
4. Do not alter proper names, even if they seem unusual, unless it's an obvious OCR error.
5. Do not add or remove content; your task is to clean, not to summarize or expand.
6. If you're unsure about a word or phrase, leave it as is.
7. Maintain the original formatting as much as possible. Remember that it is a page from an old magazine. Many pages begin with the name of the periodical, the date, and the page number. This information may be edited slightly for clarity by, for example, separating the date from the page number.

Provide the cleaned text followed by '---' before proceeding with the metadata extraction.
"""
    
    ADVERTISEMENT_INSTRUCTIONS = """
How to identify advertisements:
1. Look for company names prominently displayed
2. Check for product descriptions or benefits
3. Look for contact information or calls to action
4. Be aware of persuasive language or marketing slogans
5. Note any pricing information or special offers
"""
    
    CONTENT_WARNING_INSTRUCTIONS = """
After analyzing the content, assess whether it contains any language or themes that might warrant a content warning. Consider the following categories:
1. Racist/ethnically insensitive language
2. Sexist/gender-discriminatory content
3. Violence/graphic descriptions
4. Other potentially offensive/outdated terminology

If you identify any such content, briefly describe it using the categorization described above.
If no sensitive content is present, simply return 'None'.
"""
    
    # JSON response formats
    GENERIC_JSON_FORMAT = """{
    "cleanedText": "The cleaned OCR text with corrections applied",
    
    "tocEntry": "A descriptive entry appropriate for a table of contents. Include:
        - Page type (cover, table of contents, advertisement, editorial, article, or other)
        - Short description of the content of the page
        - Specific persons or organizations mentioned (give a brief description of their importance and relevance to the content - for example, "Zaha Hadid (Parametric architecture pioneer)", or "Frank Lloyd Wright (Prairie School architect)")
        - Key topics or themes covered",
    
    "namedEntities": [
        "List key entities including:
        - Architects and architectural firms
        - Geographic locations 
        - Building names
        - Significant people mentioned
        - Organizations
        - Historical events or periods
        - Architectural styles or movements
        - Innovations or technologies referenced
        - Any other notable entities relevant to the content of the page",
        - Write each entity as a separate string in the list
        - Simplify where possible.  No need to include titles.  
    ],
    
    "subjects": [
        "Topics that will be used as search terms in the FAST and LCSH APIs.  
        - Focus on the main topics of that page specifically, but use broad terms, so that they can be used for search.  
        - Write each search term as a separate string in the list. 
        - Up to 20 terms may be included. 
        - You may include: architectural styles or movements, building types or purposes, key themes or topics discussed, innovations or technologies, architectural features or elements, anything of historical significance, or geographic locations. 
        - Do not include proper names, as these will be captured in the namedEntities field.  
        - For example, you might include terms like 'neoclassical architecture', 'residential buildings', 'urban planning', 'sustainable design', or 'historic preservation' as well as broader terms and some variations to make sure that we return results, like “building design”, "urban design", “construction”, “planning”, “sustainable development”, or “urban infrastructure.”
    ],
    
    "contentWarning": "Assess for content that merits review by another archivist or 'None' if none exists"
}"""
    
    COVER_JSON_FORMAT = """{
    "cleanedText": "The cleaned OCR text with corrections applied",
    "tocEntry": "Start with 'Cover:' followed by a brief description of the cover content, based on the text",
    "namedEntities": ["A list of key entities mentioned in the text, including any significant names, titles, or locations"],
    "subjects": [
        "Topics that will be used as search terms in the FAST and LCSH APIs.  
        - Focus on the main topics of that page specifically, but use broad terms, so that they can be used for search.  
        - Write each search term as a separate string in the list. 
        - Up to 20 terms may be included. 
        - You may include: architectural styles or movements, building types or purposes, key themes or topics discussed, innovations or technologies, architectural features or elements, anything of historical significance, or geographic locations. 
        - Do not include proper names, as these will be captured in the namedEntities field.  
        - For example, you might include terms like 'neoclassical architecture', 'residential buildings', 'urban planning', 'sustainable design', or 'historic preservation' as well as broader terms and some variations to make sure that we return results, like “building design”, "urban design", “construction”, “planning”, “sustainable development”, or “urban infrastructure.”
    ],
    "contentWarning": "Assess for content that merits review by another archivist or 'None' if none exists"
}"""
    
    SHORT_CONTENT_JSON_FORMAT = """{
    "cleanedText": "The cleaned OCR text with corrections applied",
    "tocEntry": "If photography is mentioned in the text, start with 'Photograph:'; if there is no mention of a photo, start with 'Image:' followed by a brief description of what the image likely shows, based on the text",
    "namedEntities": ["A list of key entities mentioned in the text, including architects, locations, and any other significant names"],
    "subjects": [
        "Topics that will be used as search terms in the FAST and LCSH APIs.  
        - Focus on the main topics of that page specifically, but use broad terms, so that they can be used for search.  
        - Write each search term as a separate string in the list. 
        - Up to 20 terms may be included. 
        - You may include: architectural styles or movements, building types or purposes, key themes or topics discussed, innovations or technologies, architectural features or elements, anything of historical significance, or geographic locations. 
        - Do not include proper names, as these will be captured in the namedEntities field.  
        - For example, you might include terms like 'neoclassical architecture', 'residential buildings', 'urban planning', 'sustainable design', or 'historic preservation' as well as broader terms and some variations to make sure that we return results, like “building design”, "urban design", “construction”, “planning”, “sustainable development”, or “urban infrastructure.”
    ],
    "contentWarning": "Assess for content that merits review by another archivist or 'None' if none exists"
}"""
    
    @classmethod
    def get_combined_prompt(cls):
        """Get the comprehensive prompt for normal text analysis."""
        return f"""{cls.COLLECTION_DESCRIPTION}

{cls.OCR_CLEANING_INSTRUCTIONS}

After cleaning the OCR text, return ONLY a JSON response in this exact format:

{cls.GENERIC_JSON_FORMAT}

Focus on capturing the essence of the page's content rather than exhaustive detail.

{cls.ADVERTISEMENT_INSTRUCTIONS}

{cls.CONTENT_WARNING_INSTRUCTIONS}

Return ONLY the JSON response in the exact format specified above."""
    
    @classmethod
    def get_cover_prompt(cls):
        """Get the prompt specifically for cover pages."""
        return f"""{cls.COLLECTION_DESCRIPTION}
This page is the cover of an issue.

{cls.OCR_CLEANING_INSTRUCTIONS}

After cleaning the OCR text, please create the following metadata fields in JSON format:

{cls.COVER_JSON_FORMAT}

Return ONLY the JSON response in the exact format specified above."""
    
    @classmethod
    def get_short_content_prompt(cls):
        """Get the prompt for short content (likely images)."""
        return f"""{cls.COLLECTION_DESCRIPTION}
This page likely contains an image. Consequently, the OCR text for this image is short.

{cls.OCR_CLEANING_INSTRUCTIONS}

After cleaning the OCR text, please create the following metadata fields in JSON format:

{cls.SHORT_CONTENT_JSON_FORMAT}

Return ONLY the JSON response in the exact format specified above."""
    
    @classmethod
    def determine_prompt_type(cls, content, page_number):
        """Determine which prompt to use based on content and page number."""
        is_short = len(content.strip()) < 225
        is_cover = page_number == 1
        
        if is_cover:
            return cls.get_cover_prompt(), "cover"
        elif is_short:
            return cls.get_short_content_prompt(), "short"
        else:
            return cls.get_combined_prompt(), "normal"
    
    # Image analysis prompts
    IMAGE_ANALYSIS_PROMPT = """This image is from 'Southern Architect and Building News', a periodical published from 1892 to 1931 that covered topics of interest to persons in the architecture, building, and hardware trades in the American South. You are creating metadata for this collection for the Architecture and Planning Library Special Collections at University of Texas Libraries in Austin. You are creating this metadata for architectural historians and architectural history students. Architectural styles, movements, trends, and other historically important information should be prioritized. Please analyze this image and return ONLY a JSON response in this exact format:

{
    "textTranscription": "Full, exact transcription of all text visible in the image. 
        - Maintain original spelling, punctuation, and terminology
        - Preserve line breaks and paragraph structure
        - Include ALL text visible including headers, footers, and italicized text
        - Do not correct, modernize, or sanitize language
        - Use [illegible] for text that cannot be read
        - Use [...] for partially visible or cut-off text",
    
    "visualDescription": "Detailed but concise visual description of page. Include:
        - Layout: including design, fonts, special characters,etc.
        - Description of all illustrations and/or photographs
        - Placement and appearance of various elements on the page",
    
    "tocEntry": "A descriptive entry appropriate for a table of contents. Include:
        - Page type (cover, table of contents, advertisement, editorial, article, or other)
        - Specific persons or organizations mentioned (give a brief description of their importance and relevance to the content - for example, "Zaha Hadid (Parametric architecture pioneer)", or "Frank Lloyd Wright (Prairie School architect)")
        - Short description of the content of the page
        - Key topics or themes covered",
    
    "namedEntities": [
        "List key entities including:
        - Architects and architectural firms
        - Geographic locations 
        - Building names
        - Significant people mentioned
        - Organizations
        - Historical events or periods
        - Architectural styles or movements
        - Innovations or technologies referenced
        - Any other notable entities relevant to the content of the page",
        - Write each entity as a separate string in the list
        - Simplify where possible.  No need to include titles.  
    ],
    
    "subjects": [
        "Topics that will be used as search terms in the FAST and LCSH APIs.  
        - Focus on the main topics of that page specifically, but use broad terms, so that they can be used for search.  
        - Write each search term as a separate string in the list. 
        - Up to 20 terms may be included. 
        - You may include: architectural styles or movements, building types or purposes, key themes or topics discussed, innovations or technologies, architectural features or elements, anything of historical significance, or geographic locations. 
        - Do not include proper names, as these will be captured in the namedEntities field.  
        - For example, you might include terms like 'neoclassical architecture', 'residential buildings', 'urban planning', 'sustainable design', or 'historic preservation' as well as broader terms and some variations to make sure that we return results, like “building design”, "urban design", “construction”, “planning”, “sustainable development”, or “urban infrastructure.”
    ],
    
    "contentWarning": "Note potentially sensitive content, or 'None' if none exists. Another archivist will assess if any measures are appropriate, your job is just to note if there is anything that may be concerning. 
    Consider:
        - Biased language or terminology
        - Culturally sensitive material
        - Offensive or harmful imagery or language"
}"""
    
    @classmethod
    def get_image_analysis_prompt(cls):
        """Get the prompt for image analysis."""
        return cls.IMAGE_ANALYSIS_PROMPT
