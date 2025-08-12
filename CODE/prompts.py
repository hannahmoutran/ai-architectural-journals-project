"""
Southern Architect Prompts Module

This module contains all prompt components for the Southern Architect archival project workflow.
All components are defined once and assembled into the original method names.

Workflow Steps:
- Step 1: Initial metadata extraction (text and image analysis)
- Step 3: Vocabulary selection from controlled vocabularies  
- Step 4: Issue-level synthesis and subject heading selection
"""

class SouthernArchitectPrompts:
# Container for all Southern Architect workflow prompts.
    
    # ==================== STEP 1 PROMPTS (INITIAL METADATA EXTRACTION) ====================
    
    # Base description used in all prompts
    COLLECTION_DESCRIPTION = """
You are an archivist at the University of Texas at Austin, cataloging a collection of issues of The Southern Architect (1892-1931). You are working from OCR text that may contain errors, so use your judgment to decipher meaning. This metadata is for architectural historians and architectural history students. Architectural styles, movements, trends, and other historically important information should be prioritized.
"""

    # OCR cleaning instructions (text prompts only)
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
    
    # Advertisement identification
    ADVERTISEMENT_INSTRUCTIONS = """
How to identify advertisements:
1. Look for company names prominently displayed
2. Check for product descriptions or benefits
3. Look for contact information or calls to action
4. Be aware of persuasive language or marketing slogans
5. Note any pricing information or special offers
"""
    
    # Content warning assessment instructions
    CONTENT_WARNING_INSTRUCTIONS = """
After analyzing the content, assess whether it contains any language or themes that might warrant a content warning. Consider the following categories:
1. Racist/ethnically insensitive language
2. Sexist/gender-discriminatory content
3. Violence/graphic descriptions
4. Other potentially offensive/outdated terminology

If you identify any such content, briefly describe it using the categorization described above.
If no sensitive content is present, simply return 'None'.
"""
    
    # ==================== FIELD DEFINITIONS ====================
    
    # Cleaned text field (text prompts only)
    CLEANED_TEXT_FIELD = [
        "cleanedText",
        "The cleaned OCR text with corrections applied"
    ]
    
    # Text transcription field (image prompts only)  
    TEXT_TRANSCRIPTION_FIELD = [
        "textTranscription", 
        "Full, exact transcription of all text visible in the image",
        [
            "Maintain original spelling, punctuation, and terminology",
            "Preserve line breaks and paragraph structure", 
            "Include ALL text visible including headers, footers, and italicized text",
            "Do not correct, modernize, or sanitize language",
            "Use [illegible] for text that cannot be read",
            "Use [...] for partially visible or cut-off text"
        ]
    ]
    
    # Visual description field (image prompts only)
    VISUAL_DESCRIPTION_FIELD = [
        "visualDescription",
        "Detailed but concise visual description of page",
        [
            "Layout: including design, fonts, special characters, etc.",
            "Description of all illustrations and/or photographs", 
            "Placement and appearance of various elements on the page"
        ]
    ]
    
    # TOC entry variations
    TOC_ENTRY_COVER = [
        "tocEntry",
        "Start with 'Cover:' followed by a brief description of the cover content, based on the text."
    ]
    
    TOC_ENTRY_SHORT = [
        "tocEntry", 
        "If photography is mentioned in the text, start with 'Photograph:'; if there is no mention of a photo, start with 'Image:' followed by a brief description of what the image likely shows, based on the text"
    ]
    
    TOC_ENTRY_NORMAL = [
        "tocEntry",
        "A descriptive entry appropriate for a table of contents",
        [
            "Page type (cover, table of contents, advertisement, editorial, article, photo, image, or other)",
            "Short description of the content of the page, no more than 1-2 sentences.",
            "Specific persons or organizations mentioned (give a brief description of their importance and relevance to the content - for example, 'Zaha Hadid (Parametric architecture pioneer)', or 'Frank Lloyd Wright (Prairie School architect)')",
            "Include key topics, themes, technologies, architectural styles, or anything of historical significance covered on page"
        ]
    ]
    
    # Named entities field (all prompts)
    NAMED_ENTITIES_FIELD = [
        "namedEntities",
        "List non-geographic entities with type in parentheses",
        [
            "Format: 'Frank Lloyd Wright (Architect)', 'Empire State Building (Building)', 'Mary Johnson (Person)', 'Smith & Associates (Firm)', 'American Red Cross (Organization)', etc.",
            "Types: (Architect), (Firm), (Person), (Building), (Organization), (Style), (Material), (Event), (Publication), (School), (Award), (Competition), (Project)",
            "Do NOT include geographic locations here - use geographicEntities field instead",
            "Limit to entities that are central to the content of the page and/or historically significant"
        ]
    ]
    
    # Geographic entities field (text prompts)
    GEOGRAPHIC_ENTITIES_FIELD_TEXT = [
        "geographicEntities",
        "Geographic references mentioned in the text",
        [
            "Format examples:",
            "- 'City--State (City)' for U.S. Cities e.g. 'New York--New York (City)'",
            "- 'City--Province' for Canadian cities e.g. 'Toronto--Ontario (City)'", 
            "- 'City--Country' for all other Non-U.S. cities e.g. 'London--England (City)'",
            "- 'State (State)' for U.S. States e.g. 'Tennessee (State)'",
            "- 'Province/State--Country (Province/State)' for non-U.S. Provinces or States e.g. 'Quebec--Canada (Province)'",
            "- 'Country (Country)' for Countries e.g. 'France (Country)'",
            "Include all geographic references mentioned in the text",
            "Always include full names: 'Georgia' not 'Ga', 'Maryland' not 'Md', 'United States' not 'US'"
        ]
    ]
    
    # Geographic entities field (image prompts - includes "or images")
    GEOGRAPHIC_ENTITIES_FIELD_IMAGE = [
        "geographicEntities", 
        "Geographic references mentioned in the text or images",
        [
            "Format examples:",
            "- 'City--State (City)' for U.S. Cities e.g. 'New York--New York (City)'",
            "- 'City--Province' for Canadian cities e.g. 'Toronto--Ontario (City)'",
            "- 'City--Country' for all other Non-U.S. cities e.g. 'London--England (City)'", 
            "- 'State (State)' for U.S. States e.g. 'Tennessee (State)'",
            "- 'Province/State--Country (Province/State)' for non-U.S. Provinces or States e.g. 'Quebec--Canada (Province)'",
            "- 'Country (Country)' for Countries e.g. 'France (Country)'",
            "Include all geographic references mentioned in the text or images",
            "Always include full names: 'Georgia' not 'Ga', 'Maryland' not 'Md', 'United States' not 'US'"
        ]
    ]
    
    # Subjects/topics field (all prompts)
    SUBJECTS_FIELD = [
        "subjects",
        "Topics that will be used as search terms in controlled vocabulary APIs",
        [
            "Focus on specific but searchable words or phrases as topics",
            "Cover the breadth of what is discussed on the page",
            "Write each search term as a separate string in the list",
            "Up to 8 terms may be included",
            "Priority topics: architectural styles or movements, distinctive architectural features or elements, building types or purposes, key themes or topics discussed, innovations or technologies, anything of historical significance",
            "For architectural content, identify specific style names, construction techniques, and prominent design elements visible in images or discussed in text",
            "Do not include proper names or geographic locations, as these will be captured in the namedEntities and geographicEntities fields"
        ]
    ]
    
    # Content warning field (text prompts)
    CONTENT_WARNING_FIELD_TEXT = [
        "contentWarning",
        "Note potentially sensitive content, or 'None' if none exists. Another archivist will assess if any measures are appropriate, your job is just to note if there is anything that may be concerning",
        [
            "Consider: Biased language or terminology",
            "Consider: Culturally sensitive material", 
            "Consider: Offensive or harmful language"
        ]
    ]
    
    # Content warning field (image prompts - mentions imagery)
    CONTENT_WARNING_FIELD_IMAGE = [
        "contentWarning",
        "Note potentially sensitive content, or 'None' if none exists. Another archivist will assess if any measures are appropriate, your job is just to note if there is anything that may be concerning",
        [
            "Consider: Biased language or terminology",
            "Consider: Culturally sensitive material", 
            "Consider: Offensive or harmful imagery or language"
        ]
    ]
    
    # ==================== HELPER METHODS ====================
    
    @classmethod
    def _format_field_as_json_structure(cls, field_definition):
        """Convert list-based field definition to JSON structure format."""
        field_name = field_definition[0]
        field_description = field_definition[1]
        
        if len(field_definition) > 2 and isinstance(field_definition[2], list):
            # Field has sub-items
            sub_items = field_definition[2]
            formatted_items = []
            for item in sub_items:
                formatted_items.append(f"        - {item}")
            sub_items_str = "\n".join(formatted_items)
            return f'    "{field_name}": "{field_description}. Include:\n{sub_items_str}"'
        else:
            # Simple field
            return f'    "{field_name}": "{field_description}"'
    
    @classmethod
    def _create_json_format(cls, field_list):
        """Create JSON format string from list of field definitions."""
        formatted_fields = []
        for field_def in field_list:
            formatted_fields.append(cls._format_field_as_json_structure(field_def))
        return "{\n" + ",\n    \n".join(formatted_fields) + "\n}"
    
    # ==================== METHOD IMPLEMENTATIONS ====================
    
    @classmethod
    def get_combined_prompt(cls):
        """Get the comprehensive prompt for normal text analysis."""
        field_list = [
            cls.CLEANED_TEXT_FIELD,
            cls.TOC_ENTRY_NORMAL,
            cls.NAMED_ENTITIES_FIELD,
            cls.GEOGRAPHIC_ENTITIES_FIELD_TEXT,
            cls.SUBJECTS_FIELD,
            cls.CONTENT_WARNING_FIELD_TEXT
        ]
        
        json_format = cls._create_json_format(field_list)
        
        return f"""{cls.COLLECTION_DESCRIPTION}

{cls.OCR_CLEANING_INSTRUCTIONS}

After cleaning the OCR text, return ONLY a JSON response in this exact format:

{json_format}

Focus on capturing the essence of the page's content rather than exhaustive detail.

{cls.ADVERTISEMENT_INSTRUCTIONS}

{cls.CONTENT_WARNING_INSTRUCTIONS}

Return ONLY the JSON response in the exact format specified above."""
    
    @classmethod
    def get_cover_prompt(cls):
        """Get the prompt specifically for cover pages."""
        field_list = [
            cls.CLEANED_TEXT_FIELD,
            cls.TOC_ENTRY_COVER,
            cls.NAMED_ENTITIES_FIELD,
            cls.GEOGRAPHIC_ENTITIES_FIELD_TEXT,
            cls.SUBJECTS_FIELD,
            cls.CONTENT_WARNING_FIELD_TEXT
        ]
        
        json_format = cls._create_json_format(field_list)
        
        return f"""{cls.COLLECTION_DESCRIPTION}
This page is the cover of an issue.

{cls.OCR_CLEANING_INSTRUCTIONS}

After cleaning the OCR text, please create the following metadata fields in JSON format:

{json_format}

Return ONLY the JSON response in the exact format specified above."""
    
    @classmethod
    def get_short_content_prompt(cls):
        """Get the prompt for short content (likely images)."""
        field_list = [
            cls.CLEANED_TEXT_FIELD,
            cls.TOC_ENTRY_SHORT,
            cls.NAMED_ENTITIES_FIELD,
            cls.GEOGRAPHIC_ENTITIES_FIELD_TEXT,
            cls.SUBJECTS_FIELD,
            cls.CONTENT_WARNING_FIELD_TEXT
        ]
        
        json_format = cls._create_json_format(field_list)
        
        return f"""{cls.COLLECTION_DESCRIPTION}
This page likely contains an image. Consequently, the OCR text for this image is short.

{cls.OCR_CLEANING_INSTRUCTIONS}

After cleaning the OCR text, please create the following metadata fields in JSON format:

{json_format}

Return ONLY the JSON response in the exact format specified above."""
    
    @classmethod
    def get_image_analysis_prompt(cls):
        """Get the prompt for image analysis."""
        field_list = [
            cls.TEXT_TRANSCRIPTION_FIELD,
            cls.VISUAL_DESCRIPTION_FIELD,
            cls.TOC_ENTRY_NORMAL,
            cls.NAMED_ENTITIES_FIELD,
            cls.GEOGRAPHIC_ENTITIES_FIELD_IMAGE,
            cls.SUBJECTS_FIELD,
            cls.CONTENT_WARNING_FIELD_IMAGE
        ]
        
        json_format = cls._create_json_format(field_list)
        
        return f"""This image is from 'Southern Architect and Building News', a periodical published from 1892 to 1931 that covered topics of interest to persons in the architecture, building, and hardware trades in the American South. You are creating metadata for this collection for the Architecture and Planning Library Special Collections at University of Texas Libraries in Austin. You are creating this metadata for architectural historians and architectural history students. Architectural styles, movements, trends, and other historically important information should be prioritized. Please analyze this image and return ONLY a JSON response in this exact format:

{json_format}"""
    
    @classmethod
    def determine_prompt_type(cls, content, page_number, filename=None):
        """Determine which prompt to use based on content and page number."""
        is_short = len(content.strip()) < 225
        is_cover = "cover" in filename.lower() if filename else False
        
        if is_cover:
            return cls.get_cover_prompt(), "cover"
        elif is_short:
            return cls.get_short_content_prompt(), "short"
        else:
            return cls.get_combined_prompt(), "normal"
    
    # ==================== STEP 3 PROMPT (VOCABULARY SELECTION) ====================
        
    @classmethod
    def get_vocabulary_selection_system_prompt(cls):
        """Get the system prompt for vocabulary selection (step 3)."""
        return """
        This metadata is for a page from 'Southern Architect and Building News', a periodical published from 1892 to 1931 that covered topics of interest to persons in the architecture, building, and hardware trades in the American South. You are a professional librarian specializing in controlled vocabularies for architectural history. You work for the Architecture and Planning Library Special Collections at University of Texas Libraries in Austin. You are formalizing this metadata for architectural historians and architectural history students.

        CRITICAL CONTEXT: This content is from the early 20th century American South (1892-1931). Only select terms that are historically appropriate for this period and geographically relevant to the Southern United States.

        SELECTION CRITERIA:
        1. TEMPORAL ACCURACY: Terms must be appropriate for 1892-1931 architectural practices
        2. GEOGRAPHICAL RELEVANCE: Focus on American South; avoid other regional references unless explicitly mentioned
        3. CONTENT RELEVANCE: Terms must directly relate to what's actually described in the page summary
        4. PRECISION: Choose specific terms over general ones when available
        5. QUALITY CONTROL: Select NOTHING rather than forcing poor matches

        DECISION PROCESS:
        For each topic, evaluate whether ANY available terms genuinely describe the specific page content:
        - RELEVANCE CHECK: Does the term directly and accurately describe what's actually described in the page summary?
        - FIELD MATCH: Does the term match the subject field of the content (architectural vs. literary vs. other)?
        - HISTORICAL CONTEXT: Is this appropriate for 1892-1931 American Southern architecture?
        - GEOGRAPHICAL CONTEXT: Does this fit American South context (avoid California, Northeast, etc.)?
        - SPECIFICITY: Choose the most specific accurate term if multiple apply

        AVOID selecting terms that are:
        - Geographically incorrect (e.g., "Southern California Presbyterian Homes" for general Southern residential content)
        - Temporally inappropriate (modern movements, contemporary terminology)
        - Institutionally irrelevant (specific institution types not mentioned in content)
        - From different subject fields (e.g., literary terms for architectural content)
        - Representative of different content types (e.g., novels about architecture vs. actual architectural descriptions)
        - Based on partial word matches rather than actual content relevance
        - Overly broad when specific terms are available

        DECISION RULES:
        - If a topic has NO genuinely relevant terms, SKIP that entire topic
        - Do not force selections based on partial word matches
        - Better to select fewer accurate terms than many irrelevant ones
        - If multiple terms are exactly the same, select the one with the best source (Getty AAT > LCSH > Getty TGN > FAST)
        - For institutional content, only select institution-specific terms if the page specifically discusses that type of institution

        You will be provided with page content and available vocabulary terms organized by topic. For each relevant topic, select the most appropriate term that accurately represents the page content AND is historically/geographically appropriate.

        Return JSON format:
        {
        "selected_terms": [
            {
            "label": "Exact label",
            "source": "LCSH/FAST/Getty AAT/Getty TGN", 
            "reasoning": "Brief explanation of relevance and historical/geographical appropriateness"
            }
        ]
        }
        """

    # ==================== STEP 4 PROMPT (ISSUE SYNTHESIS) ====================
    
    @classmethod
    def get_issue_synthesis_system_prompt(cls):
        """Get the system prompt for issue synthesis (step 4)."""
        return """You are an archivist at UT Austin cataloging The Southern Architect (1892-1931) for architectural historians and students. Create metadata for this complete issue that emphasizes its unique architectural and historical content.

        TASK: Synthesize an issue-level description and select up to 10 subject headings from the provided chosen vocabulary terms.

        ISSUE DESCRIPTION GUIDELINES:
        - Write 150-250 words from a modern historian's perspective
        - Focus on SPECIFIC details: architect names, firms, buildings, cities, projects, competitions, events
        - Emphasize architectural styles, building types, construction technologies, materials
        - Highlight historically significant innovations or trends
        - Contextualize within American South architectural history (1892-1931)
        - Use scholarly tone; avoid generic statements that could apply to any issue
        - Write as "This issue features..." not "The issue includes..."

        SUBJECT HEADING SELECTION:
        - Select exactly 10 terms from provided chosen vocabulary
        - Prioritize architectural styles, building types, construction technologies
        - Focus on terms most valuable for architectural history research
        - Balance across vocabulary sources when possible

        Return JSON format:
        {
        "issue_description": "Specific description emphasizing unique architectural content and historical significance",
        "selected_subject_headings": [
            {
            "label": "Term label",
            "uri": "Term URI", 
            "source": "LCSH/FAST/Getty AAT/Getty TGN",
            "reasoning": "Why this term represents the issue"
            }
        ]
        }"""