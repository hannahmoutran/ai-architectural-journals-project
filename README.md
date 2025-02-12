# University of Texas at Austin Libraries
## Southern Architect AI Cataloging Project
### Project Overview
This project focuses on cataloging the University of Texas at Austin's collection of Southern Architect and Building News journals using Large Language Models (LLMs) to assist in the first-pass metadata creation process. This code is a part of a larger research project to explore and evaluate the effectiveness of AI in enhancing cataloging workflows. In the larger project, we are considering both API and web-based interaction with Large Language Models for metadata creation.  

[Southern Architect and Building News](https://collections.lib.utexas.edu/?f%5Bmods_relatedItem_titleInfo_title_source_t%5D%5B%5D=Southern+Architect+and+Building+News) was an illustrated monthly journal devoted to the interests of architects, builders, and the hardware trade. It was published from 1882-1932.  The collection is currently housed in the Architecture and Planning Library Special Collections at the University of Texas Libraries, The University of Texas at Austin. 

### Project Goals
We are testing efficacy in the following tasks:
OCR creation/improvement
Table of contents (TOC) entry creation
Named entity extraction
Subject heading creation

### Output
The workflows generate Excel spreadsheets containing the processed metadata. 

### Project Status
Ongoing. We are currently evaluating the results of processing either OCR texts generated from jpeg images of the sample pages or the images of the pages themselves with the following models: Claude Haiku, Claude 3 Sonnet, Claude Sonnet 3.5, GPT-4o-mini, GPT-4o, Gemini 2.0 Flash Experimental.

### Future updates may include:

Integration with data validation services
Refinement of prompts and/or addition of other metadata fields
Automated evaluation of output quality

### Getting Started
Prerequisites
Python 3.7 or higher
An Integrated Development Environment (IDE) or text editor (e.g., VS Code, PyCharm)
LLM API keys (keep these confidential - save them as environmental variables)

### Installation and Configuration

1. Clone this repository to your local machine.
2. Open a terminal or command prompt and navigate to the project directory.
3. Create an env in your project (I recommend virtualenv myenv)
4. In order to use the programs, you will need to add the following to your shell resources file: OPENAI_API_KEY=your_openai_api_key_here ANTHROPIC_API_KEY=your_anthropic_api_key_here
5. Install the required packages by running: pip install -r requirements.txt

### Running the Workflows
1. Set up the folder structure: Create a main folder to hold all your issues. Inside this main folder, create subfolders for each issue. Place the OCR text files and/or images for each issue in their respective subfolders.
2. Configure the script: Open the script in your preferred text editor. Locate the main function. Change the input_folder and output_folder names to the paths of the appropriate local folders on your machine. 
3. To run a workflow, run the script using your IDE's run command or by typing <python thatparticularcodefilename.py> in the terminal. 

### Changing Models
To change the model, edit the variable at the top of the program.
For example, model = "claude-3-sonnet-20240229" could be changed to model = "claude-3-haiku-20240307".

Find official model names and specifications here:
[Model documentation for Anthropic Models](https://docs.anthropic.com/en/docs/about-claude/models) 
[Model documentation for OpenAI Models](https://platform.openai.com/docs/models)
[Model documentation for Gemini Models](https://ai.google.dev/gemini-api/docs/models/gemini) 

Keep in mind that some models may have different capabilities, token limits, or response formats. Always refer to the respective documentation and adjust your code accordingly when changing models to ensure compatibility and optimal performance.

Remember to keep all API keys and sensitive information confidential. Do not share them or commit them to Github.
This project includes a .gitignore file to prevent unnecessary or sensitive files from being tracked by Git.

### Evaluators
The results of the testing and analysis for the larger research project are still in progress and a collaborative effort with: 
- Devon Murphy, Metadata Analyst
- Karina Sanchez, Scholars Lab Librarian
- Katie Pierce Meyer, Head of Architectural Collections
- Willem Borkgren, Scholars Lab GRA
- Josh Conrad, Digital Initiatives Archival Fellow for the Alexander Architectural Archives

### If you would like to contribute or have questions, please get in touch:
[Hannah Moutran](mailto:hlm2454@my.utexas.edu) | Library Specialist, Applications of AI, UT Libraries

Special thanks to Aaron Choate, Director of Research & Strategy, for his support and guidance. 
