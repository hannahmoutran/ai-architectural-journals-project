# issue-level metadata - using OpenAI's GPT-4o-mini model
import os
import json
import logging
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
import tenacity
from collections import defaultdict
import re

# Import our custom modules
from model_pricing import calculate_cost, get_model_info
from token_logging import create_token_usage_log, log_individual_response

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress verbose HTTP logging
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class APIStats:
    def __init__(self):
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.processing_times = []

api_stats = APIStats()

class IssueSynthesizer:
    """Class to synthesize issue-level descriptions and select top subject headings from chosen vocabulary terms."""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.system_prompt = self.create_system_prompt()
    
    def create_system_prompt(self) -> str:
        """Create the system prompt for issue synthesis."""
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

    def create_user_prompt(self, toc_content: str, selected_terms: List[Dict[str, str]]) -> str:
        formatted_terms = []
        for term in selected_terms:
            formatted_terms.append(f"- {term['label']} ({term['uri']}) [{term['source']}]")
        
        terms_section = "\n".join(formatted_terms)
        
        user_prompt = f"""Analyze this issue content index and create issue metadata:

        ISSUE CONTENT INDEX:
        {toc_content}

        SELECTED SUBJECT HEADINGS:
        {terms_section}

        Create a scholarly description emphasizing this issue's unique architectural content and historical significance. Use the provided subject headings as-is."""
        
        return user_prompt

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        stop=tenacity.stop_after_attempt(5),
        retry=tenacity.retry_if_exception_type(Exception)
    )
    def synthesize_issue(self, toc_content: str, all_chosen_terms: List[Dict[str, str]]) -> Tuple[Dict[str, Any], str, Any, float]:
        """Synthesize issue-level description and select subject headings from chosen vocabulary terms."""
        user_prompt = self.create_user_prompt(toc_content, all_chosen_terms)
        
        api_stats.total_requests += 1
        start_time = time.time()
        
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=2500,
            temperature=0.3  # Slightly higher temperature for more creative synthesis
        )
        
        processing_time = time.time() - start_time
        api_stats.processing_times.append(processing_time)
        
        api_stats.total_input_tokens += response.usage.prompt_tokens
        api_stats.total_output_tokens += response.usage.completion_tokens
        
        raw_response = response.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            parsed_response = self.parse_json_response(raw_response)
            
            # Validate response structure
            if not isinstance(parsed_response, dict):
                raise ValueError("Response is not a JSON object")
            
            if "issue_description" not in parsed_response:
                raise ValueError("Missing 'issue_description' field")
            
            if "selected_subject_headings" not in parsed_response:
                raise ValueError("Missing 'selected_subject_headings' field")
            
            # Validate that we have exactly 10 subject headings
            headings = parsed_response["selected_subject_headings"]
            if not isinstance(headings, list):
                raise ValueError("'selected_subject_headings' must be a list")
            
            if len(headings) != 10:
                logging.warning(f"Expected 10 subject headings, got {len(headings)}")
                # Adjust to exactly 10 - truncate or keep what we have
                if len(headings) > 10:
                    parsed_response["selected_subject_headings"] = headings[:10]
            
            return parsed_response, raw_response, response.usage, processing_time
            
        except Exception as e:
            logging.error(f"Error parsing issue synthesis response: {e}")
            # Return error response
            return {
                "issue_description": f"Error synthesizing issue description: {str(e)}",
                "selected_subject_headings": []
            }, raw_response, response.usage, processing_time

    def parse_json_response(self, raw_response: str) -> Dict[str, Any]:
        """Parse JSON response from the API."""
        # Remove markdown formatting
        cleaned_response = re.sub(r'```json\s*|\s*```', '', raw_response)
        cleaned_response = re.sub(r'^[^{]*({.*})[^}]*$', r'\1', cleaned_response, flags=re.DOTALL)
        
        # Remove trailing commas
        cleaned_response = re.sub(r',(\s*[}\]])', r'\1', cleaned_response)
        
        try:
            parsed_json = json.loads(cleaned_response)
            return parsed_json
            
        except json.JSONDecodeError as e:
            # Try to extract JSON object
            match = re.search(r'{.*}', raw_response, re.DOTALL)
            if match:
                json_str = match.group(0)
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                return json.loads(json_str)
            else:
                raise ValueError(f"Could not parse JSON response: {e}")

class SouthernArchitectIssueSynthesizer:
    """Main class for synthesizing issue-level descriptions and selecting from chosen vocabulary terms."""
    
    def __init__(self, folder_path: str, model_name: str = "gpt-4o-2024-08-06"):
        self.folder_path = folder_path
        self.model_name = model_name
        self.workflow_type = None
        self.json_data = None
        self.toc_file_path = None
        self.synthesizer = IssueSynthesizer(model_name)
    
    def detect_workflow_type(self) -> bool:
        """Detect workflow type and check for required files."""
        # Check for expected files
        text_files = ['text_workflow.xlsx', 'text_workflow.json']
        image_files = ['image_workflow.xlsx', 'image_workflow.json']
        
        has_text_files = all(os.path.exists(os.path.join(self.folder_path, f)) for f in text_files)
        has_image_files = all(os.path.exists(os.path.join(self.folder_path, f)) for f in image_files)
        
        if has_text_files and not has_image_files:
            self.workflow_type = 'text'
        elif has_image_files and not has_text_files:
            self.workflow_type = 'image'
        else:
            logging.error("Could not determine workflow type or multiple workflow files found.")
            return False
        
        # Check if step 3 has been run (page metadata generation)
        page_metadata_folder = os.path.join(self.folder_path, 'page_level_metadata')
        if not os.path.exists(page_metadata_folder):
            logging.error("Page metadata generation (step 3) must be run before step 4.")
            return False
        
        return True

    def find_issue_content_index_file(self) -> bool:
        """Find the issue content index file created in step 3."""
        # Look for files ending with "_Issue_Content_Index.txt"
        issue_content_index_files = [f for f in os.listdir(self.folder_path) if f.endswith('_Issue_Content_Index.txt')]

        if not issue_content_index_files:
            logging.error("No issue content index file found. Please run step 3 first.")
            return False

        if len(issue_content_index_files) > 1:
            logging.warning(f"Multiple issue content index files found: {issue_content_index_files}. Using the first one.")

        self.issue_content_index_file_path = os.path.join(self.folder_path, issue_content_index_files[0])
        print(f"📋 Found issue content index: {issue_content_index_files[0]}")
        return True
    
    def load_json_data(self) -> bool:
        """Load JSON data to extract chosen vocabulary terms."""
        json_filename = f"{self.workflow_type}_workflow.json"
        json_path = os.path.join(self.folder_path, json_filename)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.json_data = json.load(f)
            print(f"📊 Loaded JSON data from {json_filename}")
            return True
        except Exception as e:
            logging.error(f"Error loading JSON data: {e}")
            return False
    
    def extract_all_chosen_terms(self) -> List[Dict[str, str]]:
        """Extract all unique chosen vocabulary terms from the JSON data."""
        all_terms = []
        seen_uris = set()
        
        # Skip the last item if it's API stats
        data_items = self.json_data[:-1] if self.json_data and 'api_stats' in self.json_data[-1] else self.json_data
        
        for item in data_items:
            if 'analysis' in item:
                analysis = item['analysis']
                
                # Get chosen vocabulary terms from step 3
                vocabulary_terms = analysis.get('final_selected_terms', [])
                for term in vocabulary_terms:
                    if isinstance(term, dict) and 'uri' in term and term['uri'] not in seen_uris:
                        all_terms.append({
                            'label': term.get('label', ''),
                            'uri': term.get('uri', ''),
                            'source': term.get('source', 'Unknown')
                        })
                        seen_uris.add(term['uri'])
        
        return all_terms
    
    def extract_geographic_terms(self) -> List[Dict[str, Any]]:
        """Extract all unique geographic terms from the JSON data with counts."""
        geographic_terms_count = {}
        
        # Skip the last item if it's API stats
        data_items = self.json_data[:-1] if self.json_data and 'api_stats' in self.json_data[-1] else self.json_data
        
        for item in data_items:
            if 'analysis' in item:
                analysis = item['analysis']
                
                # Get geographic vocabulary search results
                geo_results = analysis.get('geographic_vocabulary_search_results', {})
                
                for location_key, terms_list in geo_results.items():
                    if isinstance(terms_list, list):
                        for term in terms_list:
                            if isinstance(term, dict) and 'uri' in term:
                                uri = term.get('uri', '')
                                if uri:
                                    if uri not in geographic_terms_count:
                                        geographic_terms_count[uri] = {
                                            'label': term.get('label', ''),
                                            'uri': uri,
                                            'source': term.get('source', 'Unknown'),
                                            'count': 0
                                        }
                                    geographic_terms_count[uri]['count'] += 1
        
        # Convert to list and sort by count (descending) then by label
        geographic_terms = list(geographic_terms_count.values())
        geographic_terms.sort(key=lambda x: (-x['count'], x['label']))
        
        return geographic_terms

    def read_issue_content_index(self) -> str:
        """Read the issue content index."""
        try:
            with open(self.issue_content_index_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            logging.error(f"Error reading issue content index file: {e}")
            return ""
    
    def get_issue_name(self) -> str:
        """Extract issue name from the issue content index file name."""
        issue_content_index_filename = os.path.basename(self.issue_content_index_file_path)
        # Remove "_Issue_Content_Index.txt" suffix
        issue_name = issue_content_index_filename.replace('_Issue_Content_Index.txt', '')
        return issue_name
    
    def create_issue_metadata_file(self, synthesis_result: Dict[str, Any]) -> bool:
        """Create clean issue metadata file without processing information."""
        try:
            issue_name = self.get_issue_name()
            metadata_filename = f"{issue_name}_Issue_Metadata.txt"
            metadata_path = os.path.join(self.folder_path, metadata_filename)
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                f.write(f"{issue_name} - Issue Metadata\n")
                f.write("=" * (len(issue_name) + 17) + "\n\n")
                
                # Issue description
                f.write("ISSUE DESCRIPTION:\n")
                f.write("-" * 20 + "\n")
                f.write(f"{synthesis_result['issue_description']}\n\n")
                
                # Selected subject headings
                selected_headings = synthesis_result.get('selected_subject_headings', [])
                f.write(f"SELECTED SUBJECT HEADINGS ({len(selected_headings)} terms):\n")
                f.write("-" * 35 + "\n")
                
                for i, heading in enumerate(selected_headings, 1):
                    if isinstance(heading, dict):
                        label = heading.get('label', 'Unknown')
                        source = heading.get('source', 'Unknown')
                        f.write(f"{i:2d}. {label} [{source}]\n")
                    else:
                        # Handle string format as fallback
                        f.write(f"{i:2d}. {heading}\n")
                
                f.write("\n" + "=" * (len(issue_name) + 17) + "\n")
            
            print(f"📄 Created issue metadata: {metadata_filename}")
            return True
            
        except Exception as e:
            logging.error(f"Error creating issue metadata file: {e}")
            return False
    
    def append_geographic_terms_to_file(self, geographic_terms: List[Dict[str, Any]]) -> bool:
        """Append geographic terms to the existing issue metadata file."""
        try:
            issue_name = self.get_issue_name()
            metadata_filename = f"{issue_name}_Issue_Metadata.txt"
            metadata_path = os.path.join(self.folder_path, metadata_filename)
            
            # Check if file exists
            if not os.path.exists(metadata_path):
                logging.error(f"Issue metadata file not found: {metadata_filename}")
                return False
            
            # Calculate total mentions
            total_mentions = sum(term.get('count', 1) for term in geographic_terms)
            
            # Append geographic terms to the existing file
            with open(metadata_path, 'a', encoding='utf-8') as f:
                f.write(f"\nGEOGRAPHIC TERMS ({len(geographic_terms)} unique terms, {total_mentions} total mentions):\n")
                f.write("-" * 60 + "\n")
                
                for i, term in enumerate(geographic_terms, 1):
                    label = term.get('label', 'Unknown')
                    uri = term.get('uri', '')
                    source = term.get('source', 'Unknown')
                    count = term.get('count', 1)
                    
                    # Format with count
                    if count > 1:
                        f.write(f"{i:2d}. {label} [{source}] ({count} mentions)\n")
                    else:
                        f.write(f"{i:2d}. {label} [{source}]\n")
                    
                    if uri:  # Only show URI if it exists
                        f.write(f"    URI: {uri}\n")
                
                f.write("\n" + "=" * (len(issue_name) + 17) + "\n")
            
            print(f"📍 Appended {len(geographic_terms)} unique geographic terms ({total_mentions} total mentions) to issue metadata")
            return True
            
        except Exception as e:
            logging.error(f"Error appending geographic terms to file: {e}")
            return False
    
    def update_json_with_synthesis(self, synthesis_result: Dict[str, Any]) -> bool:
        """Update the JSON file with issue-level synthesis."""
        try:
            # Add issue-level synthesis to the JSON data
            issue_synthesis = {
                "issue_synthesis": {
                    "issue_description": synthesis_result['issue_description'],
                    "selected_subject_headings": synthesis_result['selected_subject_headings'],
                    "generated_date": datetime.now().isoformat(),
                    "model_used": self.model_name
                }
            }
            
            # Add synthesis to the JSON data
            self.json_data.append(issue_synthesis)
            
            # Save updated JSON
            json_filename = f"{self.workflow_type}_workflow.json"
            json_path = os.path.join(self.folder_path, json_filename)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.json_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Updated JSON file with issue synthesis")
            return True
            
        except Exception as e:
            logging.error(f"Error updating JSON with synthesis: {e}")
            return False
    
    def run(self) -> bool:
        """Main execution method."""
        print(f"\n🎯 SOUTHERN ARCHITECT STEP 4 - ISSUE SYNTHESIS")
        print(f"📁 Processing folder: {self.folder_path}")
        print(f"🤖 Model: {self.model_name}")
        print("-" * 50)
        
        # Detect workflow type
        if not self.detect_workflow_type():
            return False
        
        print(f"🔍 Detected workflow type: {self.workflow_type.upper()}")
        
        # Find issue content index file
        if not self.find_issue_content_index_file():
            return False
        
        # Load JSON data
        if not self.load_json_data():
            return False
        
        # Extract all chosen vocabulary terms
        all_chosen_terms = self.extract_all_chosen_terms()
        if not all_chosen_terms:
            print("⚠️  No chosen vocabulary terms found in the data")
            return False
        
        print(f"📚 Found {len(all_chosen_terms)} unique chosen vocabulary terms")
        
        # Show breakdown by source
        terms_by_source = defaultdict(int)
        for term in all_chosen_terms:
            source = term.get('source', 'Unknown')
            terms_by_source[source] += 1
        
        for source, count in terms_by_source.items():
            print(f"   - {source}: {count} terms")
        
        # Extract geographic terms
        geographic_terms = self.extract_geographic_terms()
        total_geo_mentions = sum(term.get('count', 1) for term in geographic_terms)
        print(f"📍 Found {len(geographic_terms)} unique geographic terms ({total_geo_mentions} total mentions)")
        
        # Show geographic terms breakdown by source
        if geographic_terms:
            geo_by_source = defaultdict(int)
            for term in geographic_terms:
                source = term.get('source', 'Unknown')
                geo_by_source[source] += 1
            
            for source, count in geo_by_source.items():
                print(f"   - {source}: {count} unique geographic terms")
            
            # Show most frequently mentioned terms
            frequent_terms = [term for term in geographic_terms if term.get('count', 1) > 1]
            if frequent_terms:
                print(f"   - {len(frequent_terms)} terms mentioned multiple times")
                for term in frequent_terms[:3]:  # Show top 3
                    print(f"     • {term['label']}: {term['count']} mentions")

        # Read issue content index content
        issue_content_index = self.read_issue_content_index()
        if not issue_content_index:
            return False
        
        # Show model pricing info
        model_info = get_model_info(self.model_name)
        if model_info:
            print(f"💰 Pricing: ${model_info['input_per_1k']:.5f}/1K input, ${model_info['output_per_1k']:.5f}/1K output")
        
        # Synthesize issue description and select vocabulary terms
        print(f"\n🔄 Synthesizing issue description and selecting vocabulary terms...")
        synthesis_result, raw_response, usage, processing_time = self.synthesizer.synthesize_issue(
            issue_content_index, all_chosen_terms
        )
        
        if not synthesis_result:
            print("❌ Issue synthesis failed")
            return False
        
        # Create clean issue metadata file
        if not self.create_issue_metadata_file(synthesis_result):
            return False
        
        # Append geographic terms to the file (after LLM synthesis is complete)
        if geographic_terms:
            if not self.append_geographic_terms_to_file(geographic_terms):
                print("⚠️  Warning: Failed to append geographic terms to metadata file")
        else:
            print("ℹ️  No geographic terms found to append")
        
        # Update JSON with synthesis
        if not self.update_json_with_synthesis(synthesis_result):
            return False
        
        # Create logs folder and log response
        logs_folder_path = os.path.join(self.folder_path, "logs")
        if not os.path.exists(logs_folder_path):
            os.makedirs(logs_folder_path)
        
        # Log individual response
        issue_name = self.get_issue_name()
        log_individual_response(
            logs_folder_path=logs_folder_path,
            script_name="southern_architect_issue_synthesis",
            row_number=1,
            barcode=issue_name,
            response_text=raw_response,
            model_name=self.model_name,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            processing_time=processing_time
        )
        
        # Calculate cost
        estimated_cost = calculate_cost(
            model_name=self.model_name,
            prompt_tokens=api_stats.total_input_tokens,
            completion_tokens=api_stats.total_output_tokens,
            is_batch=False
        )
        
        create_token_usage_log(
            logs_folder_path=logs_folder_path,
            script_name="southern_architect_issue_synthesis",
            model_name=self.model_name,
            total_items=1,
            items_with_issues=0,
            total_time=processing_time,
            total_prompt_tokens=api_stats.total_input_tokens,
            total_completion_tokens=api_stats.total_output_tokens,
            additional_metrics={
                "Issue name": issue_name,
                "Available chosen terms": len(all_chosen_terms),
                "Selected subject headings": len(synthesis_result.get('selected_subject_headings', [])),
                "Geographic terms found": len(geographic_terms),
                "Total geographic mentions": sum(term.get('count', 1) for term in geographic_terms),
                "Processing mode": "INDIVIDUAL",
                "Actual cost": f"${estimated_cost:.4f}",
                "Description length": f"{len(synthesis_result['issue_description'])} characters",
                "LCSH terms available": terms_by_source.get('LCSH', 0),
                "FAST terms available": terms_by_source.get('FAST', 0),
                "Getty terms available": terms_by_source.get('Getty AAT', 0) + terms_by_source.get('Getty TGN', 0)
            }
        )
        
        # Final summary
        selected_headings = synthesis_result.get('selected_subject_headings', [])
        print(f"\n🎉 ISSUE SYNTHESIS COMPLETED!")
        print(f"📝 Issue description: {len(synthesis_result['issue_description'])} characters")
        print(f"📚 Selected {len(selected_headings)} subject headings from {len(all_chosen_terms)} available chosen terms")
        print(f"📍 Appended {len(geographic_terms)} unique geographic terms ({sum(term.get('count', 1) for term in geographic_terms)} total mentions)")
        print(f"⏱️  Processing time: {processing_time:.2f} seconds")
        print(f"🎯 Total tokens: {api_stats.total_input_tokens + api_stats.total_output_tokens:,}")
        print(f"💰 Estimated cost: ${estimated_cost:.4f}")
        print(f"\n📁 GENERATED FILES:")
        print(f"  📄 Issue metadata: {os.path.join(self.folder_path, f'{issue_name}_Issue_Metadata.txt')}")
        print(f"  📊 Updated JSON: {os.path.join(self.folder_path, f'{self.workflow_type}_workflow.json')}")
        
        return True

def find_newest_folder(base_directory: str) -> Optional[str]:
    """Find the newest folder in the base directory."""
    if not os.path.exists(base_directory):
        return None
    
    folders = [f for f in os.listdir(base_directory) 
              if os.path.isdir(os.path.join(base_directory, f))]
    
    if not folders:
        return None
    
    # Sort by modification time (newest first)
    folders.sort(key=lambda x: os.path.getmtime(os.path.join(base_directory, x)), reverse=True)
    
    return os.path.join(base_directory, folders[0])

def main():
    parser = argparse.ArgumentParser(description='Synthesize issue-level descriptions and select from chosen vocabulary terms')
    parser.add_argument('--folder', help='Specific folder path to process')
    parser.add_argument('--newest', action='store_true', help='Process the newest folder in the output directory (default: True if no folder specified)')
    parser.add_argument('--model', default="gpt-4o-2024-08-06", help='Model name to use for synthesis')
    args = parser.parse_args()
    
    # Default base directory for Southern Architect output folders
    base_output_dir = "/Users/hannahmoutran/Desktop/southern_architect/CODE/output_folders"
    
    if args.folder:
        if not os.path.exists(args.folder):
            print(f"❌ Folder not found: {args.folder}")
            return 1
        folder_path = args.folder
    else:
        # Default to newest folder if no specific folder provided
        folder_path = find_newest_folder(base_output_dir)
        if not folder_path:
            print(f"❌ No folders found in: {base_output_dir}")
            return 1
        print(f"🔄 Auto-selected newest folder: {os.path.basename(folder_path)}")
    
    # Create and run the synthesizer
    synthesizer = SouthernArchitectIssueSynthesizer(folder_path, args.model)
    success = synthesizer.run()
    
    if not success:
        print("❌ Issue synthesis failed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())