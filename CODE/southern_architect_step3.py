# This code sends metadata, including verified LCSH terms, to an LLM to verify their relevancy to each page.
import os
import json
import logging
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
import tenacity
from openpyxl import load_workbook
from openpyxl.styles import Alignment
from collections import defaultdict

# Import our custom modules
from model_pricing import calculate_cost, get_model_info
from token_logging import create_token_usage_log, log_individual_response
from batch_processor import BatchProcessor

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
        self.total_cached_tokens = 0
        self.processing_times = []

api_stats = APIStats()

class LCSHSelector:
    """Class to handle LCSH selection using OpenAI API."""
    
    def __init__(self, model_name: str = "gpt-4o-mini-2024-07-18"):
        self.model_name = model_name
        self.system_prompt = self.create_system_prompt()
    
    def create_system_prompt(self) -> str:
        """Create the system prompt for LCSH selection."""
        return """You are a professional librarian and metadata expert specializing in Library of Congress Subject Headings (LCSH). Your task is to analyze archival content and select the most appropriate LCSH terms that best describe the content.

You will be provided with:
1. The full context of an archival item (cleaned text, visual description, etc.)
2. A list of available LCSH terms with their URIs
3. The original subject headings that were extracted

Your job is to:
1. Carefully analyze the content to understand what it's about
2. Select ONLY the LCSH terms that are most relevant and appropriate
3. Prioritize precision over recall - it's better to select fewer, more accurate terms
4. Consider both the main topics and any significant secondary subjects
5. Avoid selecting terms that are too broad or too narrow for the content

Return your response as a JSON object with this structure:
{
  "selected_lcsh_terms": [
    {
      "label": "Selected LCSH term",
      "uri": "http://id.loc.gov/authorities/subjects/sh123456",
      "reasoning": "Brief explanation of why this term was selected"
    }
  ],
  "rejected_terms": [
    {
      "label": "Rejected term",
      "reasoning": "Brief explanation of why this term was not selected"
    }
  ]
}

Guidelines:
- Select 1-5 terms maximum per item
- Each selected term should have clear relevance to the content
- Provide brief reasoning for your selections
- Be conservative - only select terms you're confident about"""

    def create_user_prompt(self, entry_data: Dict[str, Any]) -> str:
        """Create the user prompt for a specific entry."""
        analysis = entry_data.get('analysis', {})
        
        # Build content description
        content_parts = []
        
        # Add main content
        if analysis.get('cleaned_text'):
            content_parts.append(f"CLEANED TEXT:\n{analysis['cleaned_text']}")
        elif analysis.get('text_transcription'):
            content_parts.append(f"TEXT TRANSCRIPTION:\n{analysis['text_transcription']}")
        
        # Add visual description for images
        if analysis.get('visual_description'):
            content_parts.append(f"VISUAL DESCRIPTION:\n{analysis['visual_description']}")
        
        # Add TOC entry
        if analysis.get('toc_entry'):
            content_parts.append(f"TOC ENTRY:\n{analysis['toc_entry']}")
        
        # Add named entities
        if analysis.get('named_entities'):
            entities = analysis['named_entities']
            if isinstance(entities, list):
                content_parts.append(f"NAMED ENTITIES:\n{', '.join(entities)}")
            else:
                content_parts.append(f"NAMED ENTITIES:\n{entities}")
        
        # Add original subject headings
        if analysis.get('subject_headings'):
            subjects = analysis['subject_headings']
            if isinstance(subjects, list):
                content_parts.append(f"ORIGINAL SUBJECT HEADINGS:\n{', '.join(subjects)}")
            else:
                content_parts.append(f"ORIGINAL SUBJECT HEADINGS:\n{subjects}")
        
        # Add content warning if present
        if analysis.get('content_warning') and analysis['content_warning'] != 'None':
            content_parts.append(f"CONTENT WARNING:\n{analysis['content_warning']}")
        
        content_description = "\n\n".join(content_parts)
        
        # Build available LCSH terms
        lcsh_terms = analysis.get('lcsh_headings', [])
        if not lcsh_terms:
            return None  # No LCSH terms available for selection
        
        lcsh_list = []
        for term in lcsh_terms:
            lcsh_list.append(f"- {term['label']} ({term['uri']})")
        
        lcsh_section = "AVAILABLE LCSH TERMS:\n" + "\n".join(lcsh_list)
        
        # Combine everything
        user_prompt = f"""Please analyze this archival content and select the most appropriate LCSH terms:

{content_description}

{lcsh_section}

Select only the LCSH terms that are most relevant and appropriate for this content. Provide reasoning for your selections."""
        
        return user_prompt

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        stop=tenacity.stop_after_attempt(5),
        retry=tenacity.retry_if_exception_type(Exception)
    )
    def select_lcsh_terms(self, entry_data: Dict[str, Any]) -> Tuple[Dict[str, Any], str, Any, float]:
        """Select LCSH terms for a single entry."""
        user_prompt = self.create_user_prompt(entry_data)
        
        if not user_prompt:
            # No LCSH terms available
            return {
                "selected_lcsh_terms": []
            }, "No LCSH terms available for selection", None, 0
        
        api_stats.total_requests += 1
        start_time = time.time()
        
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1500,
            temperature=0.1  # Low temperature for consistent, precise selections
        )
        
        processing_time = time.time() - start_time
        api_stats.processing_times.append(processing_time)
        
        api_stats.total_input_tokens += response.usage.prompt_tokens
        api_stats.total_output_tokens += response.usage.completion_tokens
        
        raw_response = response.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            parsed_response = self.parse_json_response(raw_response)
            return parsed_response, raw_response, response.usage, processing_time
        except Exception as e:
            logging.error(f"Error parsing LCSH selection response: {e}")
            # Return empty selection on parsing error
            return {
                "selected_lcsh_terms": []
            }, raw_response, response.usage, processing_time

    def parse_json_response(self, raw_response: str) -> Dict[str, Any]:
        """Parse JSON response from the API."""
        import re
        
        # Remove markdown formatting
        cleaned_response = re.sub(r'```json\s*|\s*```', '', raw_response)
        cleaned_response = re.sub(r'^[^{]*({.*})[^}]*$', r'\1', cleaned_response, flags=re.DOTALL)
        
        # Remove trailing commas
        cleaned_response = re.sub(r',(\s*[}\]])', r'\1', cleaned_response)
        
        try:
            parsed_json = json.loads(cleaned_response)
            
            # Validate structure
            if 'selected_lcsh_terms' not in parsed_json:
                parsed_json['selected_lcsh_terms'] = []
            
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

    def prepare_batch_requests(self, entries_with_lcsh: List[Tuple[int, Dict[str, Any]]]) -> Tuple[List[Dict], Dict[str, Dict]]:
        """Prepare batch requests for LCSH selection."""
        batch_requests = []
        custom_id_mapping = {}
        
        for i, (entry_index, entry_data) in enumerate(entries_with_lcsh):
            user_prompt = self.create_user_prompt(entry_data)
            
            if not user_prompt:
                continue  # Skip entries without LCSH terms
            
            request_data = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 1500,
                "temperature": 0.1
            }
            
            batch_requests.append(request_data)
            custom_id_mapping[f"lcsh_selection_{i}"] = {
                "entry_index": entry_index,
                "entry_data": entry_data
            }
        
        return batch_requests, custom_id_mapping

class SouthernArchitectLCSHSelector:
    """Main class for selecting optimal LCSH terms."""
    
    def __init__(self, folder_path: str, model_name: str = "gpt-4o-mini-2024-07-18"):
        self.folder_path = folder_path
        self.model_name = model_name
        self.workflow_type = None
        self.json_data = None
        self.excel_path = None
        self.lcsh_selector = LCSHSelector(model_name)
    
    def detect_workflow_type(self) -> bool:
        """Detect workflow type and check for LCSH enhancement."""
        # Check for expected files
        text_files = ['text_workflow.xlsx', 'text_workflow.json']
        image_files = ['image_workflow.xlsx', 'image_workflow.json']
        
        has_text_files = all(os.path.exists(os.path.join(self.folder_path, f)) for f in text_files)
        has_image_files = all(os.path.exists(os.path.join(self.folder_path, f)) for f in image_files)
        
        if has_text_files and not has_image_files:
            self.workflow_type = 'text'
            self.excel_path = os.path.join(self.folder_path, 'text_workflow.xlsx')
        elif has_image_files and not has_text_files:
            self.workflow_type = 'image'
            self.excel_path = os.path.join(self.folder_path, 'image_workflow.xlsx')
        else:
            logging.error("Could not determine workflow type or multiple workflow files found.")
            return False
        
        # Check if LCSH enhancement has been run (step 2)
        if not os.path.exists(os.path.join(self.folder_path, 'lcsh_mapping_report.txt')):
            logging.error("LCSH enhancement (step 2) must be run before step 3.")
            return False
        
        return True
    
    def load_json_data(self) -> bool:
        """Load JSON data and verify LCSH headings exist."""
        json_filename = f"{self.workflow_type}_workflow.json"
        json_path = os.path.join(self.folder_path, json_filename)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.json_data = json.load(f)
            
            # Check if LCSH headings exist in the data
            data_items = self.json_data[:-1] if self.json_data and 'api_stats' in self.json_data[-1] else self.json_data
            
            has_lcsh = False
            for item in data_items:
                if 'analysis' in item and 'lcsh_headings' in item['analysis']:
                    lcsh_headings = item['analysis']['lcsh_headings']
                    if lcsh_headings:  # Check if there are any LCSH headings
                        has_lcsh = True
                        break
            
            if not has_lcsh:
                logging.error("No LCSH headings found in JSON data. Please run step 2 first.")
                return False
            
            print(f"📊 Loaded JSON data from {json_filename}")
            return True
            
        except Exception as e:
            logging.error(f"Error loading JSON data: {e}")
            return False
    
    def find_entries_with_lcsh(self) -> List[Tuple[int, Dict[str, Any]]]:
        """Find entries that have LCSH headings available for selection."""
        entries_with_lcsh = []
        
        # Skip the last item if it's API stats
        data_items = self.json_data[:-1] if self.json_data and 'api_stats' in self.json_data[-1] else self.json_data
        
        for i, item in enumerate(data_items):
            if 'analysis' in item and 'lcsh_headings' in item['analysis']:
                lcsh_headings = item['analysis']['lcsh_headings']
                if lcsh_headings:  # Only include items with LCSH headings
                    entries_with_lcsh.append((i, item))
        
        return entries_with_lcsh
    
    def process_lcsh_selection(self, entries_with_lcsh: List[Tuple[int, Dict[str, Any]]], use_batch: bool = False) -> Dict[int, Dict[str, Any]]:
        """Process LCSH selection for entries."""
        selection_results = {}
        
        if use_batch:
            return self.process_batch_selection(entries_with_lcsh)
        else:
            return self.process_individual_selection(entries_with_lcsh)
    
    def process_individual_selection(self, entries_with_lcsh: List[Tuple[int, Dict[str, Any]]]) -> Dict[int, Dict[str, Any]]:
        """Process LCSH selection using individual API calls."""
        selection_results = {}
        
        # Create logs folder
        logs_folder_path = os.path.join(self.folder_path, "logs")
        if not os.path.exists(logs_folder_path):
            os.makedirs(logs_folder_path)
        
        total_entries = len(entries_with_lcsh)
        processed_entries = 0
        
        for i, (entry_index, entry_data) in enumerate(entries_with_lcsh):
            print(f"\n📋 Processing entry {i+1}/{total_entries}")
            print(f"   Entry index: {entry_index}")
            print(f"   Progress: {((i+1)/total_entries)*100:.1f}%")
            
            try:
                selection_result, raw_response, usage, processing_time = self.lcsh_selector.select_lcsh_terms(entry_data)
                
                # Log individual response
                log_individual_response(
                    logs_folder_path=logs_folder_path,
                    script_name="southern_architect_lcsh_selection",
                    row_number=entry_index + 2,  # +2 for header row
                    barcode=f"{entry_data.get('folder', 'unknown')}_page{entry_data.get('page_number', 'unknown')}",
                    response_text=raw_response,
                    model_name=self.model_name,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    processing_time=processing_time
                )
                
                selection_results[entry_index] = {
                    'selection_result': selection_result,
                    'raw_response': raw_response,
                    'processing_time': processing_time
                }
                
                selected_count = len(selection_result.get('selected_lcsh_terms', []))
                print(f"   ✅ Selected {selected_count} LCSH terms")
                processed_entries += 1
                
            except Exception as e:
                logging.error(f"Error processing entry {entry_index}: {e}")
                selection_results[entry_index] = {
                    'selection_result': {'selected_lcsh_terms': []},
                    'raw_response': f"Error: {str(e)}",
                    'processing_time': 0
                }
                print(f"   ❌ Processing failed: {str(e)}")
            
            # Add delay between requests
            time.sleep(0.5)
        
        print(f"\n📊 Individual processing completed: {processed_entries}/{total_entries} entries processed")
        return selection_results
    
    def process_batch_selection(self, entries_with_lcsh: List[Tuple[int, Dict[str, Any]]]) -> Dict[int, Dict[str, Any]]:
        """Process LCSH selection using batch API."""
        print(f"📦 Preparing {len(entries_with_lcsh)} requests for batch processing...")
        
        # Prepare batch requests
        batch_requests, custom_id_mapping = self.lcsh_selector.prepare_batch_requests(entries_with_lcsh)
        
        if not batch_requests:
            print("⚠️  No valid requests to process")
            return {}
        
        # Initialize batch processor
        processor = BatchProcessor()
        
        # Estimate costs
        cost_estimate = processor.estimate_batch_cost(batch_requests, self.model_name)
        print(f"💰 Estimated cost: ${cost_estimate['batch_cost']:.4f} (${cost_estimate['savings']:.4f} savings)")
        
        # Convert to batch format
        formatted_requests = processor.create_batch_requests(batch_requests, "lcsh_selection")
        
        # Submit batch
        batch_id = processor.submit_batch(
            formatted_requests,
            f"Southern Architect LCSH Selection - {len(batch_requests)} entries - {datetime.now().strftime('%Y-%m-%d')}"
        )
        
        # Wait for completion
        batch_results = processor.wait_for_completion(batch_id, max_wait_hours=24, check_interval_minutes=5)
        
        if not batch_results:
            print("❌ Batch processing failed")
            return {}
        
        # Process batch results
        processed_results = processor.process_batch_results(batch_results, custom_id_mapping)
        
        # Convert to selection results format
        selection_results = {}
        
        for custom_id, result_data in processed_results["results"].items():
            if custom_id.startswith("lcsh_selection_"):
                mapping_data = custom_id_mapping.get(custom_id, {})
                entry_index = mapping_data.get('entry_index')
                
                if entry_index is not None:
                    if result_data["success"]:
                        raw_response = result_data["content"]
                        try:
                            selection_result = self.lcsh_selector.parse_json_response(raw_response)
                        except Exception as e:
                            logging.error(f"Error parsing batch response for entry {entry_index}: {e}")
                            selection_result = {'selected_lcsh_terms': []}
                        
                        selection_results[entry_index] = {
                            'selection_result': selection_result,
                            'raw_response': raw_response,
                            'processing_time': 0  # Batch doesn't track individual timing
                        }
                    else:
                        selection_results[entry_index] = {
                            'selection_result': {'selected_lcsh_terms': []},
                            'raw_response': f"Error: {result_data['error']}",
                            'processing_time': 0
                        }
        
        print(f"📊 Batch processing completed: {len(selection_results)} entries processed")
        return selection_results
    
    def update_json_data(self, selection_results: Dict[int, Dict[str, Any]]) -> bool:
        """Update JSON data with selected LCSH terms."""
        try:
            # Skip the last item if it's API stats
            data_items = self.json_data[:-1] if self.json_data and 'api_stats' in self.json_data[-1] else self.json_data
            api_stats = self.json_data[-1] if self.json_data and 'api_stats' in self.json_data[-1] else None
            
            updated_items = []
            
            for i, item in enumerate(data_items):
                if i in selection_results:
                    # Add selected LCSH terms
                    selected_terms = selection_results[i]['selection_result'].get('selected_lcsh_terms', [])
                    item['analysis']['selected_lcsh_headings'] = selected_terms
                else:
                    # Add empty selected terms for entries without LCSH headings
                    item['analysis']['selected_lcsh_headings'] = []
                
                updated_items.append(item)
            
            # Add API stats back if it existed
            if api_stats:
                updated_items.append(api_stats)
            
            # Save updated JSON
            json_filename = f"{self.workflow_type}_workflow.json"
            json_path = os.path.join(self.folder_path, json_filename)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(updated_items, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Updated JSON file with selected LCSH terms")
            return True
            
        except Exception as e:
            logging.error(f"Error updating JSON data: {e}")
            return False
    
    def update_excel_file(self, selection_results: Dict[int, Dict[str, Any]]) -> bool:
        """Update Excel file with selected LCSH terms."""
        try:
            # Load the existing workbook
            wb = load_workbook(self.excel_path)
            analysis_sheet = wb['Analysis']
            
            # Find the LCSH Headings column (should be after Subject Headings)
            lcsh_col = None
            for col in range(1, analysis_sheet.max_column + 1):
                if analysis_sheet.cell(row=1, column=col).value == "LCSH Headings":
                    lcsh_col = col
                    break
            
            if lcsh_col is None:
                logging.error("LCSH Headings column not found. Please run step 2 first.")
                return False
            
            # Insert a new column for selected LCSH terms
            selected_lcsh_col = lcsh_col + 1
            analysis_sheet.insert_cols(selected_lcsh_col)
            
            # Add header
            header_cell = analysis_sheet.cell(row=1, column=selected_lcsh_col)
            header_cell.value = "Selected LCSH Headings"
            header_cell.alignment = Alignment(vertical='top', wrap_text=True)
            
            # Set column width
            col_letter = header_cell.column_letter
            analysis_sheet.column_dimensions[col_letter].width = 60
            
            # Update data rows
            updated_rows = 0
            for entry_index, result_data in selection_results.items():
                row_num = entry_index + 2  # +2 for header row
                
                selected_terms = result_data['selection_result'].get('selected_lcsh_terms', [])
                
                if selected_terms:
                    # Format selected terms with labels and URIs
                    formatted_terms = []
                    for term in selected_terms:
                        formatted_terms.append(f"{term['label']} ({term['uri']})")
                    
                    cell_value = "; ".join(formatted_terms)
                    updated_rows += 1
                else:
                    cell_value = ""
                
                # Set the cell value
                cell = analysis_sheet.cell(row=row_num, column=selected_lcsh_col)
                cell.value = cell_value
                cell.alignment = Alignment(vertical='top', wrap_text=True)
            
            # Save the updated workbook
            wb.save(self.excel_path)
            print(f"✅ Updated Excel file with selected LCSH terms in {updated_rows} rows")
            return True
            
        except Exception as e:
            logging.error(f"Error updating Excel file: {e}")
            return False
    
    def create_selection_report(self, selection_results: Dict[int, Dict[str, Any]]) -> bool:
        """Create a detailed report of LCSH selections."""
        try:
            report_path = os.path.join(self.folder_path, "lcsh_selection_report.txt")
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("SOUTHERN ARCHITECT LCSH SELECTION REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model: {self.model_name}\n")
                f.write(f"Workflow Type: {self.workflow_type.upper()}\n")
                f.write(f"Total Entries Processed: {len(selection_results)}\n\n")
                
                # Statistics
                total_selected = sum(len(result['selection_result'].get('selected_lcsh_terms', [])) 
                                   for result in selection_results.values())
                entries_with_selections = sum(1 for result in selection_results.values() 
                                            if result['selection_result'].get('selected_lcsh_terms'))
                
                f.write("STATISTICS:\n")
                f.write(f"- Total LCSH terms selected: {total_selected}\n")
                f.write(f"- Entries with selections: {entries_with_selections}/{len(selection_results)}\n")
                f.write(f"- Average selections per entry: {total_selected/len(selection_results):.1f}\n")
                f.write(f"- Selection rate: {(entries_with_selections/len(selection_results)*100):.1f}%\n\n")
                
                # Detailed selections
                f.write("DETAILED SELECTIONS:\n")
                f.write("-" * 30 + "\n\n")
                
                for entry_index, result_data in sorted(selection_results.items()):
                    selected_terms = result_data['selection_result'].get('selected_lcsh_terms', [])
                    
                    f.write(f"Entry {entry_index + 1}:\n")
                    
                    if selected_terms:
                        f.write("SELECTED TERMS:\n")
                        for term in selected_terms:
                            f.write(f"  - {term['label']} ({term['uri']})\n")
                    else:
                        f.write("SELECTED TERMS: None\n")
                    
                    f.write("\n")
            
            print(f"📋 LCSH selection report saved to: {report_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error creating selection report: {e}")
            return False
    
    def run(self, use_batch: bool = False) -> bool:
        """Main execution method."""
        print(f"\n🎯 SOUTHERN ARCHITECT STEP 3 - LCSH SELECTION")
        print(f"📁 Processing folder: {self.folder_path}")
        print(f"🤖 Model: {self.model_name}")
        print("-" * 50)
        
        # Detect workflow type
        if not self.detect_workflow_type():
            return False
        
        print(f"🔍 Detected workflow type: {self.workflow_type.upper()}")
        
        # Load JSON data
        if not self.load_json_data():
            return False
        
        # Find entries with LCSH headings
        entries_with_lcsh = self.find_entries_with_lcsh()
        if not entries_with_lcsh:
            print("⚠️  No entries with LCSH headings found")
            return False
        
        print(f"📚 Found {len(entries_with_lcsh)} entries with LCSH headings")
        
        # Show model pricing info
        model_info = get_model_info(self.model_name)
        if model_info:
            print(f"💰 Pricing: ${model_info['input_per_1k']:.5f}/1K input, ${model_info['output_per_1k']:.5f}/1K output")
        
        # Process LCSH selection
        selection_results = self.process_lcsh_selection(entries_with_lcsh, use_batch)
        
        if not selection_results:
            print("❌ LCSH selection failed")
            return False
        
        # Update JSON data
        if not self.update_json_data(selection_results):
            return False
        
        # Update Excel file
        if not self.update_excel_file(selection_results):
            return False
        
        # Create selection report
        self.create_selection_report(selection_results)
        
        # Calculate and log final metrics
        script_start_time = time.time()
        total_processing_time = sum(result.get('processing_time', 0) for result in selection_results.values())
        
        # Calculate cost
        estimated_cost = calculate_cost(
            model_name=self.model_name,
            prompt_tokens=api_stats.total_input_tokens,
            completion_tokens=api_stats.total_output_tokens,
            is_batch=use_batch
        )
        
        # Create logs folder and token usage log
        logs_folder_path = os.path.join(self.folder_path, "logs")
        if not os.path.exists(logs_folder_path):
            os.makedirs(logs_folder_path)
        
        create_token_usage_log(
            logs_folder_path=logs_folder_path,
            script_name="southern_architect_lcsh_selection",
            model_name=self.model_name,
            total_items=len(selection_results),
            items_with_issues=0,  # We handle errors gracefully
            total_time=total_processing_time,
            total_prompt_tokens=api_stats.total_input_tokens,
            total_completion_tokens=api_stats.total_output_tokens,
            additional_metrics={
                "Processing mode": "BATCH" if use_batch else "INDIVIDUAL",
                "Actual cost": f"${estimated_cost:.4f}",
                "Average tokens per entry": f"{(api_stats.total_input_tokens + api_stats.total_output_tokens)/len(selection_results):.0f}" if selection_results else "0"
            }
        )
        
        # Final summary
        total_selected = sum(len(result['selection_result'].get('selected_lcsh_terms', [])) 
                           for result in selection_results.values())
        entries_with_selections = sum(1 for result in selection_results.values() 
                                    if result['selection_result'].get('selected_lcsh_terms'))
        
        print(f"\n🎉 LCSH SELECTION COMPLETED!")
        print(f"✅ Entries processed: {len(selection_results)}")
        print(f"📚 Total LCSH terms selected: {total_selected}")
        print(f"📊 Entries with selections: {entries_with_selections}/{len(selection_results)}")
        print(f"📈 Selection rate: {(entries_with_selections/len(selection_results)*100):.1f}%")
        print(f"🎯 Total tokens: {api_stats.total_input_tokens + api_stats.total_output_tokens:,}")
        print(f"💰 Estimated cost: ${estimated_cost:.4f}")
        print(f"📄 Enhanced Excel file: {self.excel_path}")
        print(f"📄 Enhanced JSON file: {os.path.join(self.folder_path, f'{self.workflow_type}_workflow.json')}")
        print(f"📋 Selection report: {os.path.join(self.folder_path, 'lcsh_selection_report.txt')}")
        
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
    parser = argparse.ArgumentParser(description='Select optimal LCSH terms using AI')
    parser.add_argument('--folder', help='Specific folder path to process')
    parser.add_argument('--newest', action='store_true', help='Process the newest folder in the output directory')
    parser.add_argument('--model', default="gpt-4o-mini-2024-07-18", help='Model name to use for LCSH selection')
    parser.add_argument('--batch', action='store_true', help='Use batch processing (for large datasets)')
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
    
    # Create and run the selector
    selector = SouthernArchitectLCSHSelector(folder_path, args.model)
    success = selector.run(use_batch=args.batch)
    
    if not success:
        print("❌ LCSH selection failed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())