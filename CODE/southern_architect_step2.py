# This code enhances Southern Architect metadata with verified LCSH headings.
import os
import json
import logging
import time
import requests
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from openpyxl import load_workbook
from openpyxl.styles import Alignment
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LOCAuthorizedTermFinder:
    """Enhanced LOC term finder with rate limiting and error handling."""
    
    def __init__(self):
        self.base_url = "https://id.loc.gov/authorities/subjects/suggest2"
        self.headers = {
            'User-Agent': 'Python-LOC-Term-Finder/1.0 (Educational/Research Use)'
        }
        self.lcsh_authorized_headings = "http://id.loc.gov/authorities/subjects/collection_LCSHAuthorizedHeadings"
        self.max_results = 3  # Keep it concise for spreadsheet
        self.request_delay = 0.5  # Respectful rate limiting
        self.cache = {}  # Cache results to avoid duplicate requests
        
    def search(self, query: str, search_type: str) -> List[Dict[str, str]]:
        """Search LOC for authorized terms."""
        cache_key = f"{query}_{search_type}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        params = {
            'q': query,
            'searchtype': search_type,
            'count': self.max_results,
            'memberOf': self.lcsh_authorized_headings
        }
        
        try:
            resp = requests.get(self.base_url, params=params, headers=self.headers, timeout=10)
            resp.raise_for_status()
            hits = resp.json().get('hits', [])
            
            results = [
                {'label': h['aLabel'], 'uri': h['uri']}
                for h in hits
                if h.get('aLabel') and h.get('uri')
            ]
            
            self.cache[cache_key] = results
            time.sleep(self.request_delay)
            return results
            
        except requests.exceptions.RequestException as e:
            logging.warning(f"Error searching for '{query}': {e}")
            return []
    
    def find_terms(self, topics: List[str]) -> Dict[str, List[Dict[str, str]]]:
        """Find LCSH terms for multiple topics."""
        if not topics:
            return {}
            
        results = {}
        
        for topic in topics:
            if not topic or topic.strip() == "":
                continue
                
            topic = topic.strip()
            
            # Skip if already processed
            if topic in results:
                continue
                
            # Try keyword search first
            keyword_results = self.search(topic, "keyword")
            
            # If we need more results, try left-anchored search
            if len(keyword_results) < self.max_results:
                leftanchored_results = self.search(topic, "leftanchored")
                
                # Merge results, avoiding duplicates
                existing_uris = {r['uri'] for r in keyword_results}
                for result in leftanchored_results:
                    if len(keyword_results) >= self.max_results:
                        break
                    if result['uri'] not in existing_uris:
                        keyword_results.append(result)
            
            results[topic] = keyword_results[:self.max_results]
            
            if keyword_results:
                print(f"   📚 Found {len(keyword_results)} LCSH terms for '{topic}'")
            else:
                print(f"   ⚠️  No LCSH terms found for '{topic}'")
        
        return results
    
    def format_results_for_excel(self, results: Dict[str, List[Dict[str, str]]]) -> str:
        """Format results for spreadsheet display with labels and URIs."""
        if not results:
            return ""
            
        formatted_terms = []
        for topic, terms in results.items():
            if terms:
                # Format as "Label (URI)"
                term_strings = [f"{term['label']} ({term['uri']})" for term in terms]
                formatted_terms.extend(term_strings)
        
        # Remove duplicates while preserving order
        unique_terms = []
        seen = set()
        for term in formatted_terms:
            if term not in seen:
                unique_terms.append(term)
                seen.add(term)
        
        return "; ".join(unique_terms)
    
    def format_results_for_json(self, results: Dict[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
        """Format results for JSON storage with full label/URI structure."""
        if not results:
            return []
            
        all_terms = []
        seen_uris = set()
        
        for topic, terms in results.items():
            for term in terms:
                if term['uri'] not in seen_uris:
                    all_terms.append({
                        'label': term['label'],
                        'uri': term['uri']
                    })
                    seen_uris.add(term['uri'])
        
        return all_terms

class SouthernArchitectEnhancer:
    """Main class for enhancing Southern Architect results with LCSH headings."""
    
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.loc_finder = LOCAuthorizedTermFinder()
        self.workflow_type = None  # 'text' or 'image'
        self.json_data = None
        self.excel_path = None
    
    def format_results_for_excel(self, results: Dict[str, List[Dict[str, str]]]) -> str:
        """Format results for spreadsheet display with labels and URIs."""
        if not results:
            return ""
            
        formatted_terms = []
        for topic, terms in results.items():
            if terms:
                # Format as "Label (URI)"
                term_strings = [f"{term['label']} ({term['uri']})" for term in terms]
                formatted_terms.extend(term_strings)
        
        # Remove duplicates while preserving order
        unique_terms = []
        seen = set()
        for term in formatted_terms:
            if term not in seen:
                unique_terms.append(term)
                seen.add(term)
        
        return "; ".join(unique_terms)
    
    def format_results_for_json(self, results: Dict[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
        """Format results for JSON storage with full label/URI structure."""
        if not results:
            return []
            
        all_terms = []
        seen_uris = set()
        
        for topic, terms in results.items():
            for term in terms:
                if term['uri'] not in seen_uris:
                    all_terms.append({
                        'label': term['label'],
                        'uri': term['uri']
                    })
                    seen_uris.add(term['uri'])
        
        return all_terms
        
    def detect_workflow_type(self) -> bool:
        """Detect whether this is a text or image workflow folder."""
        # Check for expected files
        text_files = ['text_workflow.xlsx', 'text_workflow.json']
        image_files = ['image_workflow.xlsx', 'image_workflow.json']
        
        has_text_files = all(os.path.exists(os.path.join(self.folder_path, f)) for f in text_files)
        has_image_files = all(os.path.exists(os.path.join(self.folder_path, f)) for f in image_files)
        
        if has_text_files and not has_image_files:
            self.workflow_type = 'text'
            self.excel_path = os.path.join(self.folder_path, 'text_workflow.xlsx')
            return True
        elif has_image_files and not has_text_files:
            self.workflow_type = 'image'
            self.excel_path = os.path.join(self.folder_path, 'image_workflow.xlsx')
            return True
        elif has_text_files and has_image_files:
            logging.error("Both text and image workflow files found. Please specify workflow type.")
            return False
        else:
            logging.error("No recognized workflow files found in the folder.")
            return False
    
    def load_json_data(self) -> bool:
        """Load the JSON data from the appropriate workflow file."""
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
    
    def extract_subject_headings(self) -> List[str]:
        """Extract all unique subject headings from the JSON data."""
        all_subjects = set()
        
        # Skip the last item if it's API stats
        data_items = self.json_data[:-1] if self.json_data and 'api_stats' in self.json_data[-1] else self.json_data
        
        for item in data_items:
            if 'analysis' in item and 'subject_headings' in item['analysis']:
                subjects = item['analysis']['subject_headings']
                if isinstance(subjects, list):
                    for subject in subjects:
                        if subject and subject.strip():
                            all_subjects.add(subject.strip())
                elif isinstance(subjects, str) and subjects.strip():
                    # Handle comma-separated string format
                    for subject in subjects.split(','):
                        if subject.strip():
                            all_subjects.add(subject.strip())
        
        return sorted(list(all_subjects))
    
    def process_lcsh_lookup(self, subjects: List[str]) -> Tuple[Dict[str, str], Dict[str, List[Dict[str, str]]]]:
        """Process LCSH lookup for all subjects and return mapping for both Excel and JSON."""
        if not subjects:
            return {}, {}
        
        print(f"\n🔍 Processing LCSH lookup for {len(subjects)} unique subjects...")
        
        # Group similar subjects to reduce API calls
        subject_groups = self.group_similar_subjects(subjects)
        
        # Process each group
        subject_to_lcsh_excel = {}
        subject_to_lcsh_json = {}
        total_groups = len(subject_groups)
        
        for i, (representative, group_subjects) in enumerate(subject_groups.items(), 1):
            print(f"\n📋 Processing group {i}/{total_groups}: '{representative}'")
            
            # Find LCSH terms for the representative subject
            lcsh_results = self.loc_finder.find_terms([representative])
            formatted_lcsh_excel = self.format_results_for_excel(lcsh_results)
            formatted_lcsh_json = self.format_results_for_json(lcsh_results)
            
            # Apply the same LCSH terms to all subjects in the group
            for subject in group_subjects:
                subject_to_lcsh_excel[subject] = formatted_lcsh_excel
                subject_to_lcsh_json[subject] = formatted_lcsh_json
        
        return subject_to_lcsh_excel, subject_to_lcsh_json
    
    def group_similar_subjects(self, subjects: List[str]) -> Dict[str, List[str]]:
        """Group similar subjects to reduce API calls."""
        groups = defaultdict(list)
        
        for subject in subjects:
            # Use the subject itself as the key (representative)
            # In a more sophisticated version, you could implement fuzzy matching
            groups[subject].append(subject)
        
        return dict(groups)
    
    def enhance_excel_file(self, subject_to_lcsh: Dict[str, str]) -> bool:
        """Add LCSH headings column to the Excel file."""
        try:
            # Load the existing workbook
            wb = load_workbook(self.excel_path)
            analysis_sheet = wb['Analysis']
            
            # Determine the column structure based on workflow type
            if self.workflow_type == 'text':
                # Text workflow: ['Folder', 'Page Number', 'Page Title', 'Cleaned OCR Text', 
                #                'TOC Entry', 'Named Entities', 'Subject Headings', 'Content Warning']
                subject_col = 7  # Subject Headings column
                insert_col = 8  # Insert LCSH after Subject Headings
            else:  # image workflow
                # Image workflow: ['Folder', 'Page Number', 'Image Path', 'Text Transcription', 
                #                 'Visual Description', 'TOC Entry', 'Named Entities', 'Subject Headings', 'Content Warning']
                subject_col = 8  # Subject Headings column
                insert_col = 9  # Insert LCSH after Subject Headings
            
            # Insert new column
            analysis_sheet.insert_cols(insert_col)
            
            # Add header
            header_cell = analysis_sheet.cell(row=1, column=insert_col)
            header_cell.value = "LCSH Headings"
            header_cell.alignment = Alignment(vertical='top', wrap_text=True)
            
            # Set column width
            col_letter = header_cell.column_letter
            analysis_sheet.column_dimensions[col_letter].width = 60  # Wider to accommodate URIs
            
            # Process each data row
            processed_rows = 0
            for row_num in range(2, analysis_sheet.max_row + 1):
                # Get the subject headings from the current row
                subject_cell = analysis_sheet.cell(row=row_num, column=subject_col)
                subject_headings = subject_cell.value or ""
                
                # Process subject headings
                if subject_headings and subject_headings.strip():
                    # Split by comma and lookup each subject
                    subjects = [s.strip() for s in subject_headings.split(',') if s.strip()]
                    lcsh_terms = []
                    
                    for subject in subjects:
                        if subject in subject_to_lcsh and subject_to_lcsh[subject]:
                            # Split the LCSH result and add individual terms
                            terms = [t.strip() for t in subject_to_lcsh[subject].split(';') if t.strip()]
                            lcsh_terms.extend(terms)
                    
                    # Remove duplicates while preserving order
                    unique_lcsh_terms = []
                    seen = set()
                    for term in lcsh_terms:
                        if term not in seen:
                            unique_lcsh_terms.append(term)
                            seen.add(term)
                    
                    # Set the LCSH cell value
                    lcsh_cell = analysis_sheet.cell(row=row_num, column=insert_col)
                    lcsh_cell.value = "; ".join(unique_lcsh_terms) if unique_lcsh_terms else ""
                    lcsh_cell.alignment = Alignment(vertical='top', wrap_text=True)
                    
                    if unique_lcsh_terms:
                        processed_rows += 1
                else:
                    # Empty subject headings
                    lcsh_cell = analysis_sheet.cell(row=row_num, column=insert_col)
                    lcsh_cell.value = ""
                    lcsh_cell.alignment = Alignment(vertical='top', wrap_text=True)
            
            # Save the enhanced workbook
            wb.save(self.excel_path)
            print(f"✅ Enhanced Excel file saved with LCSH headings in {processed_rows} rows")
            return True
            
        except Exception as e:
            logging.error(f"Error enhancing Excel file: {e}")
            return False
    
    def enhance_json_file(self, subject_to_lcsh_json: Dict[str, List[Dict[str, str]]]) -> bool:
        """Add LCSH headings data to the JSON file."""
        try:
            # Skip the last item if it's API stats
            data_items = self.json_data[:-1] if self.json_data and 'api_stats' in self.json_data[-1] else self.json_data
            api_stats = self.json_data[-1] if self.json_data and 'api_stats' in self.json_data[-1] else None
            
            enhanced_items = []
            processed_items = 0
            
            for item in data_items:
                if 'analysis' in item:
                    # Get subject headings for this item
                    subject_headings = item['analysis'].get('subject_headings', [])
                    
                    # Normalize subject headings to list format
                    if isinstance(subject_headings, str):
                        subjects = [s.strip() for s in subject_headings.split(',') if s.strip()]
                    else:
                        subjects = subject_headings if isinstance(subject_headings, list) else []
                    
                    # Collect LCSH terms for these subjects
                    lcsh_terms = []
                    seen_uris = set()
                    
                    for subject in subjects:
                        if subject in subject_to_lcsh_json and subject_to_lcsh_json[subject]:
                            for term in subject_to_lcsh_json[subject]:
                                if term['uri'] not in seen_uris:
                                    lcsh_terms.append(term)
                                    seen_uris.add(term['uri'])
                    
                    # Add LCSH headings to the analysis
                    item['analysis']['lcsh_headings'] = lcsh_terms
                    
                    if lcsh_terms:
                        processed_items += 1
                
                enhanced_items.append(item)
            
            # Add API stats back if it existed
            if api_stats:
                enhanced_items.append(api_stats)
            
            # Save the enhanced JSON
            json_filename = f"{self.workflow_type}_workflow.json"
            json_path = os.path.join(self.folder_path, json_filename)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_items, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Enhanced JSON file saved with LCSH headings in {processed_items} items")
            return True
            
        except Exception as e:
            logging.error(f"Error enhancing JSON file: {e}")
            return False
    
    def create_lcsh_report(self, subject_to_lcsh: Dict[str, str]) -> bool:
        """Create a detailed LCSH mapping report."""
        try:
            report_path = os.path.join(self.folder_path, "lcsh_mapping_report.txt")
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("SOUTHERN ARCHITECT LCSH MAPPING REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Workflow Type: {self.workflow_type.upper()}\n")
                f.write(f"Total Subjects Processed: {len(subject_to_lcsh)}\n\n")
                
                # Statistics
                subjects_with_lcsh = sum(1 for lcsh in subject_to_lcsh.values() if lcsh)
                subjects_without_lcsh = len(subject_to_lcsh) - subjects_with_lcsh
                
                f.write("STATISTICS:\n")
                f.write(f"- Subjects with LCSH terms: {subjects_with_lcsh}\n")
                f.write(f"- Subjects without LCSH terms: {subjects_without_lcsh}\n")
                f.write(f"- Success rate: {(subjects_with_lcsh/len(subject_to_lcsh)*100):.1f}%\n\n")
                
                # Detailed mappings
                f.write("DETAILED MAPPINGS:\n")
                f.write("-" * 30 + "\n\n")
                
                for subject, lcsh_terms in sorted(subject_to_lcsh.items()):
                    f.write(f"Subject: {subject}\n")
                    if lcsh_terms:
                        f.write(f"LCSH: {lcsh_terms}\n")
                    else:
                        f.write("LCSH: No terms found\n")
                    f.write("\n")
            
            print(f"📋 LCSH mapping report saved to: {report_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error creating LCSH report: {e}")
            return False
        """Create a detailed LCSH mapping report."""
        try:
            report_path = os.path.join(self.folder_path, "lcsh_mapping_report.txt")
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("SOUTHERN ARCHITECT LCSH MAPPING REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Workflow Type: {self.workflow_type.upper()}\n")
                f.write(f"Total Subjects Processed: {len(subject_to_lcsh)}\n\n")
                
                # Statistics
                subjects_with_lcsh = sum(1 for lcsh in subject_to_lcsh.values() if lcsh)
                subjects_without_lcsh = len(subject_to_lcsh) - subjects_with_lcsh
                
                f.write("STATISTICS:\n")
                f.write(f"- Subjects with LCSH terms: {subjects_with_lcsh}\n")
                f.write(f"- Subjects without LCSH terms: {subjects_without_lcsh}\n")
                f.write(f"- Success rate: {(subjects_with_lcsh/len(subject_to_lcsh)*100):.1f}%\n\n")
                
                # Detailed mappings
                f.write("DETAILED MAPPINGS:\n")
                f.write("-" * 30 + "\n\n")
                
                for subject, lcsh_terms in sorted(subject_to_lcsh.items()):
                    f.write(f"Subject: {subject}\n")
                    if lcsh_terms:
                        f.write(f"LCSH: {lcsh_terms}\n")
                    else:
                        f.write("LCSH: No terms found\n")
                    f.write("\n")
            
            print(f"📋 LCSH mapping report saved to: {report_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error creating LCSH report: {e}")
            return False
    
    def run(self) -> bool:
        """Main execution method."""
        print(f"\n🎯 SOUTHERN ARCHITECT STEP 2 - LCSH ENHANCEMENT")
        print(f"📁 Processing folder: {self.folder_path}")
        print("-" * 50)
        
        # Detect workflow type
        if not self.detect_workflow_type():
            return False
        
        print(f"🔍 Detected workflow type: {self.workflow_type.upper()}")
        
        # Load JSON data
        if not self.load_json_data():
            return False
        
        # Extract subject headings
        subjects = self.extract_subject_headings()
        if not subjects:
            print("⚠️  No subject headings found in the data")
            return False
        
        print(f"📚 Found {len(subjects)} unique subject headings")
        
        # Process LCSH lookup
        subject_to_lcsh_excel, subject_to_lcsh_json = self.process_lcsh_lookup(subjects)
        
        # Enhance Excel file
        if not self.enhance_excel_file(subject_to_lcsh_excel):
            return False
        
        # Enhance JSON file
        if not self.enhance_json_file(subject_to_lcsh_json):
            return False
        
        # Create LCSH report
        self.create_lcsh_report(subject_to_lcsh_excel)
        
        # Final summary
        subjects_with_lcsh = sum(1 for lcsh in subject_to_lcsh_excel.values() if lcsh)
        print(f"\n🎉 LCSH ENHANCEMENT COMPLETED!")
        print(f"✅ Subjects with LCSH terms: {subjects_with_lcsh}/{len(subjects)}")
        print(f"📊 Success rate: {(subjects_with_lcsh/len(subjects)*100):.1f}%")
        print(f"📄 Enhanced Excel file: {self.excel_path}")
        print(f"📄 Enhanced JSON file: {os.path.join(self.folder_path, f'{self.workflow_type}_workflow.json')}")
        print(f"📋 Mapping report: {os.path.join(self.folder_path, 'lcsh_mapping_report.txt')}")
        
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
    parser = argparse.ArgumentParser(description='Enhance Southern Architect results with LCSH headings')
    parser.add_argument('--folder', help='Specific folder path to process')
    parser.add_argument('--newest', action='store_true', help='Process the newest folder in the output directory (default: True if no folder specified)')
    args = parser.parse_args()
    
    # Default base directory for Southern Architect output folders
    base_output_dir = "/Users/hannahmoutran/Desktop/southern_architect/CODE/output_folders"
    
    if args.folder:
        if not os.path.exists(args.folder):
            print(f"❌ Folder not found: {args.folder}")
            return
        folder_path = args.folder
    else:
        # Default to newest folder if no specific folder provided
        folder_path = find_newest_folder(base_output_dir)
        if not folder_path:
            print(f"❌ No folders found in: {base_output_dir}")
            return
        print(f"🔄 Auto-selected newest folder: {os.path.basename(folder_path)}")
    
    # Create and run the enhancer
    enhancer = SouthernArchitectEnhancer(folder_path)
    success = enhancer.run()
    
    if not success:
        print("❌ LCSH enhancement failed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())