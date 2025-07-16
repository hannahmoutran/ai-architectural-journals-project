import os
import json
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PageMetadataGenerator:
    """Class to generate individual page-level metadata files."""
    
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.workflow_type = None
        self.json_data = None
        self.output_folder = None
    
    def detect_workflow_type(self) -> bool:
        """Detect workflow type and check for required files."""
        # Check for expected files
        text_files = ['text_workflow.xlsx', 'text_workflow.json']
        image_files = ['image_workflow.xlsx', 'image_workflow.json']
        
        has_text_files = all(os.path.exists(os.path.join(self.folder_path, f)) for f in text_files)
        has_image_files = all(os.path.exists(os.path.join(self.folder_path, f)) for f in image_files)
        
        if has_text_files and not has_image_files:
            self.workflow_type = 'text'
            return True
        elif has_image_files and not has_text_files:
            self.workflow_type = 'image'
            return True
        else:
            logging.error("Could not determine workflow type or multiple workflow files found.")
            return False
    
    def load_json_data(self) -> bool:
        """Load JSON data from the appropriate workflow file."""
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
    
    def create_output_folder(self) -> bool:
        """Create the page_level_metadata output folder."""
        self.output_folder = os.path.join(self.folder_path, "page_level_metadata")
        
        try:
            os.makedirs(self.output_folder, exist_ok=True)
            print(f"📁 Created output folder: {self.output_folder}")
            return True
        except Exception as e:
            logging.error(f"Error creating output folder: {e}")
            return False
    
    def format_list_items(self, items: List[str], label: str) -> str:
        """Format a list of items with proper formatting."""
        if not items:
            return f"{label}: None\n"
        
        if len(items) == 1:
            return f"{label}: {items[0]}\n"
        else:
            formatted_items = '\n'.join([f"  - {item}" for item in items])
            return f"{label}:\n{formatted_items}\n"
    
    def format_lcsh_headings(self, lcsh_headings: List[Dict[str, str]]) -> str:
        """Format LCSH headings with labels and URIs."""
        if not lcsh_headings:
            return "Selected LCSH Headings: None\n"
        
        if len(lcsh_headings) == 1:
            heading = lcsh_headings[0]
            return f"Selected LCSH Headings: {heading['label']} ({heading['uri']})\n"
        else:
            formatted_headings = []
            for heading in lcsh_headings:
                formatted_headings.append(f"  - {heading['label']} ({heading['uri']})")
            return f"Selected LCSH Headings:\n" + '\n'.join(formatted_headings) + "\n"
    
    def generate_page_metadata(self, entry: Dict[str, Any]) -> str:
        """Generate formatted metadata text for a single page."""
        analysis = entry.get('analysis', {})
        
        # Basic page information
        metadata_lines = []
        metadata_lines.append("=" * 60)
        metadata_lines.append("SOUTHERN ARCHITECT - PAGE METADATA")
        metadata_lines.append("=" * 60)
        metadata_lines.append("")
        
        # Page identification
        metadata_lines.append("PAGE IDENTIFICATION:")
        metadata_lines.append(f"Folder: {entry.get('folder', 'Unknown')}")
        metadata_lines.append(f"Page Number: {entry.get('page_number', 'Unknown')}")
        
        # Original file path
        if self.workflow_type == 'text':
            original_path = entry.get('file_path', 'Unknown')
        else:  # image workflow
            original_path = entry.get('image_path', 'Unknown')
        metadata_lines.append(f"Original File: {original_path}")
        metadata_lines.append("")
        
        # Content sections
        metadata_lines.append("CONTENT:")
        metadata_lines.append("-" * 30)
        
        # Text content (different field names for text vs image workflows)
        if self.workflow_type == 'text':
            text_content = analysis.get('cleaned_text', '').strip()
            if text_content:
                metadata_lines.append("Cleaned OCR Text:")
                metadata_lines.append(text_content)
            else:
                metadata_lines.append("Cleaned OCR Text: [No text content]")
        else:  # image workflow
            # Text transcription
            text_transcription = analysis.get('text_transcription', '').strip()
            if text_transcription:
                metadata_lines.append("Text Transcription:")
                metadata_lines.append(text_transcription)
            else:
                metadata_lines.append("Text Transcription: [No text found in image]")
            
            metadata_lines.append("")
            
            # Visual description
            visual_description = analysis.get('visual_description', '').strip()
            if visual_description:
                metadata_lines.append("Visual Description:")
                metadata_lines.append(visual_description)
            else:
                metadata_lines.append("Visual Description: [No visual description available]")
        
        metadata_lines.append("")
        
        # Table of Contents entry
        toc_entry = analysis.get('toc_entry', '').strip()
        if toc_entry:
            metadata_lines.append("Table of Contents Entry:")
            metadata_lines.append(toc_entry)
        else:
            metadata_lines.append("Table of Contents Entry: [No TOC entry]")
        
        metadata_lines.append("")
        
        # Metadata sections
        metadata_lines.append("METADATA:")
        metadata_lines.append("-" * 30)
        
        # Named entities
        named_entities = analysis.get('named_entities', [])
        if isinstance(named_entities, str):
            # Handle case where it might be stored as comma-separated string
            named_entities = [e.strip() for e in named_entities.split(',') if e.strip()]
        metadata_lines.append(self.format_list_items(named_entities, "Named Entities").rstrip())
        metadata_lines.append("")
        
        # Subject headings (renamed to Topics)
        subject_headings = analysis.get('subject_headings', [])
        if isinstance(subject_headings, str):
            # Handle case where it might be stored as comma-separated string
            subject_headings = [s.strip() for s in subject_headings.split(',') if s.strip()]
        metadata_lines.append(self.format_list_items(subject_headings, "Topics").rstrip())
        metadata_lines.append("")
        
        # Content warning (only if not 'None')
        content_warning = analysis.get('content_warning', '').strip()
        if content_warning and content_warning.lower() != 'none':
            metadata_lines.append(f"Content Warning: {content_warning}")
            metadata_lines.append("")
        
        # Selected LCSH headings (if available)
        selected_lcsh = analysis.get('selected_lcsh_headings', [])
        if selected_lcsh:
            metadata_lines.append(self.format_lcsh_headings(selected_lcsh).rstrip())
            metadata_lines.append("")
        
        # Generation timestamp
        metadata_lines.append("PROCESSING INFORMATION:")
        metadata_lines.append("-" * 30)
        metadata_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        metadata_lines.append(f"Workflow Type: {self.workflow_type.upper()}")
        metadata_lines.append("")
        metadata_lines.append("=" * 60)
        
        return '\n'.join(metadata_lines)
    
    def generate_filename(self, entry: Dict[str, Any]) -> str:
        """Generate a clean filename for the metadata file."""
        folder = entry.get('folder', 'unknown')
        page_number = entry.get('page_number', 0)
        
        # Clean folder name for filename
        clean_folder = "".join(c for c in folder if c.isalnum() or c in ('-', '_')).strip()
        if not clean_folder:
            clean_folder = "unknown"
        
        return f"{clean_folder}_page{page_number:03d}_metadata.txt"
    
    def generate_all_metadata_files(self) -> bool:
        """Generate metadata files for all pages."""
        try:
            # Skip the last item if it's API stats
            data_items = self.json_data[:-1] if self.json_data and 'api_stats' in self.json_data[-1] else self.json_data
            
            if not data_items:
                print("⚠️  No data items found to process")
                return False
            
            generated_files = 0
            
            for i, entry in enumerate(data_items):
                try:
                    # Generate metadata content
                    metadata_content = self.generate_page_metadata(entry)
                    
                    # Generate filename
                    filename = self.generate_filename(entry)
                    file_path = os.path.join(self.output_folder, filename)
                    
                    # Write metadata file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(metadata_content)
                    
                    generated_files += 1
                    
                    if (i + 1) % 10 == 0:
                        print(f"   📝 Generated {i + 1}/{len(data_items)} metadata files...")
                
                except Exception as e:
                    logging.error(f"Error generating metadata for entry {i}: {e}")
                    continue
            
            print(f"✅ Successfully generated {generated_files} metadata files")
            return True
            
        except Exception as e:
            logging.error(f"Error generating metadata files: {e}")
            return False
    
    def create_index_file(self) -> bool:
        """Create an index file listing all generated metadata files."""
        try:
            index_path = os.path.join(self.output_folder, "00_INDEX.txt")
            
            # Get list of metadata files
            metadata_files = [f for f in os.listdir(self.output_folder) 
                            if f.endswith('_metadata.txt')]
            metadata_files.sort()
            
            with open(index_path, 'w', encoding='utf-8') as f:
                f.write("SOUTHERN ARCHITECT - PAGE METADATA INDEX\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Workflow Type: {self.workflow_type.upper()}\n")
                f.write(f"Total Files: {len(metadata_files)}\n\n")
                
                f.write("METADATA FILES:\n")
                f.write("-" * 20 + "\n")
                
                for filename in metadata_files:
                    # Extract page info from filename
                    parts = filename.replace('_metadata.txt', '').split('_page')
                    if len(parts) == 2:
                        folder_name = parts[0]
                        page_num = parts[1].lstrip('0') or '0'
                        f.write(f"{filename:<40} (Folder: {folder_name}, Page: {page_num})\n")
                    else:
                        f.write(f"{filename}\n")
                
                f.write(f"\n" + "=" * 50 + "\n")
            
            print(f"📋 Created index file: {index_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error creating index file: {e}")
            return False
    
    def create_issue_toc(self) -> bool:
        """Create an issue-level table of contents with TOC entries and verified subject headings."""
        try:
            # Skip API stats for processing
            data_items = self.json_data[:-1] if self.json_data and 'api_stats' in self.json_data[-1] else self.json_data
            
            if not data_items:
                print("⚠️  No data items found for table of contents")
                return False
            
            # Get folder name from first entry to create filename
            first_entry = data_items[0]
            folder_name = first_entry.get('folder', 'Unknown')
            
            # Create filename and path
            toc_filename = f"{folder_name}_Table_of_Contents.txt"
            toc_path = os.path.join(self.folder_path, toc_filename)
            
            with open(toc_path, 'w', encoding='utf-8') as f:
                f.write(f"{folder_name} Table of Contents\n")
                f.write("=" * (len(folder_name) + 18) + "\n\n")
                
                for entry in data_items:
                    analysis = entry.get('analysis', {})
                    page_number = entry.get('page_number', 'Unknown')
                    
                    # TOC entry
                    toc_entry = analysis.get('toc_entry', '').strip()
                    if not toc_entry or toc_entry.lower() == '[no toc entry]':
                        toc_entry = ""
                    
                    # Write page entry
                    f.write(f"Page {page_number}: {toc_entry}\n")
                    
                    # Only include verified subject headings (selected LCSH headings)
                    selected_lcsh = analysis.get('selected_lcsh_headings', [])
                    if selected_lcsh:
                        for heading in selected_lcsh:
                            f.write(f"  - {heading['label']}\n")
                    
                    f.write("\n")
            
            print(f"📋 Created issue-level table of contents: {toc_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error creating issue-level table of contents: {e}")
            return False

    def create_summary_report(self) -> bool:
        """Create a summary report of the metadata generation process."""
        try:
            report_path = os.path.join(self.folder_path, "page_metadata_generation_report.txt")
            
            # Count generated files
            metadata_files = [f for f in os.listdir(self.output_folder) 
                            if f.endswith('_metadata.txt')]
            
            # Skip API stats for counting
            data_items = self.json_data[:-1] if self.json_data and 'api_stats' in self.json_data[-1] else self.json_data
            total_entries = len(data_items)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("SOUTHERN ARCHITECT - PAGE METADATA GENERATION REPORT\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Workflow Type: {self.workflow_type.upper()}\n")
                f.write(f"Source Folder: {self.folder_path}\n")
                f.write(f"Output Folder: {self.output_folder}\n\n")
                
                f.write("GENERATION STATISTICS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total entries in JSON: {total_entries}\n")
                f.write(f"Metadata files generated: {len(metadata_files)}\n")
                f.write(f"Success rate: {(len(metadata_files)/total_entries*100):.1f}%\n\n")
                
                f.write("METADATA STRUCTURE:\n")
                f.write("-" * 30 + "\n")
                f.write("Each metadata file contains:\n")
                f.write("- Page identification (folder, page number, original file path)\n")
                if self.workflow_type == 'text':
                    f.write("- Cleaned OCR text\n")
                else:
                    f.write("- Text transcription\n")
                    f.write("- Visual description\n")
                f.write("- Table of Contents entry\n")
                f.write("- Named entities\n")
                f.write("- Topics (subject headings)\n")
                f.write("- Content warning (if applicable)\n")
                f.write("- Selected LCSH headings (if available)\n")
                f.write("- Processing information\n\n")
                
                f.write("FILES GENERATED:\n")
                f.write("-" * 30 + "\n")
                f.write(f"- {len(metadata_files)} individual page metadata files\n")
                f.write("- 1 index file (00_INDEX.txt)\n")
                f.write("- 1 issue-level table of contents (using folder name)\n")
                f.write("- 1 generation report (this file)\n\n")
                
                f.write("=" * 60 + "\n")
            
            print(f"📋 Created generation report: {report_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error creating summary report: {e}")
            return False
    
    def run(self) -> bool:
        """Main execution method."""
        print(f"\n🎯 SOUTHERN ARCHITECT STEP 4 - PAGE METADATA GENERATION")
        print(f"📁 Processing folder: {self.folder_path}")
        print("-" * 50)
        
        # Detect workflow type
        if not self.detect_workflow_type():
            return False
        
        print(f"🔍 Detected workflow type: {self.workflow_type.upper()}")
        
        # Load JSON data
        if not self.load_json_data():
            return False
        
        # Create output folder
        if not self.create_output_folder():
            return False
        
        # Generate metadata files
        print(f"📝 Generating individual page metadata files...")
        if not self.generate_all_metadata_files():
            return False
        
        # Create index file
        if not self.create_index_file():
            return False
        
        # Create issue-level table of contents
        if not self.create_issue_toc():
            return False
        
        # Create summary report
        if not self.create_summary_report():
            return False
        
        # Final summary
        metadata_files = [f for f in os.listdir(self.output_folder) 
                         if f.endswith('_metadata.txt')]
        
        print(f"\n🎉 PAGE METADATA GENERATION COMPLETED!")
        print(f"✅ Generated {len(metadata_files)} metadata files")
        print(f"📁 Output folder: {self.output_folder}")
        print(f"📋 Index file: {os.path.join(self.output_folder, '00_INDEX.txt')}")
        
        # Get the actual TOC filename that was created
        data_items = self.json_data[:-1] if self.json_data and 'api_stats' in self.json_data[-1] else self.json_data
        if data_items:
            first_entry = data_items[0]
            folder_name = first_entry.get('folder', 'Unknown')
            toc_filename = f"{folder_name}_Table_of_Contents.txt"
            print(f"📋 Issue-level TOC: {os.path.join(self.folder_path, toc_filename)}")
        
        print(f"📋 Generation report: {os.path.join(self.folder_path, 'page_metadata_generation_report.txt')}")
        
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
    parser = argparse.ArgumentParser(description='Generate individual page-level metadata files')
    parser.add_argument('--folder', help='Specific folder path to process')
    parser.add_argument('--newest', action='store_true', help='Process the newest folder in the output directory (default: True if no folder specified)')
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
    
    # Create and run the generator
    generator = PageMetadataGenerator(folder_path)
    success = generator.run()
    
    if not success:
        print("❌ Page metadata generation failed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())