# southern_architect_entity_authority.py
# Create local authority file for Southern Architect named entities

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import re
import argparse

class EntityAuthority:
    """Build authority records for Southern Architect entities."""
    
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.workflow_type = None
        self.json_data = None
        
    def detect_workflow_type(self) -> bool:
        """Detect workflow type."""
        text_files = ['text_workflow.xlsx', 'text_workflow.json']
        image_files = ['image_workflow.xlsx', 'image_workflow.json']
        
        has_text_files = all(os.path.exists(os.path.join(self.folder_path, f)) for f in text_files)
        has_image_files = all(os.path.exists(os.path.join(self.folder_path, f)) for f in image_files)
        
        if has_text_files and not has_image_files:
            self.workflow_type = 'text'
        elif has_image_files and not has_text_files:
            self.workflow_type = 'image'
        else:
            return False
        return True
    
    def load_json_data(self) -> bool:
        """Load JSON data."""
        json_filename = f"{self.workflow_type}_workflow.json"
        json_path = os.path.join(self.folder_path, json_filename)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.json_data = json.load(f)
            return True
        except Exception as e:
            logging.error(f"Error loading JSON data: {e}")
            return False
    
    def parse_typed_entity(self, entity_str: str) -> Tuple[str, str]:
        """Parse entity string to extract name and type.
        
        Input: 'John F. Staub (architect)' or 'Atlanta, Ga. (location)'
        Output: ('John F. Staub', 'architect')
        """
        # Look for pattern: 'Name (type)'
        match = re.match(r'^(.+?)\s*\(([^)]+)\)\s*$', entity_str.strip())
        
        if match:
            name = match.group(1).strip()
            entity_type = match.group(2).strip().lower()
            return name, entity_type
        else:
            # If no type indicator found, return original string and classify it
            name = entity_str.strip()
            entity_type = self.classify_entity_fallback(name)
            return name, entity_type
    
    def classify_entity_fallback(self, entity: str) -> str:
        """Fallback classification for entities without type indicators."""
        entity_lower = entity.lower()
        
        # Geographic locations (common suffixes)
        if any(suffix in entity for suffix in [', Tenn.', ', Md.', ', Va.', ', Ga.', ', Ala.', ', N. C.', ', Fla.', ', S. C.', ', Tex.']):
            return 'location'
        
        # Residences
        if 'residence' in entity_lower or 'house' in entity_lower:
            return 'building'
        
        # Organizations/Companies
        if any(term in entity_lower for term in ['company', 'co.', 'inc.', 'corp.', 'publishing', 'association', 'institute']):
            return 'organization'
        
        # Architectural firms (multiple names with &, commas)
        if any(pattern in entity for pattern in ['&', ' and ', ', ']):
            return 'firm'
        
        # Personal names (has initials or multiple capitalized words)
        if re.match(r'^[A-Z]\. ?[A-Z]\.? [A-Z]', entity):  # J. Staub, E. R. Denmark
            return 'person'
        elif re.match(r'^[A-Z][a-z]+ [A-Z]\. [A-Z]', entity):  # Laurence H. Fowler
            return 'person'
        elif len(entity.split()) >= 2 and all(word[0].isupper() for word in entity.split()[:2]):
            return 'person'
        
        return 'unknown'
    
    def extract_all_entities(self) -> Dict[str, Dict]:
        """Extract all named entities and analyze patterns."""
        entity_records = defaultdict(lambda: {
            'appearances': [],
            'contexts': [],
            'entity_type': 'unknown',
            'locations': set(),
            'time_periods': set(),
            'original_forms': set()  # Track different ways entity appears
        })
        
        # Skip API stats
        data_items = self.json_data[:-1] if self.json_data and 'api_stats' in self.json_data[-1] else self.json_data
        
        for item in data_items:
            if 'analysis' in item:
                analysis = item['analysis']
                folder = item.get('folder', 'unknown')
                page = item.get('page_number', 0)
                
                # Extract entities
                entities = analysis.get('named_entities', [])
                if isinstance(entities, str):
                    entities = [e.strip() for e in entities.split(',') if e.strip()]
                
                # Extract context for additional analysis
                text_content = analysis.get('cleaned_text', '') or analysis.get('text_transcription', '')
                toc_entry = analysis.get('toc_entry', '')
                context = f"{text_content} {toc_entry}".lower()
                
                for entity_str in entities:
                    if not entity_str or len(entity_str.strip()) < 2:
                        continue
                    
                    # Parse the typed entity
                    entity_name, entity_type = self.parse_typed_entity(entity_str)
                    
                    if not entity_name:
                        continue
                    
                    record = entity_records[entity_name]
                    record['appearances'].append(f"{folder}_page{page}")
                    record['contexts'].append(context)
                    record['original_forms'].add(entity_str.strip())
                    
                    # Set entity type (use the parsed type or keep existing if already set)
                    if record['entity_type'] == 'unknown' or entity_type != 'unknown':
                        record['entity_type'] = entity_type
                    
                    # Extract year from folder name
                    year_match = re.search(r'(\d{4})', folder)
                    if year_match:
                        record['time_periods'].add(year_match.group(1))
        
        # Convert sets to lists for JSON serialization and calculate frequency
        for entity, record in entity_records.items():
            record['locations'] = list(record['locations'])
            record['time_periods'] = list(record['time_periods'])
            record['original_forms'] = list(record['original_forms'])
            record['frequency'] = len(record['appearances'])
        
        return dict(entity_records)
    
    def create_authority_file(self, entity_records: Dict) -> bool:
        """Create comprehensive authority file."""
        try:
            authority_path = os.path.join(self.folder_path, "southern_architect_entity_authority.json")
            
            # Create structured authority data
            authority_data = {
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "source": "Southern Architect and Building News (1892-1931)",
                    "total_entities": len(entity_records),
                    "extraction_method": "Named Entity Recognition via GPT-4 with Type Classification",
                    "entity_types": list(set(record['entity_type'] for record in entity_records.values()))
                },
                "entities": entity_records,
                "statistics": self.generate_statistics(entity_records)
            }
            
            with open(authority_path, 'w', encoding='utf-8') as f:
                json.dump(authority_data, f, indent=2, ensure_ascii=False)
            
            print(f"📋 Created entity authority file: {authority_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error creating authority file: {e}")
            return False
    
    def generate_statistics(self, entity_records: Dict) -> Dict:
        """Generate statistics about entities."""
        stats = {
            'by_type': defaultdict(int),
            'by_frequency': defaultdict(int),
            'by_decade': defaultdict(int),
            'type_examples': defaultdict(list)
        }
        
        for entity, record in entity_records.items():
            entity_type = record['entity_type']
            frequency = record['frequency']
            
            stats['by_type'][entity_type] += 1
            
            # Add examples for each type (up to 3)
            if len(stats['type_examples'][entity_type]) < 3:
                stats['type_examples'][entity_type].append(entity)
            
            # Frequency buckets
            if frequency == 1:
                stats['by_frequency']['single_mention'] += 1
            elif frequency <= 3:
                stats['by_frequency']['low_frequency'] += 1
            elif frequency <= 10:
                stats['by_frequency']['medium_frequency'] += 1
            else:
                stats['by_frequency']['high_frequency'] += 1
            
            # Decade analysis
            for year in record['time_periods']:
                decade = f"{year[:3]}0s"
                stats['by_decade'][decade] += 1
        
        # Convert defaultdicts to regular dicts
        return {
            'by_type': dict(stats['by_type']),
            'by_frequency': dict(stats['by_frequency']),
            'by_decade': dict(stats['by_decade']),
            'type_examples': dict(stats['type_examples'])
        }
    
    def create_human_readable_report(self, entity_records: Dict) -> bool:
        """Create human-readable authority report."""
        try:
            report_path = os.path.join(self.folder_path, "entity_authority_report.txt")
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("SOUTHERN ARCHITECT ENTITY AUTHORITY REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Entities: {len(entity_records)}\n\n")
                
                # Show statistics summary
                stats = self.generate_statistics(entity_records)
                f.write("ENTITY TYPE BREAKDOWN:\n")
                f.write("-" * 25 + "\n")
                for entity_type, count in sorted(stats['by_type'].items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {entity_type.upper().replace('_', ' ')}: {count} entities\n")
                f.write("\n")
                
                # Group by type for detailed listing
                by_type = defaultdict(list)
                for entity, record in entity_records.items():
                    by_type[record['entity_type']].append((entity, record))
                
                for entity_type in sorted(by_type.keys()):
                    entities = by_type[entity_type]
                    f.write(f"{entity_type.upper().replace('_', ' ')} ({len(entities)} entities):\n")
                    f.write("-" * 40 + "\n")
                    
                    # Sort by frequency (most mentioned first)
                    entities.sort(key=lambda x: x[1]['frequency'], reverse=True)
                    
                    for entity, record in entities:
                        f.write(f"  {entity}\n")
                        f.write(f"    Frequency: {record['frequency']} appearances\n")
                        f.write(f"    Time periods: {', '.join(sorted(record['time_periods']))}\n")
                        
                        # Show different forms if multiple
                        if len(record['original_forms']) > 1:
                            f.write(f"    Forms: {', '.join(record['original_forms'])}\n")
                        
                        f.write(f"    Pages: {', '.join(record['appearances'][:5])}")
                        if len(record['appearances']) > 5:
                            f.write(f" ... (+{len(record['appearances'])-5} more)")
                        f.write("\n\n")
                
            print(f"📋 Created human-readable report: {report_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error creating report: {e}")
            return False
    
    def run(self) -> bool:
        """Main execution method."""
        print(f"\n🎯 SOUTHERN ARCHITECT ENTITY AUTHORITY BUILDER")
        print(f"📁 Processing folder: {self.folder_path}")
        print("-" * 50)
        
        if not self.detect_workflow_type():
            print("❌ Could not detect workflow type")
            return False
        
        print(f"🔍 Detected workflow type: {self.workflow_type.upper()}")
        
        if not self.load_json_data():
            print("❌ Could not load JSON data")
            return False
        
        # Extract all entities
        print("🔄 Extracting and analyzing typed named entities...")
        entity_records = self.extract_all_entities()
        
        if not entity_records:
            print("⚠️  No named entities found")
            return False
        
        print(f"📊 Found {len(entity_records)} unique entities")
        
        # Show type breakdown
        type_counts = defaultdict(int)
        for record in entity_records.values():
            type_counts[record['entity_type']] += 1
        
        for entity_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {entity_type}: {count} entities")
        
        # Create authority files
        if not self.create_authority_file(entity_records):
            return False
        
        if not self.create_human_readable_report(entity_records):
            return False
        
        print(f"\n🎉 ENTITY AUTHORITY BUILDING COMPLETED!")
        print(f"📋 JSON authority file: southern_architect_entity_authority.json")
        print(f"📋 Human-readable report: entity_authority_report.txt")
        
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
    parser = argparse.ArgumentParser(description='Build entity authority file for Southern Architect with typed entities')
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

    builder = EntityAuthority(folder_path)
    success = builder.run()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())