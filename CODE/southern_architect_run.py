#!/usr/bin/env python3
"""
Southern Architect Master Workflow Runner
=========================================

This script runs the complete Southern Architect workflow pipeline:
1. Step 1: OCR Text/Image Metadata Extraction
2. Step 2: Multi-Vocabulary Enhancement (LCSH, FAST, Getty)
3. Step 3: AI-Powered Vocabulary Selection & Clean Output Generation
4. Step 4: Issue-Level Synthesis & Final Metadata
5. Step 5: Entity Authority File Creation

Author: Southern Architect Processing Team
"""

import os
import sys
import subprocess
import logging
import argparse
from datetime import datetime
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SouthernArchitectWorkflowRunner:
    """Master workflow runner for the Southern Architect processing pipeline."""
    
    def __init__(self):
        self.workflow_type = None
        self.output_folder = None
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Define script paths
        self.scripts = {
            'text_step1': os.path.join(self.script_dir, 'southern_architect_step1_text.py'),
            'image_step1': os.path.join(self.script_dir, 'southern_architect_step1_image.py'),
            'step2': os.path.join(self.script_dir, 'southern_architect_step2.py'),
            'step3': os.path.join(self.script_dir, 'southern_architect_step3.py'),
            'step4': os.path.join(self.script_dir, 'southern_architect_step4.py'),
            'step5': os.path.join(self.script_dir, 'southern_architect_step5.py')
        }
    
    def print_banner(self):
        """Print the workflow banner."""
        print("\n" + "="*70)
        print("  SOUTHERN ARCHITECT COMPLETE WORKFLOW PIPELINE")
        print("="*70)
        print("Processing historical architectural journals (1892-1931)")
        print("University of Texas at Austin Digital Collections")
        print("="*70 + "\n")
    
    def get_workflow_type(self, workflow_type: Optional[str] = None) -> bool:
        """Get the workflow type from user input or parameter."""
        if workflow_type:
            if workflow_type.lower() in ['text', 'image']:
                self.workflow_type = workflow_type.lower()
                print(f" Using specified workflow type: {self.workflow_type.upper()}")
                return True
            else:
                print(f" Invalid workflow type: {workflow_type}. Must be 'text' or 'image'.")
                return False
        
        # Interactive selection
        print("📋 SELECT WORKFLOW TYPE:")
        print("1. Text Workflow   - Process OCR text files (.txt)")
        print("2. Image Workflow  - Process image files (.jpg, .png)")
        print()
        
        while True:
            try:
                choice = input("Enter your choice (1 or 2): ").strip()
                
                if choice == "1":
                    self.workflow_type = "text"
                    break
                elif choice == "2":
                    self.workflow_type = "image"
                    break
                else:
                    print(" Invalid choice. Please enter 1 or 2.")
                    
            except KeyboardInterrupt:
                print("\n\n Workflow cancelled by user.")
                return False
        
        print(f" Selected: {self.workflow_type.upper()} workflow\n")
        return True
    
    def announce_step(self, step_name: str, step_description: str) -> None:
        """Announce the next step that will run automatically."""
        print(f"\n ➡️ STARTING: {step_name}")
        print(f" {step_description}")
        print(" Running automatically...")
        print("-" * 50)
    
    def run_script(self, script_path: str, args: list = None) -> bool:
        """Run a Python script with optional arguments."""
        if not os.path.exists(script_path):
            print(f" Script not found: {script_path}")
            return False
        
        cmd = [sys.executable, script_path]
        if args:
            cmd.extend(args)
        
        print(f" Running: {' '.join(cmd)}")
        print("-" * 50)
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            print("-" * 50)
            print(f" Script completed successfully: {os.path.basename(script_path)}")
            return True
            
        except subprocess.CalledProcessError as e:
            print("-" * 50)
            print(f" Script failed: {os.path.basename(script_path)}")
            print(f"Exit code: {e.returncode}")
            return False
        except KeyboardInterrupt:
            print("\n\n  Script interrupted by user.")
            return False
    
    def find_newest_output_folder(self) -> Optional[str]:
        """Find the newest output folder for the current workflow type."""
        base_output_dir = "/Users/hannahmoutran/Desktop/southern_architect/CODE/output_folders"
        
        if not os.path.exists(base_output_dir):
            return None
        
        folders = [f for f in os.listdir(base_output_dir) 
                  if os.path.isdir(os.path.join(base_output_dir, f))]
        
        if not folders:
            return None
        
        # Sort by modification time (newest first)
        folders.sort(key=lambda x: os.path.getmtime(os.path.join(base_output_dir, x)), reverse=True)
        
        return os.path.join(base_output_dir, folders[0])
    
    def run_step1(self, resume: bool = False, model: str = None) -> bool:
        """Run Step 1: Initial metadata extraction."""
        step_name = f"STEP 1: {self.workflow_type.upper()} METADATA EXTRACTION"
        step_description = f"Extract metadata from {self.workflow_type} files using AI with enhanced parsing and checkpoint support"
        
        self.announce_step(step_name, step_description)
        
        script_key = f"{self.workflow_type}_step1"
        script_path = self.scripts[script_key]
        
        args = []
        if resume:
            args.append("--resume")
        if model:
            args.extend(["--model", model])
        
        success = self.run_script(script_path, args)
        
        if success:
            # Find the output folder that was just created
            self.output_folder = self.find_newest_output_folder()
            if self.output_folder:
                print(f"📁 Output folder: {self.output_folder}")
            else:
                print("⚠️  Could not find output folder")
        
        return success
    
    def run_step2(self) -> bool:
        """Run Step 2: Multi-Vocabulary Enhancement."""
        step_name = "STEP 2: MULTI-VOCABULARY ENHANCEMENT"
        step_description = "Add LCSH, FAST, Getty AAT and TGN controlled vocabulary terms with API logging"
        
        self.announce_step(step_name, step_description)
        
        args = []
        if self.output_folder:
            args.extend(["--folder", self.output_folder])
        else:
            args.append("--newest")
        
        return self.run_script(self.scripts['step2'], args)
    
    def run_step3(self, model: str = "gpt-4o-mini-2024-07-18") -> bool:
        """Run Step 3: AI-Powered Vocabulary Selection."""
        step_name = "STEP 3: AI-POWERED VOCABULARY SELECTION"
        step_description = "Use AI to select best vocabulary terms and generate clean metadata outputs"
        
        self.announce_step(step_name, step_description)
        
        args = ["--model", model]
        if self.output_folder:
            args.extend(["--folder", self.output_folder])
        else:
            args.append("--newest")
        
        return self.run_script(self.scripts['step3'], args)
    
    def run_step4(self, model: str = "gpt-4o-2024-08-06") -> bool:
        """Run Step 4: Issue-Level Synthesis."""
        step_name = "STEP 4: ISSUE-LEVEL SYNTHESIS"
        step_description = "Create issue-level metadata file with subject headings and geographic terms"
        
        self.announce_step(step_name, step_description)
        
        args = ["--model", model]
        if self.output_folder:
            args.extend(["--folder", self.output_folder])
        else:
            args.append("--newest")
        
        return self.run_script(self.scripts['step4'], args)
    
    def run_step5(self) -> bool:
        """Run Step 5: Entity Authority File Creation."""
        step_name = "STEP 5: ENTITY AUTHORITY FILE CREATION"
        step_description = "Build authority file for named entities with type classification"
        
        self.announce_step(step_name, step_description)
        
        args = []
        if self.output_folder:
            args.extend(["--folder", self.output_folder])
        else:
            args.append("--newest")
        
        return self.run_script(self.scripts['step5'], args)
    
    def check_dependencies(self) -> bool:
        """Check if all required scripts exist."""
        missing_scripts = []
        
        for script_name, script_path in self.scripts.items():
            if not os.path.exists(script_path):
                missing_scripts.append(script_path)
        
        if missing_scripts:
            print(" Missing required scripts:")
            for script in missing_scripts:
                print(f"   - {script}")
            return False
        
        print(" All required scripts found")
        return True
    
    def run_complete_workflow(self, resume: bool = False, 
                            step1_model: str = None,
                            step3_model: str = "gpt-4o-mini-2024-07-18",
                            step4_model: str = "gpt-4o-2024-08-06") -> bool:
        """Run the complete workflow pipeline."""
        
        self.print_banner()
        
        # Check dependencies
        if not self.check_dependencies():
            return False
        
        print(" WORKFLOW OVERVIEW:")
        print("Step 1: Extract metadata from source files (with batch processing if applicable)")
        print("Step 2: Enhance with controlled vocabulary terms (LCSH, FAST, Getty)")
        print("Step 3: AI-powered selection of optimal vocabulary terms")
        print("Step 4: Create issue-level description")
        print("Step 5: Build entity authority file with type classification")
        print(" All steps will run automatically - you can step away!")
        print(" Comprehensive logging and cost tracking throughout")
        print()
        
        # Track overall success
        overall_success = True
        failed_steps = []
        start_time = datetime.now()
        
        print(f" Starting complete workflow at {start_time.strftime('%H:%M:%S')}")
        print("="*70)
        
        # Run Step 1
        if not self.run_step1(resume, step1_model):
            print(" Step 1 failed - stopping workflow")
            return False
        
        # Run Step 2
        if not self.run_step2():
            overall_success = False
            failed_steps.append("Step 2: Multi-Vocabulary Enhancement")
        
        # Run Step 3
        if not self.run_step3(step3_model):
            overall_success = False
            failed_steps.append("Step 3: AI-Powered Vocabulary Selection")
        
        # Run Step 4
        if not self.run_step4(step4_model):
            overall_success = False
            failed_steps.append("Step 4: Issue-Level Synthesis")
        
        # Run Step 5
        if not self.run_step5():
            overall_success = False
            failed_steps.append("Step 5: Entity Authority File Creation")
        
        # Final summary
        end_time = datetime.now()
        total_duration = end_time - start_time
        
        print("\n" + "="*70)
        print(" WORKFLOW COMPLETION SUMMARY")
        print("="*70)
        print(f"  Total runtime: {total_duration}")
        print(f" Started: {start_time.strftime('%H:%M:%S')}")
        print(f" Finished: {end_time.strftime('%H:%M:%S')}")
        print()
        
        if overall_success:
            print(" All steps completed successfully!")
            print(" Your Southern Architect workflow is complete!")
        else:
            print(f"  Workflow completed with {len(failed_steps)} failed step(s):")
            for step in failed_steps:
                print(f"    {step}")
            print("\n You can re-run individual steps if needed")
        
        if self.output_folder:
            print(f"\n Final output folder: {self.output_folder}")
            print(" Generated files include:")
            print("   Excel and JSON workflow files")
            print("   Page-level metadata files")
            print("   Issue-level metadata files")
            print("   Issue index with key metadata")
            print("   Vocabulary mapping report")
            print("   Entity authority file with type classification")
            print("   Processing reports and API logs")
            print("   Cost tracking and token usage logs")
        
        print("\n" + "="*70)
        
        return overall_success

def main():
    parser = argparse.ArgumentParser(
        description='Run the complete Southern Architect workflow pipeline with enhanced features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python southern_architect_run.py                              # Interactive mode
  python southern_architect_run.py --workflow text              # Text workflow
  python southern_architect_run.py --workflow image             # Image workflow
  python southern_architect_run.py --resume                     # Resume text workflow from checkpoint
  python southern_architect_run.py --workflow text --step1-model gpt-4o-2024-08-06
  python southern_architect_run.py --step3-model gpt-4o-2024-08-06 --step4-model gpt-4o-mini-2024-07-18

Features:
  - Step 1: AI: Initial metadata extraction with batch processing
  - Step 2: AI: Multi-vocabulary enhancement (LCSH, FAST, Getty) with API logging
  - Step 3: AI: Vocabulary selection
  - Step 4: AI: Issue-level synthesis with geographic terms
  - Step 5: Entity authority file creation with type classification
  - Comprehensive cost tracking and token logging throughout
        """
    )
    
    parser.add_argument('--workflow', choices=['text', 'image'], 
                       help='Workflow type (text or image)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint (Step 1 only)')
    parser.add_argument('--step1-model', 
                       help='Model for Step 1 (default: gpt-4o-2024-08-06 for images, gpt-4o-2024-08-06 for text)')
    parser.add_argument('--step3-model', default="gpt-4o-mini-2024-07-18",
                       help='Model for Step 3 (vocabulary selection)')
    parser.add_argument('--step4-model', default="gpt-4o-2024-08-06",
                       help='Model for Step 4 (issue synthesis)')
    
    args = parser.parse_args()
    
    try:
        # Create workflow runner
        runner = SouthernArchitectWorkflowRunner()
        
        # Get workflow type
        if not runner.get_workflow_type(args.workflow):
            return 1
        
        # Set default step1 model if not specified
        step1_model = args.step1_model
        if not step1_model:
            step1_model = "gpt-4o-2024-08-06"  # Same default for both text and image
        
        # Run complete workflow
        success = runner.run_complete_workflow(
            resume=args.resume,
            step1_model=step1_model,
            step3_model=args.step3_model,
            step4_model=args.step4_model
        )
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\n Workflow cancelled by user.")
        return 1
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        logging.exception("Unexpected error in workflow runner")
        return 1

if __name__ == "__main__":
    exit(main())