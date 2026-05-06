#!/usr/bin/env python3
"""
SA HTML Review Interface
========================

Generates a local HTML review interface for Southern Architect magazine page metadata.
Allows reviewers to edit AI-generated fields, approve or reject vocabulary terms,
and export decisions to JSON for integration back into the workflow.

Usage:
    python sa_html_review.py
    python sa_html_review.py --folder /path/to/output_folder
    python sa_html_review.py --records-per-page 5
"""

import os
import sys
import json
import shutil
import argparse
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from shared_utilities import find_newest_folder
from sa_workflow_config import FOLDER_CONFIG


def detect_workflow_type(folder_path):
    """Detect whether this is an image or text workflow by checking which JSON exists."""
    collection_metadata_dir = os.path.join(folder_path, "metadata", "collection_metadata")
    image_json = os.path.join(collection_metadata_dir, "image_workflow.json")
    text_json = os.path.join(collection_metadata_dir, "text_workflow.json")

    if os.path.exists(image_json):
        return "image", image_json
    if os.path.exists(text_json):
        return "text", text_json
    return None, None


class SAHTMLReviewBuilder:
    """Builds interactive HTML review pages for Southern Architect magazine metadata."""

    def __init__(self, folder_path, records_per_page=10):
        self.folder_path = folder_path
        self.records_per_page = records_per_page
        self.json_data = None
        self.data_items = []
        self.workflow_type = None
        self.workflow_timestamp = None
        self.review_folder = None
        self.images_folder = None
        self.folder_name = os.path.basename(folder_path)

    def load_json_data(self):
        """Detect workflow type and load the appropriate workflow JSON."""
        workflow_type, json_path = detect_workflow_type(self.folder_path)

        if not workflow_type:
            print("Error: Neither image_workflow.json nor text_workflow.json found.")
            print("Expected location: metadata/collection_metadata/")
            return False

        self.workflow_type = workflow_type
        print("Detected workflow type: " + workflow_type)
        print("Loading: " + os.path.basename(json_path))

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.json_data = json.load(f)

            # Filter to only real page records (must have page_number field).
            # Summary items like api_stats and issue_syntheses are excluded.
            self.data_items = [
                item for item in self.json_data
                if isinstance(item, dict) and 'page_number' in item
            ]

            # Extract a timestamp string from the folder name for localStorage namespacing.
            # Folder format: SABN_Metadata_Created_YYYY-MM-DD_Time_HH-MM-SS
            parts = self.folder_name.split('_')
            if len(parts) >= 2:
                self.workflow_timestamp = '_'.join(parts[-2:])
            else:
                self.workflow_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            print("Loaded " + str(len(self.data_items)) + " page records from workflow JSON")
            return True

        except Exception as exc:
            print("Error loading JSON: " + str(exc))
            return False

    def create_review_folder(self):
        """Create review/, review/images/, and review/exports/ directories."""
        self.review_folder = os.path.join(self.folder_path, "review")
        self.images_folder = os.path.join(self.review_folder, "images")
        exports_folder = os.path.join(self.review_folder, "exports")

        os.makedirs(self.review_folder, exist_ok=True)
        os.makedirs(self.images_folder, exist_ok=True)
        os.makedirs(exports_folder, exist_ok=True)

        print("Created review folder structure at: " + self.review_folder)
        return True

    def copy_images(self):
        """Copy page images to review/images/ for portability."""
        copied_count = 0
        skipped_count = 0

        for item in self.data_items:
            image_path = item.get('image_path', '')
            if not image_path:
                continue
            if os.path.exists(image_path):
                filename = os.path.basename(image_path)
                dest_path = os.path.join(self.images_folder, filename)
                try:
                    shutil.copy2(image_path, dest_path)
                    copied_count += 1
                except Exception as exc:
                    print("Warning: Could not copy " + filename + ": " + str(exc))
            else:
                skipped_count += 1

        print("Copied " + str(copied_count) + " images to review/images/")
        if skipped_count > 0:
            print("Skipped " + str(skipped_count) + " images (file not found)")
        return True

    def escape_html(self, text):
        """Escape HTML special characters."""
        if text is None:
            return ""
        return (str(text)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#39;"))

    def escape_js_string(self, text):
        """Escape text for embedding inside a JavaScript string literal."""
        if text is None:
            return ""
        return (str(text)
                .replace("\\", "\\\\")
                .replace("'", "\\'")
                .replace('"', '\\"')
                .replace("\n", "\\n")
                .replace("\r", "\\r"))

    # ------------------------------------------------------------------
    # CSS
    # ------------------------------------------------------------------

    def get_css_styles(self):
        """Return CSS styles for the review interface."""
        return """
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.5;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .header h1 { margin: 0 0 10px 0; font-size: 22px; }
        .header p { margin: 4px 0; opacity: 0.9; font-size: 14px; }

        .navigation {
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .nav-btn {
            background-color: #3498db;
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 5px;
            margin: 0 8px;
            font-weight: bold;
            display: inline-block;
            border: none;
            cursor: pointer;
            font-size: 14px;
        }
        .nav-btn:hover { background-color: #2980b9; }
        .nav-btn.disabled { background-color: #95a5a6; pointer-events: none; }

        .record {
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            margin-bottom: 30px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .record.reviewed { border-left: 5px solid #27ae60; }
        .record.has-edits { border-left: 5px solid #f39c12; }

        .record-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }
        .record-title { font-size: 18px; font-weight: bold; color: #2c3e50; }

        .content-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        @media (max-width: 900px) {
            .content-grid { grid-template-columns: 1fr; }
        }

        .image-container {
            text-align: center;
            background: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
        }
        .image-container img {
            max-width: 100%;
            height: auto;
            border: 2px solid #ddd;
            border-radius: 5px;
            display: block;
            transform-origin: 0 0;
            cursor: grab;
            user-select: none;
            -webkit-user-drag: none;
        }
        .image-container img.panning { cursor: grabbing; }

        /* In-place zoom controls */
        .zoom-controls {
            position: absolute;
            top: 8px;
            right: 8px;
            display: flex;
            gap: 4px;
            z-index: 10;
        }
        .zoom-btn {
            background: rgba(0,0,0,0.52);
            border: 1px solid rgba(255,255,255,0.25);
            color: #fff;
            font-size: 15px;
            font-weight: bold;
            width: 28px;
            height: 28px;
            border-radius: 4px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            line-height: 1;
        }
        .zoom-btn:hover { background: rgba(0,0,0,0.78); }
        .zoom-hint {
            margin-top: 5px;
            font-size: 11px;
            color: #aaa;
            text-align: center;
        }

        .image-filename {
            margin-top: 8px;
            font-size: 11px;
            color: #666;
            word-break: break-all;
        }

        .section-header {
            font-size: 15px;
            font-weight: bold;
            color: #2c3e50;
            margin: 18px 0 10px 0;
            padding-bottom: 5px;
            border-bottom: 2px solid #3498db;
        }

        .field-group {
            margin-bottom: 14px;
            padding: 12px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        .field-group.edited { background: #fff3cd; }

        .field-label {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
            font-size: 13px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .field-label .edit-indicator {
            font-size: 11px;
            color: #f39c12;
            font-weight: normal;
        }

        .field-input {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 13px;
            font-family: inherit;
        }
        .field-input:focus {
            border-color: #3498db;
            outline: none;
            box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
        }
        .field-input.edited { border-color: #f39c12; background: #fffbf0; }

        textarea.field-input { overflow: hidden; resize: none; }

        .list-field { margin-bottom: 5px; }
        .list-item {
            display: flex;
            gap: 8px;
            margin-bottom: 7px;
            align-items: center;
        }
        .list-item input { flex: 1; }
        .remove-btn {
            background: #e74c3c;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 5px 10px;
            cursor: pointer;
            font-size: 12px;
            white-space: nowrap;
        }
        .remove-btn:hover { background: #c0392b; }
        .add-btn {
            background: #27ae60;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 7px 14px;
            cursor: pointer;
            font-size: 13px;
            margin-top: 5px;
        }
        .add-btn:hover { background: #229954; }

        .vocab-section { margin-top: 20px; }
        .vocab-group {
            margin-bottom: 15px;
            padding: 12px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        .vocab-group-title {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 13px;
        }
        .vocab-term {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 7px 10px;
            margin: 5px 0;
            background: white;
            border-radius: 4px;
            border: 1px solid #ddd;
            flex-wrap: wrap;
        }
        .vocab-term.approved { background: #d4edda; border-color: #28a745; }
        .vocab-term.rejected { background: #f8d7da; border-color: #dc3545; text-decoration: line-through; opacity: 0.7; }
        .vocab-term-label { flex: 1; font-size: 13px; min-width: 100px; }
        .vocab-term-source {
            font-size: 11px;
            padding: 2px 6px;
            background: #6c757d;
            color: white;
            border-radius: 3px;
            white-space: nowrap;
        }
        .vocab-term-source.lcsh { background: #007bff; }
        .vocab-term-source.fast { background: #28a745; }
        .vocab-term-source.fastgeographic { background: #17a2b8; }
        .vocab-term-source.aat { background: #dc3545; }
        .vocab-term-source.tgn { background: #6f42c1; }
        .for-badge {
            font-size: 10px;
            padding: 2px 6px;
            background: #e9ecef;
            color: #555;
            border-radius: 3px;
            white-space: nowrap;
            max-width: 160px;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .term-uri {
            font-size: 11px;
            color: #3498db;
            text-decoration: none;
            padding: 2px 6px;
            background: #e8f4fc;
            border-radius: 3px;
        }
        .term-uri:hover { background: #d0e8f7; text-decoration: underline; }
        .term-actions { display: flex; gap: 5px; }
        .term-btn {
            padding: 4px 10px;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 12px;
        }
        .term-btn.reject { background: #dc3545; color: white; }
        .term-btn.approve { background: #28a745; color: white; }
        .term-btn.reject:hover { background: #c82333; }
        .term-btn.approve:hover { background: #218838; }

        /* Topic cascade-reject */
        .topic-item { align-items: center; }
        .topic-item.topic-rejected .field-input {
            text-decoration: line-through;
            color: #999;
            background: #fdf3f3;
        }
        .reject-topic-btn {
            padding: 3px 8px;
            background: #dc3545;
            color: white;
            border: none;
            border-radius: 3px;
            font-size: 11px;
            cursor: pointer;
            white-space: nowrap;
        }
        .reject-topic-btn:hover { background: #c82333; }
        .restore-topic-btn {
            padding: 3px 8px;
            background: #6c757d;
            color: white;
            border: none;
            border-radius: 3px;
            font-size: 11px;
            cursor: pointer;
            white-space: nowrap;
        }
        .restore-topic-btn:hover { background: #545b62; }

        .export-section {
            background: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .export-btn {
            background: #e74c3c;
            color: white;
            border: none;
            padding: 14px 28px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            font-size: 15px;
        }
        .export-btn:hover { background: #c0392b; }

        .progress-bar {
            width: 100%;
            height: 18px;
            background: #ddd;
            border-radius: 10px;
            overflow: hidden;
            margin: 8px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #27ae60, #2ecc71);
            transition: width 0.3s;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .summary-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .summary-card .number { font-size: 30px; font-weight: bold; color: #2c3e50; }
        .summary-card .label { color: #666; font-size: 13px; }

        .page-links {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .page-link {
            background-color: #3498db;
            color: white;
            padding: 14px;
            text-decoration: none;
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
            font-size: 14px;
        }
        .page-link:hover { background-color: #2980b9; }

        .notes-section { margin-top: 18px; }
        .reviewed-checkbox {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 10px;
            font-size: 14px;
        }
        .reviewed-checkbox input { width: 18px; height: 18px; cursor: pointer; }

        .add-term-form {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 10px;
        }
        .add-term-form label { font-size: 11px; color: #666; display: block; margin-bottom: 3px; }
        .add-term-form input,
        .add-term-form select {
            width: 100%;
            padding: 7px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 13px;
        }
        """

    # ------------------------------------------------------------------
    # JavaScript
    # ------------------------------------------------------------------

    def get_javascript(self):
        """Return JavaScript for the review interface."""
        return f"""
        const STORAGE_PREFIX = 'sa-review-{self.escape_js_string(self.workflow_timestamp)}-';
        const WORKFLOW_FOLDER = '{self.escape_js_string(self.folder_name)}';
        const WORKFLOW_TYPE = '{self.escape_js_string(self.workflow_type)}';

        function autoResize(el) {{
            el.style.height = 'auto';
            el.style.height = el.scrollHeight + 'px';
        }}

        function setStorage(key, value) {{
            try {{
                localStorage.setItem(STORAGE_PREFIX + key, JSON.stringify(value));
            }} catch (e) {{
                console.error('Storage write error:', e);
            }}
        }}

        function getStorage(key, defaultValue) {{
            if (defaultValue === undefined) defaultValue = null;
            try {{
                const raw = localStorage.getItem(STORAGE_PREFIX + key);
                return raw !== null ? JSON.parse(raw) : defaultValue;
            }} catch (e) {{
                console.error('Storage read error:', e);
                return defaultValue;
            }}
        }}

        function promptForReviewerName() {{
            let name = getStorage('reviewer-name', null);
            if (!name) {{
                name = prompt('Welcome! Please enter your name (saved for this review session):');
                if (name && name.trim()) {{
                    setStorage('reviewer-name', name.trim());
                }}
            }}
            return name;
        }}

        function autoMarkReviewed(recordId) {{
            const cb = document.getElementById('reviewed-' + recordId);
            if (cb && !cb.checked) {{
                cb.checked = true;
                setStorage('reviewed-' + recordId, true);
                updateProgress();
            }}
        }}

        function saveFieldValue(recordId, fieldName, value) {{
            const edits = getStorage('edits-' + recordId, {{}});
            const original = getStorage('original-field-' + fieldName + '-' + recordId, '');
            if (value !== original) {{
                edits[fieldName] = {{ value: value, original: original, edited: true }};
                autoMarkReviewed(recordId);
            }} else {{
                delete edits[fieldName];
            }}
            setStorage('edits-' + recordId, edits);
            updateFieldStyle(recordId, fieldName, value !== original);
            updateRecordStatus(recordId);
        }}

        function updateFieldStyle(recordId, fieldName, isEdited) {{
            const input = document.getElementById('field-' + recordId + '-' + fieldName);
            const group = input ? input.closest('.field-group') : null;
            if (input) {{
                input.classList.toggle('edited', isEdited);
            }}
            if (group) {{
                group.classList.toggle('edited', isEdited);
            }}
        }}

        function updateRecordStatus(recordId) {{
            const el = document.getElementById('record-' + recordId);
            if (!el) return;
            const edits = getStorage('edits-' + recordId, {{}});
            const reviewed = getStorage('reviewed-' + recordId, false);
            el.classList.toggle('reviewed', reviewed);
            el.classList.toggle('has-edits', Object.keys(edits).length > 0);
        }}

        function setReviewed(recordId, checked) {{
            setStorage('reviewed-' + recordId, checked);
            updateRecordStatus(recordId);
            updateProgress();
        }}

        function saveNotes(recordId, notes) {{
            setStorage('notes-' + recordId, notes);
        }}

        function saveListField(recordId, fieldName) {{
            const container = document.getElementById(fieldName + '-' + recordId);
            if (!container) return;
            const inputs = container.querySelectorAll('.list-item input');
            const values = [];
            inputs.forEach(function(input) {{
                const v = input.value.trim();
                if (v) values.push(v);
            }});
            const edits = getStorage('edits-' + recordId, {{}});
            const original = getStorage('original-list-' + fieldName + '-' + recordId, null);
            edits[fieldName] = {{ value: values, original: original, edited: true }};
            setStorage('edits-' + recordId, edits);
            autoMarkReviewed(recordId);
            updateRecordStatus(recordId);
        }}

        function addListItem(recordId, fieldName) {{
            const container = document.getElementById(fieldName + '-' + recordId);
            const div = document.createElement('div');
            div.className = 'list-item';
            div.innerHTML =
                '<input type="text" class="field-input" placeholder="Enter value..." ' +
                'onchange="saveListField(' + recordId + ', \\'' + fieldName + '\\')">' +
                '<button class="remove-btn" onclick="this.parentElement.remove(); ' +
                'saveListField(' + recordId + ', \\'' + fieldName + '\\')">Remove</button>';
            container.appendChild(div);
        }}

        function setTermStatus(recordId, termId, status) {{
            const decisions = getStorage('terms-' + recordId, {{}});
            decisions[termId] = status;
            setStorage('terms-' + recordId, decisions);

            const termEl = document.getElementById('term-' + recordId + '-' + termId);
            if (termEl) {{
                termEl.classList.remove('approved', 'rejected');
                termEl.classList.add(status);
                const rejectBtn = termEl.querySelector('.term-btn.reject');
                const approveBtn = termEl.querySelector('.term-btn.approve');
                const defaultStatus = termEl.dataset.default;

                if (status === 'rejected') {{
                    if (rejectBtn) rejectBtn.style.display = 'none';
                    if (approveBtn) {{
                        approveBtn.style.display = 'inline-block';
                        approveBtn.textContent = 'Restore';
                    }}
                }} else if (status === 'approved') {{
                    if (defaultStatus === 'approved') {{
                        // Default-approved: show Reject, hide Restore
                        if (rejectBtn) rejectBtn.style.display = 'inline-block';
                        if (approveBtn) approveBtn.style.display = 'none';
                    }} else {{
                        // Opt-in: show Remove, hide Add
                        if (rejectBtn) {{
                            rejectBtn.style.display = 'inline-block';
                            rejectBtn.textContent = 'Remove';
                        }}
                        if (approveBtn) approveBtn.style.display = 'none';
                    }}
                }}
            }}
            autoMarkReviewed(recordId);
            updateRecordStatus(recordId);
        }}

        function addCustomTerm(recordId) {{
            const labelInput = document.getElementById('new-term-label-' + recordId);
            const uriInput = document.getElementById('new-term-uri-' + recordId);
            const sourceSelect = document.getElementById('new-term-source-' + recordId);

            const label = labelInput.value.trim();
            const uri = uriInput.value.trim();
            const source = sourceSelect.value;

            if (!label) {{
                alert('Please enter a term label.');
                return;
            }}

            const customTerms = getStorage('custom-terms-' + recordId, []);
            const termId = 'custom-' + Date.now();
            customTerms.push({{ id: termId, label: label, uri: uri, source: source }});
            setStorage('custom-terms-' + recordId, customTerms);

            const container = document.getElementById('custom-terms-container-' + recordId);
            const uriDisplay = uri
                ? '<a href="' + uri + '" target="_blank" class="term-uri" title="' + uri + '">URI</a>'
                : '';
            const sourceClass = source.toLowerCase().replace(/[^a-z]/g, '');
            const html =
                '<div class="vocab-term approved" id="term-' + recordId + '-' + termId + '">' +
                '<span class="vocab-term-label">' + label + '</span>' +
                uriDisplay +
                '<span class="vocab-term-source ' + sourceClass + '">' + source + '</span>' +
                '<div class="term-actions">' +
                '<button class="remove-btn" onclick="removeCustomTerm(' + recordId + ', \\'' + termId + '\\')">Remove</button>' +
                '</div></div>';
            container.insertAdjacentHTML('beforeend', html);

            labelInput.value = '';
            uriInput.value = '';
            autoMarkReviewed(recordId);
            updateRecordStatus(recordId);
        }}

        function removeCustomTerm(recordId, termId) {{
            let customTerms = getStorage('custom-terms-' + recordId, []);
            customTerms = customTerms.filter(function(t) {{ return t.id !== termId; }});
            setStorage('custom-terms-' + recordId, customTerms);
            const el = document.getElementById('term-' + recordId + '-' + termId);
            if (el) el.remove();
            updateRecordStatus(recordId);
        }}

        function rejectTopicCascade(recordId, topicKey) {{
            // Mark the topic item visually
            const item = document.getElementById('topic-item-' + recordId + '-' + topicKey);
            if (item) {{
                item.classList.add('topic-rejected');
                const rejectBtn = item.querySelector('.reject-topic-btn');
                const restoreBtn = item.querySelector('.restore-topic-btn');
                if (rejectBtn) rejectBtn.style.display = 'none';
                if (restoreBtn) restoreBtn.style.display = 'inline-block';
            }}
            // Persist
            const rejected = getStorage('rejected-topics-' + recordId, {{}});
            rejected[topicKey] = true;
            setStorage('rejected-topics-' + recordId, rejected);
            // Cascade: reject all vocab terms whose data-for-topics includes this key
            const record = document.getElementById('record-' + recordId);
            if (!record) return;
            record.querySelectorAll('.vocab-term[data-for-topics]').forEach(function(termEl) {{
                const keys = termEl.getAttribute('data-for-topics').split(' ');
                if (keys.indexOf(topicKey) === -1) return;
                const prefix = 'term-' + recordId + '-';
                const termId = termEl.id.slice(prefix.length);
                setTermStatus(recordId, termId, 'rejected');
            }});
            updateRecordStatus(recordId);
        }}

        function restoreTopicItem(recordId, topicKey) {{
            const item = document.getElementById('topic-item-' + recordId + '-' + topicKey);
            if (item) {{
                item.classList.remove('topic-rejected');
                const rejectBtn = item.querySelector('.reject-topic-btn');
                const restoreBtn = item.querySelector('.restore-topic-btn');
                if (rejectBtn) rejectBtn.style.display = 'inline-block';
                if (restoreBtn) restoreBtn.style.display = 'none';
            }}
            const rejected = getStorage('rejected-topics-' + recordId, {{}});
            delete rejected[topicKey];
            setStorage('rejected-topics-' + recordId, rejected);
            // Note: cascaded term rejections are not automatically undone.
            updateRecordStatus(recordId);
        }}

        function updateProgress() {{
            const total = parseInt(document.body.dataset.totalRecords || '0');
            let reviewed = 0;
            for (let i = 1; i <= total; i++) {{
                if (getStorage('reviewed-' + i, false)) reviewed++;
            }}
            const fill = document.getElementById('progress-fill');
            const text = document.getElementById('progress-text');
            if (fill) {{
                const pct = total > 0 ? (reviewed / total * 100) : 0;
                fill.style.width = pct + '%';
            }}
            if (text) {{
                text.textContent = reviewed + ' / ' + total + ' reviewed';
            }}
        }}

        function storeOriginalValues(recordId) {{
            // Text/single-line fields
            const textFields = [
                'text_transcription', 'visual_description', 'toc_entry', 'content_warning'
            ];
            textFields.forEach(function(fieldName) {{
                const key = 'original-field-' + fieldName + '-' + recordId;
                if (getStorage(key, null) === null) {{
                    const input = document.getElementById('field-' + recordId + '-' + fieldName);
                    if (input) setStorage(key, input.value);
                }}
            }});

            // List fields
            const listFields = ['named_entities', 'geographic_entities', 'topics'];
            listFields.forEach(function(fieldName) {{
                const key = 'original-list-' + fieldName + '-' + recordId;
                if (getStorage(key, null) === null) {{
                    const container = document.getElementById(fieldName + '-' + recordId);
                    if (container) {{
                        const inputs = container.querySelectorAll('.list-item input');
                        const values = [];
                        inputs.forEach(function(inp) {{
                            const v = inp.value.trim();
                            if (v) values.push(v);
                        }});
                        setStorage(key, values);
                    }}
                }}
            }});
        }}

        function restoreState() {{
            promptForReviewerName();

            document.querySelectorAll('.record').forEach(function(record) {{
                const recordId = parseInt(record.id.replace('record-', ''));

                storeOriginalValues(recordId);

                // Restore text field edits
                const edits = getStorage('edits-' + recordId, {{}});
                for (const fieldName in edits) {{
                    if (['named_entities', 'geographic_entities', 'topics'].indexOf(fieldName) !== -1) continue;
                    const data = edits[fieldName];
                    const input = document.getElementById('field-' + recordId + '-' + fieldName);
                    if (input && data && data.value !== undefined) {{
                        input.value = data.value;
                        updateFieldStyle(recordId, fieldName, true);
                    }}
                }}

                // Restore reviewed checkbox
                const cb = document.getElementById('reviewed-' + recordId);
                if (cb) cb.checked = getStorage('reviewed-' + recordId, false);

                // Restore notes
                const notesEl = document.getElementById('notes-' + recordId);
                const savedNotes = getStorage('notes-' + recordId, '');
                if (notesEl && savedNotes) notesEl.value = savedNotes;

                // Restore term decisions
                const termDecisions = getStorage('terms-' + recordId, {{}});
                for (const termId in termDecisions) {{
                    const status = termDecisions[termId];
                    const termEl = document.getElementById('term-' + recordId + '-' + termId);
                    if (!termEl) continue;
                    termEl.classList.remove('approved', 'rejected');
                    termEl.classList.add(status);
                    const rejectBtn = termEl.querySelector('.term-btn.reject');
                    const approveBtn = termEl.querySelector('.term-btn.approve');
                    const defaultStatus = termEl.dataset.default;
                    if (status === 'rejected') {{
                        if (rejectBtn) rejectBtn.style.display = 'none';
                        if (approveBtn) {{
                            approveBtn.style.display = 'inline-block';
                            approveBtn.textContent = 'Restore';
                        }}
                    }} else if (status === 'approved') {{
                        if (defaultStatus === 'approved') {{
                            if (rejectBtn) rejectBtn.style.display = 'inline-block';
                            if (approveBtn) approveBtn.style.display = 'none';
                        }} else {{
                            if (rejectBtn) {{
                                rejectBtn.style.display = 'inline-block';
                                rejectBtn.textContent = 'Remove';
                            }}
                            if (approveBtn) approveBtn.style.display = 'none';
                        }}
                    }}
                }}

                // Restore rejected topic items
                const rejectedTopics = getStorage('rejected-topics-' + recordId, {{}});
                for (const topicKey in rejectedTopics) {{
                    const item = document.getElementById('topic-item-' + recordId + '-' + topicKey);
                    if (!item) continue;
                    item.classList.add('topic-rejected');
                    const rejectBtn = item.querySelector('.reject-topic-btn');
                    const restoreBtn = item.querySelector('.restore-topic-btn');
                    if (rejectBtn) rejectBtn.style.display = 'none';
                    if (restoreBtn) restoreBtn.style.display = 'inline-block';
                }}

                // Restore custom terms
                const customTerms = getStorage('custom-terms-' + recordId, []);
                const container = document.getElementById('custom-terms-container-' + recordId);
                if (container && customTerms.length > 0) {{
                    customTerms.forEach(function(term) {{
                        const uri = term.uri || '';
                        const uriDisplay = uri
                            ? '<a href="' + uri + '" target="_blank" class="term-uri" title="' + uri + '">URI</a>'
                            : '';
                        const sourceClass = (term.source || '').toLowerCase().replace(/[^a-z]/g, '');
                        const html =
                            '<div class="vocab-term approved" id="term-' + recordId + '-' + term.id + '">' +
                            '<span class="vocab-term-label">' + term.label + '</span>' +
                            uriDisplay +
                            '<span class="vocab-term-source ' + sourceClass + '">' + (term.source || '') + '</span>' +
                            '<div class="term-actions">' +
                            '<button class="remove-btn" onclick="removeCustomTerm(' + recordId + ', \\'' + term.id + '\\')">Remove</button>' +
                            '</div></div>';
                        container.insertAdjacentHTML('beforeend', html);
                    }});
                }}

                updateRecordStatus(recordId);
            }});

            document.querySelectorAll('textarea.field-input').forEach(autoResize);
            updateProgress();
        }}

        function exportDecisions() {{
            let reviewerName = getStorage('reviewer-name', null);
            if (!reviewerName) {{
                reviewerName = prompt('Enter your name for the export:');
                if (reviewerName && reviewerName.trim()) {{
                    setStorage('reviewer-name', reviewerName.trim());
                }} else {{
                    return;
                }}
            }}

            const total = parseInt(document.body.dataset.totalRecords || '0');
            const decisions = [];

            for (let i = 1; i <= total; i++) {{
                const edits = getStorage('edits-' + i, {{}});
                const reviewed = getStorage('reviewed-' + i, false);
                const notes = getStorage('notes-' + i, '');
                const termDecisions = getStorage('terms-' + i, {{}});
                const customTerms = getStorage('custom-terms-' + i, []);
                const recordData = getStorage('record-data-' + i, {{}});

                const rejectedTopics = getStorage('rejected-topics-' + i, {{}});
                const hasActivity = (
                    reviewed ||
                    Object.keys(edits).length > 0 ||
                    Object.keys(termDecisions).length > 0 ||
                    customTerms.length > 0 ||
                    Object.keys(rejectedTopics).length > 0
                );

                if (hasActivity) {{
                    decisions.push({{
                        record_id: i,
                        folder: recordData.folder || '',
                        page_number: recordData.page_number || i,
                        image_path: recordData.image_path || '',
                        reviewed: reviewed,
                        edits: edits,
                        term_decisions: termDecisions,
                        custom_terms: customTerms,
                        rejected_topics: rejectedTopics,
                        reviewer_notes: notes
                    }});
                }}
            }}

            if (decisions.length === 0) {{
                alert('No records have been reviewed or edited yet.');
                return;
            }}

            const exportData = {{
                export_timestamp: new Date().toISOString(),
                workflow_folder: WORKFLOW_FOLDER,
                workflow_type: WORKFLOW_TYPE,
                reviewer_name: reviewerName,
                total_records: total,
                reviewed_count: decisions.filter(function(d) {{ return d.reviewed; }}).length,
                decisions: decisions
            }};

            const safeName = reviewerName.replace(/[^\\w\\-]/g, '_');
            const safeFolder = WORKFLOW_FOLDER.replace(/[^\\w\\-]/g, '_') || 'unknown';
            const dateStr = new Date().toISOString().split('T')[0];
            const filename = safeFolder + '_' + safeName + '_' + dateStr + '.json';

            const blob = new Blob([JSON.stringify(exportData, null, 2)], {{ type: 'application/json' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            setTimeout(function() {{
                URL.revokeObjectURL(url);
                a.remove();
            }}, 100);

            alert('Exported ' + decisions.length + ' record decisions to: ' + filename +
                  '\\n\\nMove this file to the review/exports/ folder before running sa_integrate_edits.py');
        }}

        // --- In-place image zoom/pan ---
        (function() {{
            const MIN_SCALE = 0.5;
            const MAX_SCALE = 8;
            const ZOOM_STEP = 0.25;
            const states = new WeakMap();  // container el -> {{scale, tx, ty}}

            function getState(container) {{
                if (!states.has(container)) states.set(container, {{ scale: 1, tx: 0, ty: 0 }});
                return states.get(container);
            }}

            function apply(container) {{
                const img = container.querySelector('img');
                if (!img) return;
                const s = getState(container);
                img.style.transform = 'translate(' + s.tx + 'px,' + s.ty + 'px) scale(' + s.scale + ')';
            }}

            // Called by +/- buttons (button is inside container)
            window.zoomImgBtn = function(btn, delta) {{
                const container = btn.closest('.image-container');
                if (!container) return;
                const s = getState(container);
                s.scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, s.scale + delta));
                apply(container);
            }};

            // Called by reset button
            window.resetImgZoom = function(btn) {{
                const container = btn.closest('.image-container');
                if (!container) return;
                const s = getState(container);
                s.scale = 1; s.tx = 0; s.ty = 0;
                apply(container);
            }};

            document.addEventListener('DOMContentLoaded', function() {{
                let activeContainer = null;   // currently being dragged
                let hoveredContainer = null;  // cursor is over this container
                let dragStartX, dragStartY, dragTx, dragTy;

                document.querySelectorAll('.image-container').forEach(function(container) {{
                    // Track which container the cursor is over for keyboard zoom
                    container.addEventListener('mouseenter', function() {{ hoveredContainer = container; }});
                    container.addEventListener('mouseleave', function() {{ hoveredContainer = null; }});

                    // Mousedown starts a pan
                    container.addEventListener('mousedown', function(e) {{
                        if (e.target.classList.contains('zoom-btn')) return;
                        if (e.button !== 0) return;
                        e.preventDefault();
                        const s = getState(container);
                        activeContainer = container;
                        dragStartX = e.clientX;
                        dragStartY = e.clientY;
                        dragTx = s.tx;
                        dragTy = s.ty;
                        container.querySelector('img').classList.add('panning');
                    }});
                }});

                document.addEventListener('mousemove', function(e) {{
                    if (!activeContainer) return;
                    const s = getState(activeContainer);
                    s.tx = dragTx + (e.clientX - dragStartX);
                    s.ty = dragTy + (e.clientY - dragStartY);
                    apply(activeContainer);
                }});

                document.addEventListener('mouseup', function() {{
                    if (activeContainer) {{
                        const img = activeContainer.querySelector('img');
                        if (img) img.classList.remove('panning');
                        activeContainer = null;
                    }}
                }});

                // Keyboard zoom: +/= in, - out, 0 reset -- only when hovering over an image
                document.addEventListener('keydown', function(e) {{
                    if (!hoveredContainer) return;
                    // Ignore keystrokes while typing in a text field
                    const tag = document.activeElement && document.activeElement.tagName;
                    if (tag === 'INPUT' || tag === 'TEXTAREA') return;
                    if (e.key === '+' || e.key === '=') {{
                        e.preventDefault();
                        const s = getState(hoveredContainer);
                        s.scale = Math.min(MAX_SCALE, s.scale + ZOOM_STEP);
                        apply(hoveredContainer);
                    }} else if (e.key === '-') {{
                        e.preventDefault();
                        const s = getState(hoveredContainer);
                        s.scale = Math.max(MIN_SCALE, s.scale - ZOOM_STEP);
                        apply(hoveredContainer);
                    }} else if (e.key === '0') {{
                        e.preventDefault();
                        const s = getState(hoveredContainer);
                        s.scale = 1; s.tx = 0; s.ty = 0;
                        apply(hoveredContainer);
                    }}
                }});
            }});
        }})();

        document.addEventListener('DOMContentLoaded', restoreState);
        """

    # ------------------------------------------------------------------
    # Record HTML generation
    # ------------------------------------------------------------------

    def generate_list_field_html(self, items, record_id, field_name, placeholder="Enter value..."):
        """Generate HTML for an editable list field."""
        html = '<div id="' + field_name + '-' + str(record_id) + '" class="list-field">'

        if items:
            for item in items:
                value = self.escape_html(item) if isinstance(item, str) else self.escape_html(str(item))
                html += (
                    '<div class="list-item">'
                    '<input type="text" class="field-input" value="' + value + '"'
                    ' placeholder="' + placeholder + '"'
                    ' onchange="saveListField(' + str(record_id) + ', \'' + field_name + '\')">'
                    '<button class="remove-btn" onclick="this.parentElement.remove();'
                    ' saveListField(' + str(record_id) + ', \'' + field_name + '\')">Remove</button>'
                    '</div>'
                )

        html += '</div>'
        html += (
            '<button class="add-btn"'
            ' onclick="addListItem(' + str(record_id) + ', \'' + field_name + '\')">+ Add Item</button>'
        )
        return html

    def generate_topics_list_html(self, items, record_id):
        """Generate HTML for the topics list with per-item Reject buttons for cascade."""
        html = '<div id="topics-' + str(record_id) + '" class="list-field">'
        for idx, item in enumerate(items):
            value = self.escape_html(item) if isinstance(item, str) else self.escape_html(str(item))
            tkey = 't' + str(idx)
            html += (
                '<div class="list-item topic-item"'
                ' id="topic-item-' + str(record_id) + '-' + tkey + '"'
                ' data-topic-key="' + tkey + '">'
                '<input type="text" class="field-input" value="' + value + '"'
                ' placeholder="Topic..."'
                ' onchange="saveListField(' + str(record_id) + ', \'topics\')">'
                '<button class="reject-topic-btn"'
                ' onclick="rejectTopicCascade(' + str(record_id) + ', \'' + tkey + '\')">'
                'Reject</button>'
                '<button class="restore-topic-btn" style="display:none;"'
                ' onclick="restoreTopicItem(' + str(record_id) + ', \'' + tkey + '\')">'
                'Restore</button>'
                '<button class="remove-btn"'
                ' onclick="this.parentElement.remove();'
                ' saveListField(' + str(record_id) + ', \'topics\')">Remove</button>'
                '</div>'
            )
        html += '</div>'
        html += (
            '<button class="add-btn"'
            ' onclick="addListItem(' + str(record_id) + ', \'topics\')">+ Add Item</button>'
        )
        return html

    def generate_vocabulary_section(self, analysis, record_id):
        """Generate HTML for the three vocabulary review sections."""
        html = '<div class="vocab-section">'

        # ----------------------------------------
        # Section 1: Subject Headings
        # ----------------------------------------
        html += '<div class="section-header">Subject Headings</div>'

        final_selected_terms = analysis.get('final_selected_terms', [])
        topics_list = analysis.get('topics', [])
        vsr = analysis.get('vocabulary_search_results', {})

        # Build a mapping: term_id -> set of topic keys (t0, t1, ...) that produced it.
        # Match final_selected_terms against vocabulary_search_results by (label, source) pair.
        term_id_to_topic_keys = {}
        for t_idx, topic_label in enumerate(topics_list):
            tkey = 't' + str(t_idx)
            topic_matches = vsr.get(topic_label, [])
            match_keys = set()
            for m in topic_matches:
                lbl = m.get('label', '').lower()
                src = m.get('source', '')
                if lbl:
                    match_keys.add((lbl, src))
            for idx, term in enumerate(final_selected_terms):
                pair = (term.get('label', '').lower(), term.get('source', ''))
                if pair in match_keys:
                    term_id_to_topic_keys.setdefault('selected-' + str(idx), set()).add(tkey)

        # Map topic label -> tkey for alt-term lookup
        topic_label_to_key = {lbl: 't' + str(i) for i, lbl in enumerate(topics_list)}

        # Selected subject headings (auto-accepted, opt-out)
        if final_selected_terms:
            html += (
                '<div class="vocab-group">'
                '<div class="vocab-group-title">Selected Subject Headings'
                ' <span style="font-weight:normal;font-size:11px;color:#666;">'
                '(auto-accepted - click Reject to remove)</span></div>'
            )
            for idx, term in enumerate(final_selected_terms):
                term_id = 'selected-' + str(idx)
                label = self.escape_html(term.get('label', ''))
                uri = term.get('uri', '')
                source = term.get('source', 'Unknown')
                source_class = source.lower().replace(' ', '').replace('getty', '')
                uri_html = ('<a href="' + uri + '" target="_blank" class="term-uri" title="' + uri + '">URI</a>'
                            if uri else '')
                topic_keys = term_id_to_topic_keys.get(term_id, set())
                for_topics_attr = (' data-for-topics="' + ' '.join(sorted(topic_keys)) + '"'
                                   if topic_keys else '')
                html += (
                    '<div class="vocab-term approved"'
                    ' id="term-' + str(record_id) + '-' + term_id + '"'
                    ' data-default="approved"'
                    + for_topics_attr + '>'
                    '<span class="vocab-term-label">' + label + '</span>'
                    + uri_html +
                    '<span class="vocab-term-source ' + source_class + '">' + source + '</span>'
                    '<div class="term-actions">'
                    '<button class="term-btn reject"'
                    ' onclick="setTermStatus(' + str(record_id) + ', \'' + term_id + '\', \'rejected\')">'
                    'Reject</button>'
                    '<button class="term-btn approve"'
                    ' onclick="setTermStatus(' + str(record_id) + ', \'' + term_id + '\', \'approved\')"'
                    ' style="display:none;">Restore</button>'
                    '</div></div>'
                )
            html += '</div>'
        else:
            html += '<div class="vocab-group"><p style="color:#999;font-style:italic;">No subject headings selected.</p></div>'

        # Alternative subject headings from vocabulary search results (opt-in)
        vocab_results = analysis.get('vocabulary_search_results', {})
        if vocab_results:
            selected_pairs = set()
            for t in final_selected_terms:
                selected_pairs.add((t.get('label', '').lower(), t.get('source', '')))

            # Collect alternatives grouped by source
            sources_order = ['LCSH', 'FAST', 'Getty AAT', 'Getty TGN']
            sources_map = {s: [] for s in sources_order}

            for orig_topic, matches in vocab_results.items():
                for match in matches:
                    source = match.get('source', 'Unknown')
                    label = match.get('label', '')
                    if (label.lower(), source) in selected_pairs:
                        continue
                    if source in sources_map:
                        sources_map[source].append({
                            'orig_topic': orig_topic,
                            'label': label,
                            'uri': match.get('uri', ''),
                            'source': source
                        })

            has_alternatives = any(len(v) > 0 for v in sources_map.values())
            if has_alternatives:
                html += (
                    '<div class="vocab-group" style="border:1px dashed #ccc;margin-top:12px;">'
                    '<div class="vocab-group-title">Alternative Subject Headings'
                    ' <span style="font-weight:normal;font-size:11px;color:#666;">'
                    '(not selected - click Add to include)</span></div>'
                )
                alt_idx = 0
                for source_name in sources_order:
                    terms = sources_map[source_name]
                    if not terms:
                        continue
                    source_class = source_name.lower().replace(' ', '').replace('getty', '')
                    html += (
                        '<div style="margin:8px 0;">'
                        '<div style="font-weight:600;font-size:12px;color:#555;margin-bottom:4px;">'
                        + source_name + ' (' + str(len(terms)) + ' options)</div>'
                    )
                    for term in terms:
                        term_id = 'alt-' + str(alt_idx)
                        label = self.escape_html(term['label'])
                        uri = term.get('uri', '')
                        uri_html = ('<a href="' + uri + '" target="_blank" class="term-uri" title="' + uri + '">URI</a>'
                                    if uri else '')
                        orig_tkey = topic_label_to_key.get(term.get('orig_topic', ''), '')
                        for_topics_attr = (' data-for-topics="' + orig_tkey + '"'
                                           if orig_tkey else '')
                        html += (
                            '<div class="vocab-term"'
                            ' id="term-' + str(record_id) + '-' + term_id + '"'
                            ' data-default="none"'
                            + for_topics_attr + '>'
                            '<span class="vocab-term-label">' + label + '</span>'
                            + uri_html +
                            '<span class="vocab-term-source ' + source_class + '">' + source_name + '</span>'
                            '<div class="term-actions">'
                            '<button class="term-btn approve"'
                            ' onclick="setTermStatus(' + str(record_id) + ', \'' + term_id + '\', \'approved\')">Add</button>'
                            '<button class="term-btn reject"'
                            ' onclick="setTermStatus(' + str(record_id) + ', \'' + term_id + '\', \'rejected\')"'
                            ' style="display:none;">Remove</button>'
                            '</div></div>'
                        )
                        alt_idx += 1
                    html += '</div>'
                html += '</div>'

        # Custom subject headings
        html += (
            '<div class="vocab-group" style="margin-top:12px;">'
            '<div class="vocab-group-title">Add Custom Subject Heading</div>'
            '<div id="custom-terms-container-' + str(record_id) + '"></div>'
            '<div class="add-term-form">'
            '<div>'
            '<label>Term Label (required)</label>'
            '<input type="text" id="new-term-label-' + str(record_id) + '" placeholder="e.g., Reinforced concrete construction">'
            '</div>'
            '<div>'
            '<label>URI / Permalink (optional)</label>'
            '<input type="text" id="new-term-uri-' + str(record_id) + '" placeholder="e.g., http://id.loc.gov/authorities/subjects/sh85113031">'
            '</div>'
            '<div style="display:flex;gap:10px;align-items:flex-end;">'
            '<div>'
            '<label>Source</label>'
            '<select id="new-term-source-' + str(record_id) + '">'
            '<option value="LCSH">LCSH</option>'
            '<option value="FAST">FAST</option>'
            '<option value="Getty AAT">Getty AAT</option>'
            '<option value="Getty TGN">Getty TGN</option>'
            '<option value="Manual">Manual</option>'
            '</select>'
            '</div>'
            '<button class="add-btn" onclick="addCustomTerm(' + str(record_id) + ')" style="height:36px;">Add Heading</button>'
            '</div>'
            '</div>'
            '</div>'
        )

        # ----------------------------------------
        # Section 2: Geographic Headings
        # ----------------------------------------
        html += '<div class="section-header" style="margin-top:22px;">Geographic Headings</div>'

        geo_vocab = analysis.get('geographic_vocabulary_search_results', {})
        if geo_vocab:
            html += (
                '<div class="vocab-group">'
                '<div class="vocab-group-title">Geographic Vocabulary Terms'
                ' <span style="font-weight:normal;font-size:11px;color:#666;">'
                '(opt-in - click Add to include)</span></div>'
            )
            geo_idx = 0
            for source_entity, matches in geo_vocab.items():
                for_label = source_entity[:40] + ('...' if len(source_entity) > 40 else '')
                for match in matches:
                    term_id = 'geo-' + str(geo_idx)
                    label = self.escape_html(match.get('label', ''))
                    uri = match.get('uri', '')
                    source = match.get('source', 'FAST Geographic')
                    source_class = source.lower().replace(' ', '').replace('getty', '')
                    uri_html = ('<a href="' + uri + '" target="_blank" class="term-uri" title="' + uri + '">URI</a>'
                                if uri else '')
                    html += (
                        '<div class="vocab-term"'
                        ' id="term-' + str(record_id) + '-' + term_id + '"'
                        ' data-default="none">'
                        '<span class="vocab-term-label">' + label + '</span>'
                        '<span class="for-badge" title="For: ' + self.escape_html(source_entity) + '">'
                        'for: ' + self.escape_html(for_label) + '</span>'
                        + uri_html +
                        '<span class="vocab-term-source ' + source_class + '">' + source + '</span>'
                        '<div class="term-actions">'
                        '<button class="term-btn approve"'
                        ' onclick="setTermStatus(' + str(record_id) + ', \'' + term_id + '\', \'approved\')">Add</button>'
                        '<button class="term-btn reject"'
                        ' onclick="setTermStatus(' + str(record_id) + ', \'' + term_id + '\', \'rejected\')"'
                        ' style="display:none;">Remove</button>'
                        '</div></div>'
                    )
                    geo_idx += 1
            html += '</div>'
        else:
            html += '<div class="vocab-group"><p style="color:#999;font-style:italic;">No geographic vocabulary terms found.</p></div>'

        # ----------------------------------------
        # Section 3: Chronological Terms
        # ----------------------------------------
        html += '<div class="section-header" style="margin-top:22px;">Chronological Terms</div>'

        chrono_terms = analysis.get('chronological_vocabulary_terms', [])
        if chrono_terms:
            html += (
                '<div class="vocab-group">'
                '<div class="vocab-group-title">Chronological Vocabulary Terms'
                ' <span style="font-weight:normal;font-size:11px;color:#666;">'
                '(auto-accepted - click Reject to remove)</span></div>'
            )
            for idx, term in enumerate(chrono_terms):
                term_id = 'chrono-' + str(idx)
                label = self.escape_html(term.get('label', ''))
                uri = term.get('uri', '')
                source = term.get('source', 'LCSH')
                source_class = source.lower().replace(' ', '').replace('getty', '')
                uri_html = ('<a href="' + uri + '" target="_blank" class="term-uri" title="' + uri + '">URI</a>'
                            if uri else '')
                html += (
                    '<div class="vocab-term approved"'
                    ' id="term-' + str(record_id) + '-' + term_id + '"'
                    ' data-default="approved">'
                    '<span class="vocab-term-label">' + label + '</span>'
                    + uri_html +
                    '<span class="vocab-term-source ' + source_class + '">' + source + '</span>'
                    '<div class="term-actions">'
                    '<button class="term-btn reject"'
                    ' onclick="setTermStatus(' + str(record_id) + ', \'' + term_id + '\', \'rejected\')">'
                    'Reject</button>'
                    '<button class="term-btn approve"'
                    ' onclick="setTermStatus(' + str(record_id) + ', \'' + term_id + '\', \'approved\')"'
                    ' style="display:none;">Restore</button>'
                    '</div></div>'
                )
            html += '</div>'
        else:
            html += '<div class="vocab-group"><p style="color:#999;font-style:italic;">No chronological terms found.</p></div>'

        html += '</div>'
        return html

    def generate_record_html(self, record, global_id):
        """Generate HTML for a single page record."""
        analysis = record.get('analysis', {})
        folder = record.get('folder', '')
        page_number = record.get('page_number', global_id)
        image_path = record.get('image_path', '')
        image_filename = os.path.basename(image_path) if image_path else ''

        # Store record metadata in localStorage for export
        record_data_js = (
            '<script>'
            'setStorage(\'record-data-' + str(global_id) + '\', {'
            'folder: \'' + self.escape_js_string(folder) + '\','
            'page_number: ' + str(page_number) + ','
            'image_path: \'' + self.escape_js_string(image_path) + '\''
            '});'
            '</script>'
        )

        header_title = 'Record ' + str(global_id) + ': Folder ' + self.escape_html(folder) + ', Page ' + str(page_number)

        html = (
            '<div class="record" id="record-' + str(global_id) + '">'
            + record_data_js +
            '<div class="record-header">'
            '<div class="record-title">' + header_title + '</div>'
            '</div>'
            '<div class="content-grid">'
        )

        # Image column
        html += '<div>'
        if image_filename:
            html += (
                '<div class="image-container">'
                '<div class="zoom-controls">'
                '<button class="zoom-btn" onclick="zoomImgBtn(this, 0.25)" title="Zoom in">+</button>'
                '<button class="zoom-btn" onclick="zoomImgBtn(this, -0.25)" title="Zoom out">-</button>'
                '<button class="zoom-btn" onclick="resetImgZoom(this)" title="Reset zoom">&#x2715;</button>'
                '</div>'
                '<img src="images/' + image_filename + '"'
                ' alt="Page ' + str(page_number) + '"'
                ' onerror="this.style.display=\'none\';this.nextElementSibling.style.display=\'block\';">'
                '<div style="display:none;padding:20px;color:#999;font-size:13px;">Image not available</div>'
                '</div>'
                '<div class="zoom-hint">Hover image, then  +/- to zoom  |  0 to reset  |  Drag to pan</div>'
                '<div class="image-filename">' + self.escape_html(image_filename) + '</div>'
            )
        else:
            html += '<div class="image-container"><div style="padding:30px;color:#999;font-size:13px;">No image path</div></div>'
        html += '</div>'

        # Metadata column
        html += '<div><div class="section-header">Metadata Fields</div>'

        # Text transcription (large textarea)
        html += self._text_field_html(
            global_id, 'text_transcription', 'Text Transcription',
            analysis.get('text_transcription', ''), field_type='textarea-large'
        )

        # Visual description (large textarea)
        html += self._text_field_html(
            global_id, 'visual_description', 'Visual Description',
            analysis.get('visual_description', ''), field_type='textarea-large'
        )

        # TOC entry (textarea)
        html += self._text_field_html(
            global_id, 'toc_entry', 'TOC Entry',
            analysis.get('toc_entry', ''), field_type='textarea'
        )

        # Content warning (single-line)
        html += self._text_field_html(
            global_id, 'content_warning', 'Content Warning',
            analysis.get('content_warning', ''), field_type='text'
        )

        # Named entities
        html += '<div class="field-group"><div class="field-label">Named Entities</div>'
        html += self.generate_list_field_html(
            analysis.get('named_entities', []), global_id, 'named_entities', 'Named entity...'
        )
        html += '</div>'

        # Geographic entities
        html += '<div class="field-group"><div class="field-label">Geographic Entities</div>'
        html += self.generate_list_field_html(
            analysis.get('geographic_entities', []), global_id, 'geographic_entities', 'Geographic entity...'
        )
        html += '</div>'

        # Topics (with cascade-reject buttons)
        html += '<div class="field-group"><div class="field-label">Topics</div>'
        html += self.generate_topics_list_html(analysis.get('topics', []), global_id)
        html += '</div>'

        # Close metadata column and content grid
        html += '</div></div>'

        # Vocabulary section (full width)
        html += self.generate_vocabulary_section(analysis, global_id)

        # Reviewer notes and reviewed checkbox
        html += (
            '<div class="notes-section">'
            '<div class="field-label">Reviewer Notes</div>'
            '<textarea class="field-input" id="notes-' + str(global_id) + '"'
            ' placeholder="Add notes about this record..."'
            ' oninput="autoResize(this); saveNotes(' + str(global_id) + ', this.value)"></textarea>'
            '<label class="reviewed-checkbox">'
            '<input type="checkbox" id="reviewed-' + str(global_id) + '"'
            ' onchange="setReviewed(' + str(global_id) + ', this.checked)">'
            'Mark as reviewed'
            '</label>'
            '</div>'
            '</div>'
        )
        return html

    def _text_field_html(self, record_id, field_name, label, value, field_type='text'):
        """Generate HTML for a single text or textarea field."""
        escaped_value = self.escape_html(value)
        html = (
            '<div class="field-group">'
            '<div class="field-label">' + label + ' <span class="edit-indicator"></span></div>'
        )
        if field_type in ('textarea-large', 'textarea'):
            html += (
                '<textarea class="field-input"'
                ' id="field-' + str(record_id) + '-' + field_name + '"'
                ' oninput="autoResize(this); saveFieldValue(' + str(record_id) + ', \'' + field_name + '\', this.value)">'
                + escaped_value + '</textarea>'
            )
        else:
            html += (
                '<input type="text" class="field-input"'
                ' id="field-' + str(record_id) + '-' + field_name + '"'
                ' value="' + escaped_value + '"'
                ' onchange="saveFieldValue(' + str(record_id) + ', \'' + field_name + '\', this.value)">'
            )
        html += '</div>'
        return html

    # ------------------------------------------------------------------
    # Page generation
    # ------------------------------------------------------------------

    def _build_page_header(self, title, subtitle=''):
        """Build the HTML header div."""
        sub_html = '<p>' + self.escape_html(subtitle) + '</p>' if subtitle else ''
        return (
            '<div class="header">'
            '<h1>' + self.escape_html(title) + '</h1>'
            + sub_html +
            '<p>Workflow folder: ' + self.escape_html(self.folder_name) + '</p>'
            '<p>Workflow type: ' + self.escape_html(self.workflow_type) + '</p>'
            '</div>'
        )

    def _build_progress_bar(self, total_records):
        """Build the progress bar + export button HTML."""
        return (
            '<div class="export-section">'
            '<div style="display:flex;justify-content:space-between;align-items:center;">'
            '<div>'
            '<strong>Progress:</strong> <span id="progress-text">Loading...</span>'
            '</div>'
            '<button class="export-btn" onclick="exportDecisions()">Export All Decisions to JSON</button>'
            '</div>'
            '<div class="progress-bar" style="margin-top:10px;">'
            '<div class="progress-fill" id="progress-fill" style="width:0%"></div>'
            '</div>'
            '</div>'
        )

    def _build_navigation(self, page_num, total_pages):
        """Build the navigation bar HTML."""
        prev_html = (
            '<a href="review-page-' + str(page_num - 1) + '.html" class="nav-btn">Previous</a>'
            if page_num > 1
            else '<span class="nav-btn disabled">Previous</span>'
        )
        next_html = (
            '<a href="review-page-' + str(page_num + 1) + '.html" class="nav-btn">Next</a>'
            if page_num < total_pages
            else '<span class="nav-btn disabled">Next</span>'
        )
        return (
            '<div class="navigation">'
            '<a href="review-index.html" class="nav-btn">Index</a>'
            + prev_html +
            '<span style="margin:0 15px;font-weight:bold;">Page ' + str(page_num) + ' of ' + str(total_pages) + '</span>'
            + next_html +
            '</div>'
        )

    def _group_by_folder(self):
        """Return an ordered list of (folder_name, [records]) pairs."""
        seen = {}
        order = []
        for item in self.data_items:
            folder = item.get('folder', '')
            if folder not in seen:
                seen[folder] = []
                order.append(folder)
            seen[folder].append(item)
        return [(f, seen[f]) for f in order]

    def create_index_page(self, issue_groups, total_records):
        """Create the index page with one link per issue folder."""
        total_issues = len(issue_groups)

        page_links_html = ''
        for page_num, (folder_name, records) in enumerate(issue_groups, start=1):
            page_links_html += (
                '<a href="review-page-' + str(page_num) + '.html" class="page-link">'
                'Issue: ' + self.escape_html(folder_name) + '<br>'
                '<small>' + str(len(records)) + ' page' + ('s' if len(records) != 1 else '') + '</small>'
                '</a>'
            )

        html = (
            '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            '<title>SA Metadata Review - Index</title>\n'
            '<style>' + self.get_css_styles() + '</style>\n'
            '</head>\n'
            '<body data-total-records="' + str(total_records) + '">\n'
            + self._build_page_header(
                'Southern Architect Metadata Review',
                str(total_records) + ' page record' + ('s' if total_records != 1 else '')
                + ' across ' + str(total_issues) + ' issue' + ('s' if total_issues != 1 else '')
            )
            + self._build_progress_bar(total_records)
            + '<div class="summary-grid">'
            '<div class="summary-card">'
            '<div class="number">' + str(total_records) + '</div>'
            '<div class="label">Total Records</div>'
            '</div>'
            '<div class="summary-card">'
            '<div class="number">' + str(total_issues) + '</div>'
            '<div class="label">Issues</div>'
            '</div>'
            '</div>'
            '<div class="export-section">'
            '<h3 style="margin-top:0;">Issues</h3>'
            '<div class="page-links">' + page_links_html + '</div>'
            '</div>'
            '<script>' + self.get_javascript() + '</script>\n'
            '</body>\n</html>'
        )

        index_path = os.path.join(self.review_folder, "review-index.html")
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(html)

        print("Created review-index.html")
        return index_path

    def create_review_page(self, page_num, folder_name, page_records, start_idx, total_pages, total_records):
        """Create a single issue review page."""
        records_html = ''
        for i, record in enumerate(page_records):
            global_id = start_idx + i + 1
            records_html += self.generate_record_html(record, global_id)

        nav = self._build_navigation(page_num, total_pages)

        html = (
            '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            '<title>SA Review - ' + self.escape_html(folder_name) + '</title>\n'
            '<style>' + self.get_css_styles() + '</style>\n'
            '</head>\n'
            '<body data-total-records="' + str(total_records) + '">\n'
            + self._build_page_header(
                'Issue: ' + folder_name,
                str(len(page_records)) + ' page record' + ('s' if len(page_records) != 1 else '')
                + '  |  Issue ' + str(page_num) + ' of ' + str(total_pages)
            )
            + nav
            + self._build_progress_bar(total_records)
            + records_html
            + nav
            + '\n<script>' + self.get_javascript() + '</script>\n'
            '</body>\n</html>'
        )

        page_path = os.path.join(self.review_folder, 'review-page-' + str(page_num) + '.html')
        with open(page_path, 'w', encoding='utf-8') as f:
            f.write(html)

        return page_path

    def create_review_pages(self):
        """Create one review page per issue folder and the index page."""
        issue_groups = self._group_by_folder()
        total_issues = len(issue_groups)
        total_records = len(self.data_items)

        print("Creating " + str(total_issues) + " issue page(s) for " + str(total_records) + " records...")

        self.create_index_page(issue_groups, total_records)

        global_offset = 0
        for page_num, (folder_name, records) in enumerate(issue_groups, start=1):
            self.create_review_page(
                page_num, folder_name, records, global_offset, total_issues, total_records
            )
            print("  Issue " + str(page_num) + "/" + str(total_issues)
                  + " (" + folder_name + ", " + str(len(records)) + " records)")
            global_offset += len(records)

        return total_issues

    def run(self):
        """Run the full HTML review generation pipeline."""
        print("")
        print("=" * 60)
        print("SA HTML Review Interface Generator")
        print("=" * 60)
        print("Folder: " + self.folder_name)

        if not self.load_json_data():
            return False

        if not self.data_items:
            print("No page records found in workflow JSON.")
            return False

        if not self.create_review_folder():
            return False

        self.copy_images()

        total_pages = self.create_review_pages()

        index_path = os.path.join(self.review_folder, "review-index.html")
        print("")
        print("=" * 60)
        print("HTML review interface created successfully.")
        print("=" * 60)
        print("Total records: " + str(len(self.data_items)))
        print("Review pages:  " + str(total_pages))
        print("")
        print("Open this file to begin reviewing:")
        print("  " + index_path)
        print("")
        print("After reviewing, export decisions and move the JSON file to:")
        print("  " + os.path.join(self.review_folder, "exports"))
        print("Then run sa_integrate_edits.py to apply changes.")
        print("=" * 60)

        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate an HTML review interface for Southern Architect metadata.'
    )
    parser.add_argument(
        '--folder', '-f',
        help='Path to the output folder. Defaults to the newest folder in output_folders/.'
    )
    parser.add_argument(
        '--records-per-page', '-n',
        type=int,
        default=10,
        dest='records_per_page',
        help='Number of records per HTML page (default: 10).'
    )
    args = parser.parse_args()

    if args.folder:
        folder_path = args.folder
        if not os.path.isdir(folder_path):
            print("Error: Folder not found: " + folder_path)
            return 1
    else:
        base_output_dir = os.path.join(script_dir, FOLDER_CONFIG['output_dir'])
        folder_path = find_newest_folder(base_output_dir)
        if not folder_path:
            print("No output folders found in: " + base_output_dir)
            print("Run the SA workflow steps first to generate metadata.")
            return 1
        print("Auto-selected newest folder: " + os.path.basename(folder_path))

    builder = SAHTMLReviewBuilder(folder_path, records_per_page=args.records_per_page)
    success = builder.run()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
