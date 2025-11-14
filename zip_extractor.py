"""Utility to extract first N folders from a zip file."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import List, Optional, Set


def extract_first_n_folders(
    zip_path: Path,
    output_dir: Path,
    max_folders: Optional[int] = None,
    zip_internal_path: str = "DEV/",
) -> Path:
    """Extract folders from a zip file."""
    print(f"Opening zip file: {zip_path}")
    if max_folders is None:
        print(f"Extracting all folders from {zip_internal_path}")
    else:
        print(f"Extracting first {max_folders} folders from {zip_internal_path}")
    
    # Normalize the internal path
    if not zip_internal_path.endswith("/"):
        zip_internal_path = zip_internal_path + "/"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_dev_path = output_dir / "DEV"
    extracted_dev_path.mkdir(parents=True, exist_ok=True)
    
    # Get all folder names from the zip
    folder_names: Set[str] = set()
    
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # Find all folders in the zip
        for name in zip_ref.namelist():
            # Check if this file is under the internal path
            if name.startswith(zip_internal_path):
                # Remove the internal path prefix
                relative_path = name[len(zip_internal_path):]
                # Get the first folder name (domain folder)
                parts = relative_path.split("/")
                if len(parts) > 0 and parts[0]:
                    folder_names.add(parts[0])
        
        if not folder_names:
            print(f"WARNING: No folders found in {zip_internal_path} within the zip file")
            return extracted_dev_path
        
        # Sort folder names alphabetically
        sorted_folders = sorted(folder_names)
        
        # Select folders based on max_folders limit
        if max_folders is None:
            selected_folders = sorted_folders  # Extract all folders
            print(f"Found {len(folder_names)} folders in zip file")
            print(f"Extracting all {len(selected_folders)} folders (alphabetically sorted)")
        else:
            selected_folders = sorted_folders[:max_folders]
            print(f"Found {len(folder_names)} folders in zip file")
            print(f"Extracting first {len(selected_folders)} folders (alphabetically sorted)")
            if len(folder_names) > max_folders:
                print(f"  (Skipping {len(folder_names) - max_folders} remaining folders)")
        
        # Create a set for fast lookup
        selected_folders_set = set(selected_folders)
        
        # Extract only files from selected folders
        files_extracted = 0
        folders_created = set()
        
        for name in zip_ref.namelist():
            if name.startswith(zip_internal_path):
                relative_path = name[len(zip_internal_path):]
                parts = relative_path.split("/")
                
                # Check if this file belongs to a selected folder
                if len(parts) > 0 and parts[0] in selected_folders_set:
                    # Construct the output path using Path for cross-platform compatibility
                    # Join path parts using Path's join method
                    output_path = extracted_dev_path
                    for part in parts:
                        output_path = output_path / part
                    
                    # Skip if it's a directory entry (ends with /)
                    if name.endswith("/"):
                        output_path.mkdir(parents=True, exist_ok=True)
                        folders_created.add(output_path)
                        continue
                    
                    # Create parent directories if needed
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    folders_created.add(output_path.parent)
                    
                    # Extract the file
                    try:
                        with zip_ref.open(name) as source:
                            with open(output_path, "wb") as target:
                                target.write(source.read())
                        files_extracted += 1
                        
                        if files_extracted % 100 == 0:
                            print(f"  Extracted {files_extracted} files...")
                    except Exception as e:
                        print(f"  WARNING: Failed to extract {name}: {e}")
                        continue
        
        print(f"Extraction complete: {files_extracted} files extracted to {extracted_dev_path}")
        print(f"Created {len(folders_created)} directories")
    
    return extracted_dev_path


def is_zip_file(path: Path) -> bool:
    """Check if a path is a zip file."""
    return path.is_file() and path.suffix.lower() == ".zip"

