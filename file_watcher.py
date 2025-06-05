import os
import time
from pathlib import Path

class FileWatcher:
    def __init__(self, directory):
        """
        Initialize the file watcher to monitor changes in CSV files
        
        Args:
            directory (Path): Directory to monitor
        """
        self.directory = Path(directory)
        self.file_stats = {}
        self.update_file_stats()
    
    def update_file_stats(self):
        """Update the file stats for all CSV files in the directory"""
        self.file_stats = {}
        for file_path in self.directory.glob('*.csv'):
            try:
                self.file_stats[file_path.name] = {
                    'mtime': os.path.getmtime(file_path),
                    'size': os.path.getsize(file_path)
                }
            except (FileNotFoundError, PermissionError) as e:
                # Handle file access errors
                print(f"Error accessing {file_path}: {e}")
    
    def check_for_changes(self):
        """
        Check if any CSV files have changed since last check
        
        Returns:
            bool: True if any changes detected, False otherwise
        """
        changes_detected = False
        current_stats = {}
        
        # Get current stats
        for file_path in self.directory.glob('*.csv'):
            try:
                current_stats[file_path.name] = {
                    'mtime': os.path.getmtime(file_path),
                    'size': os.path.getsize(file_path)
                }
            except (FileNotFoundError, PermissionError):
                # Skip files with access errors
                continue
        
        # Check for new files or changes
        for filename, stats in current_stats.items():
            if filename not in self.file_stats:
                # New file
                changes_detected = True
            elif (stats['mtime'] != self.file_stats[filename]['mtime'] or 
                  stats['size'] != self.file_stats[filename]['size']):
                # Modified file
                changes_detected = True
        
        # Check for deleted files
        for filename in self.file_stats:
            if filename not in current_stats:
                # Deleted file
                changes_detected = True
        
        # Update stats if changes detected
        if changes_detected:
            self.file_stats = current_stats
        
        return changes_detected