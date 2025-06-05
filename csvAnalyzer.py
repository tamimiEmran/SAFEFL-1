import os
import pandas as pd
import glob
from datetime import datetime


def analyze_csv_directory(directory_path, output_file=None):
    """
    Analyzes all CSV files in a directory and creates a basic report.
    
    Parameters:
    directory_path (str): Path to the directory containing CSV files
    output_file (str, optional): Path to save the report. If None, prints to console.
    
    Returns:
    str: The full report text
    """
    # Ensure the directory exists
    if not os.path.isdir(directory_path):
        return f"Error: Directory '{directory_path}' does not exist."
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    
    if not csv_files:
        return f"No CSV files found in directory: {directory_path}"
    
    # Start the report
    report = []
    report.append(f"CSV Analysis Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Directory: {os.path.abspath(directory_path)}")
    report.append(f"Total CSV files found: {len(csv_files)}")
    report.append("=" * 80)
    
    # Analyze each CSV file
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) / 1024  # Size in KB
        
        report.append(f"\nFile: {file_name}")
        report.append(f"Size: {file_size:.2f} KB")
        report.append("-" * 50)
        
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Basic file info
            report.append(f"Rows: {len(df)}")
            report.append(f"Columns: {len(df.columns)}")
            report.append(f"Column names: {', '.join(df.columns.tolist())}")
            
            # Data types
            report.append("\nData Types:")
            for col, dtype in df.dtypes.items():
                report.append(f"  - {col}: {dtype}")
            
            # Missing values
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                report.append("\nMissing Values:")
                for col, count in missing_values.items():
                    if count > 0:
                        report.append(f"  - {col}: {count} ({(count/len(df)*100):.2f}%)")
            else:
                report.append("\nNo missing values found.")
            
            # Summary statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if not numeric_cols.empty:
                report.append("\nNumeric Column Statistics:")
                for col in numeric_cols:
                    stats = df[col].describe()
                    report.append(f"  {col}:")
                    report.append(f"    - Min: {stats['min']:.2f}")
                    report.append(f"    - Max: {stats['max']:.2f}")
                    report.append(f"    - Mean: {stats['mean']:.2f}")
                    report.append(f"    - Median: {df[col].median():.2f}")
                    report.append(f"    - Std Dev: {stats['std']:.2f}")
            
            # Sample data
            report.append("\nFirst 3 rows:")
            report.append(df.head(3).to_string())
            
            report.append("\nLast 3 rows:")
            report.append(df.tail(3).to_string())
            
        except Exception as e:
            report.append(f"Error analyzing file: {str(e)}")
        
        report.append("=" * 80)
    
    # Compile the full report
    full_report = "\n".join(report)
    
    # Output the report
    if output_file:
        with open(output_file, 'w') as f:
            f.write(full_report)
        return f"Report saved to {output_file}"
    else:
        return full_report


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze CSV files in a directory')
    parser.add_argument('directory', help='Directory containing CSV files')
    parser.add_argument('-o', '--output', help='Output file path for the report')
    
    args = parser.parse_args()
    
    result = analyze_csv_directory(args.directory, args.output)
    if not args.output:
        print(result)