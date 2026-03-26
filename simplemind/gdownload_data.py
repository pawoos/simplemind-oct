import os
import zipfile
from typing import Tuple
import gdown

def download_file_mb(file_path: str, file_url: str) -> Tuple[bool, str]:
#def download_file_mb(file_path: str, file_url: str, extracted_filename=None, csv_filename=None) -> Tuple[bool, str]:
    """
    Download and extract file from Google Drive if it doesn't exist locally.
    
    Args:
        file_path (str): Local path where weights should be stored
        file_url (str): Google Drive URL for the weights file
        
    Returns:
        Tuple[bool, str]: (success status, path to weights)
    TODO:
        To generalize the function to support more cloud storage options
    """
    temp_zip = None
    temp_extract_dir = None
    
    try:
        print(f"file_path = {file_path}")
        # Check if file already exists
        #if os.path.exists(file_path):
        #    print(f"File already exists at: {file_path}")
        #    return True, file_path
            
        print(f"Downloading file to: {file_path}")
        if not os.path.exists(file_path):
            os.makedirs(file_path)
            if not os.path.exists(file_path) or not os.path.isdir(file_path):
                print(f"Failed to create directory: {file_path}", file=sys.stderr, flush=True)
                raise FileNotFoundError(f"{file_path} was not found")
        
        # Create temp directory structure if it doesn't exist
        #temp_dir = os.path.dirname(file_path)
        #if not temp_dir:     # If no directory specified, use current directory
        #    temp_dir = '.'
        #os.makedirs(temp_dir, exist_ok=True)
        
        # Extract file ID from URL
        file_id = file_url.split("d/")[1].split("/view")[0]
        # Create download URL
        download_url = f"https://drive.google.com/uc?id={file_id}"
        #download_url = file_url
        
        # Set up temporary paths
        temp_zip = os.path.join(file_path, f"temp_file_{file_id}.zip")
        #temp_zip = os.path.join(temp_dir, "temp_file.zip")
        temp_extract_dir = os.path.join(file_path, f"temp_extract_{file_id}")
        #temp_extract_dir = os.path.join(temp_dir, "temp_extract")
        
        # Download file
        print(f"Downloading URL = {download_url}")
        success = gdown.download(download_url, temp_zip, quiet=False, verify=False)
        
        if not success or not os.path.exists(temp_zip):
            print(f"Failed to download file from {file_url}")
            return False, ""
        
        ## Create temp extraction directory
        os.makedirs(temp_extract_dir, exist_ok=True)
        
        print(f"Extracting zip file to: {file_path}")
        with zipfile.ZipFile(temp_zip, "r") as zip_ref:
            zip_ref.extractall(file_path)
                       
        # # Find the actual file in the extracted content
        # extracted_items = os.listdir(temp_extract_dir)
        # if len(extracted_items) == 1:
        #     print(extracted_items[0])
        #     source_path = os.path.join(temp_extract_dir, extracted_items[0])
        #     move_path = os.path.join(file_path, extracted_items[0])
        #     print(f"Attempting to move {source_path} to {move_path}")
        #     if os.path.exists(move_path):
        #         #os.remove(move_path)  # Remove if exists to avoid FileExistsError
        #         import shutil
        #         shutil.rmtree(move_path)
        #     #os.rename(source_path, file_path)
        #     os.rename(source_path, move_path)
        #     #print(f"Moved single file to: {file_path}")
        #     print(f"Moved single file to: {move_path}")
        #     if extracted_filename is not None:
        #         #orig_path = os.path.join(file_path, extracted_items[0])
        #         new_path = os.path.join(file_path, extracted_filename)
        #         #os.rename(orig_path, new_path)
        #         os.rename(move_path, new_path)
        #         print(f"Ranamed single file to: {new_path}")
        #     else:
        #         #new_path = os.path.join(file_path, extracted_items[0])
        #         new_path = move_path

        #     if csv_filename is not None:
        #         # Count .csv files
        #         csv_files = [file for file in os.listdir(new_path) if file.endswith(".csv")]
        #         csv_count = len(csv_files)
        #         print(f"Number of .csv files: {csv_count}")

        #         # Rename the first .csv file if it exists
        #         if csv_count == 1:
        #             old_file_path = os.path.join(new_path, csv_files[0])
        #             new_file_path = os.path.join(new_path, csv_filename)
        #             os.rename(old_file_path, new_file_path)
        #             print(f"Renamed {csv_files[0]} to {csv_filename}")
        #         elif csv_count == 0:
        #             print("No .csv files found.")
        #         else:
        #             print("More than one .csv file found. No files were renamed.")
        # else:
        #     ### NOTE: NOT TESTED -- probably not used
        #     for item in extracted_items:
        #         ### assume that `file_path` is actually a directory
        #         os.makedirs(file_path, exist_ok=True)
                
        #         source_path = os.path.join(temp_extract_dir, item)
        #         tar_path = os.path.join(file_path, item)
        #         if os.path.exists(tar_path):
        #             os.remove(tar_path)  # Remove if exists to avoid FileExistsError
        #         os.rename(source_path, file_path)
        #         print(f"Moved file to: {tar_path}")  

        # Cleanup
        if os.path.exists(temp_zip):
            os.remove(temp_zip)
        if os.path.exists(temp_extract_dir):
            import shutil
            shutil.rmtree(temp_extract_dir)
            
        if os.path.exists(file_path):
            return True, file_path
        else:
            print(f"Error: Expected file(s) not found at {file_path}")
            return False, ""

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        # Cleanup on error
        if temp_zip and os.path.exists(temp_zip):
            os.remove(temp_zip)
        if temp_extract_dir and os.path.exists(temp_extract_dir):
            import shutil
            shutil.rmtree(temp_extract_dir)
        return False, ""

if __name__ == "__main__":    
    import argparse
    parser = argparse.ArgumentParser(prog="gdownload_data.py")
    parser.add_argument('download_path', help="filepath to where data should be downloaded")
    parser.add_argument('download_url', help="url for file download")
    #parser.add_argument('extracted_foldername', help="new filename for downloaded folder")
    #parser.add_argument('csv_filename', help="new filename for csv")
    args = parser.parse_args()

    success, actual_weights_path = download_file_mb(
        args.download_path, 
        args.download_url
        #args.extracted_foldername,
        #args.csv_filename
    )
    if not success:
        raise RuntimeError(f"Failed to download files from {args.download_url}")
