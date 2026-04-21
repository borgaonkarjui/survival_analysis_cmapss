import os
import re

def get_next_version(directory, base_filename, extension):
    # Regex to find 'base_filename_v###.extension'
    pattern = re.compile(rf"^{base_filename}_v(\d+)\.{extension}$")
    
    max_v = 0
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            # Extract the version number and find the max
            version = int(match.group(1))
            if version > max_v:
                max_v = version
    
    # Increment for the new version
    next_v = max_v + 1
    return f"{base_filename}_v{next_v:03d}.{extension}"