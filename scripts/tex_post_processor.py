#!/usr/bin/env python3
import re
import sys

def mark_tables(tex_content):
    """
    Finds every occurrence of a table block defined as "table{ ... }"
    (without nested braces) and wraps it with unique markers.
    
    The table block will be replaced by:
    
    % BEGIN TABLE 1
    table{
      ... contents ...
    }
    % END TABLE 1
    
    Returns the updated tex content.
    """
    # This pattern matches "table{" followed by any content until the first "}"
    # It uses a non-greedy match.
    pattern = r"(table\s*\{.*?\})"
    counter = 0

    def replacer(match):
        nonlocal counter
        counter += 1
        block = match.group(1)
        return f"% BEGIN TABLE {counter}\n{block}\n% END TABLE {counter}"

    updated_content = re.sub(pattern, replacer, tex_content, flags=re.DOTALL)
    print(f"Marked {counter} table(s) in the file.")
    return updated_content

def main(input_filename, output_filename):
    # Read the original tex file
    with open(input_filename, 'r') as infile:
        tex_content = infile.read()

    # Mark the tables using the defined function
    updated_content = mark_tables(tex_content)

    # Write the updated content to the output file
    with open(output_filename, 'w') as outfile:
        outfile.write(updated_content)
    print(f"Converted file written to: {output_filename}")

if __name__ == '__main__':
        
    input_output_names = []

    main(input_filename, output_filename)