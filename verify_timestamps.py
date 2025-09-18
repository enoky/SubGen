import re
import sys

def parse_srt(file_path):
    """
    Parses a full SRT file to extract sequence numbers, timestamps, and text.

    Args:
        file_path (str): The path to the SRT file.

    Returns:
        A dictionary mapping sequence numbers (int) to another dictionary
        containing the 'timestamp' and 'text' for that sequence.
        Returns an empty dictionary if the file cannot be read.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read and strip to remove leading/trailing whitespace which might create empty blocks
            content = f.read().strip()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"Error reading file {file_path}: {e}", file=sys.stderr)
        return {}

    # Split the file content into subtitle blocks based on one or more blank lines
    blocks = re.split(r'\n\s*\n', content)
    srt_data = {}

    for block in blocks:
        if not block.strip():
            continue # Skip empty blocks

        lines = block.strip().split('\n')
        # A valid block needs at least a sequence number and a timestamp.
        # Text can be empty.
        if len(lines) >= 2:
            try:
                seq = int(lines[0])
                timestamp = lines[1]
                # Join the rest of the lines as text
                text = '\n'.join(lines[2:])
                srt_data[seq] = {'timestamp': timestamp, 'text': text}
            except (ValueError, IndexError):
                print(f"Warning: Could not parse block:\n---\n{block}\n---", file=sys.stderr)
    return srt_data

def create_verified_file(edited_file, original_file, output_file):
    """
    Creates a new SRT file by combining the text from the edited file
    with the timestamps from the original file.

    Args:
        edited_file (str): Path to the edited SRT file (source of text).
        original_file (str): Path to the original SRT file (source of timestamps).
        output_file (str): Path to write the new, combined SRT file.
    """
    print(f"Loading corrected text from: {edited_file}")
    edited_subs = parse_srt(edited_file)
    if not edited_subs:
        print("Could not process the edited file. Aborting.")
        return

    print(f"Loading original timestamps from: {original_file}")
    original_subs = parse_srt(original_file)
    if not original_subs:
        print("Could not process the original file. Aborting.")
        return

    verified_blocks = []
    # Iterate through the edited subtitles to maintain their order and text
    # Sorting by key (sequence number) ensures the output is in the correct order.
    for seq in sorted(edited_subs.keys()):
        edited_entry = edited_subs[seq]
        original_entry = original_subs.get(seq)

        timestamp = ""
        if original_entry:
            # Use the timestamp from the original file
            timestamp = original_entry['timestamp']
        else:
            # If sequence is not in the original, use the edited timestamp and warn the user
            timestamp = edited_entry['timestamp']
            print(f"Warning: Subtitle #{seq} not found in original file. Using timestamp from edited file.")

        # Get the text from the edited file
        text = edited_entry['text']
        
        # Construct the final SRT block string
        verified_blocks.append(f"{seq}\n{timestamp}\n{text}")

    # Write the new, verified SRT file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # Join blocks with two newlines for standard SRT formatting
            f.write('\n\n'.join(verified_blocks))
        print(f"\nVerification complete!")
        print(f"Successfully created synchronized subtitle file at: {output_file}")
    except Exception as e:
        print(f"\nError writing to output file {output_file}: {e}", file=sys.stderr)

if __name__ == '__main__':
    # Set default file names
    edited_srt_path = 'edited-subs.srt'
    original_srt_path = 'autogen-subs.srt'
    output_srt_path = 'verified-subs.srt'
    
    # The script can be run without arguments, using the default names,
    # or with arguments to specify the files.
    if len(sys.argv) == 4:
        edited_srt_path = sys.argv[1]
        original_srt_path = sys.argv[2]
        output_srt_path = sys.argv[3]
    elif len(sys.argv) != 1:
        print("Usage: python srt_sync.py")
        print("   or: python srt_sync.py <edited_file.srt> <original_file.srt> <output_file.srt>")
        sys.exit(1)

    create_verified_file(edited_srt_path, original_srt_path, output_srt_path)
