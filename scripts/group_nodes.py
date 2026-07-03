#!/usr/bin/env python3
"""
Group nodes by rack ID from node_list.txt.
"""

from collections import defaultdict
from pathlib import Path


def group_nodes_by_rack(input_file: str, output_file: str = None) -> None:
    """
    Group nodes by rack ID and write to output file.
    
    Args:
        input_file: Path to input node_list.txt
        output_file: Path to output file (optional)
    """
    # Read and group nodes
    racks = defaultdict(list)
    
    # Check if file is already grouped or original format
    with open(input_file, 'r') as f:
        first_line = f.readline().strip()
        if first_line.startswith('#'):
            # Already grouped, parse current format
            f.seek(0)
            current_rack = None
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    current_rack = line.split()[1]
                elif line:
                    racks[current_rack].append(line)
        else:
            # Original format, parse IP and rack
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ip, rack = line.split('\t')
                racks[rack].append(ip)
    
    # Generate output
    output_lines = []
    for rack in sorted(racks.keys()):
        output_lines.append(f"# {rack} - {len(racks[rack])} nodes")
        output_lines.extend(racks[rack])
        output_lines.append("")  # Empty line between racks
    
    output_content = '\n'.join(output_lines).rstrip('\n') + '\n'
    
    # Write output
    if output_file is None:
        output_file = input_file
    
    with open(output_file, 'w') as f:
        f.write(output_content)
    
    print(f"Grouped nodes written to: {output_file}")
    print(f"Total racks: {len(racks)}")
    total_nodes = sum(len(nodes) for nodes in racks.values())
    print(f"Total nodes: {total_nodes}")
    for rack in sorted(racks.keys()):
        print(f"  Rack {rack}: {len(racks[rack])} nodes")


if __name__ == '__main__':
    input_path = '/Users/robin/work_dir/ScaleTorch/docs/node_list_original.txt'
    output_path = '/Users/robin/work_dir/ScaleTorch/docs/node_list.txt'
    group_nodes_by_rack(input_path, output_path)
