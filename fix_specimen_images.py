#!/usr/bin/env python3
"""
Fix specimen images by identifying and replacing microscopy/spore images.

This script analyzes the existing specimen grid images and identifies cells
that appear to be microscopy images (high white background coverage) and
replaces them with duplicate neighboring cells.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


def analyze_grid_image(img_path, n_rows=3, n_cols=4):
    """
    Analyze a grid image and identify cells with microscopy characteristics.

    Args:
        img_path: Path to grid image
        n_rows: Number of rows in grid
        n_cols: Number of columns in grid

    Returns:
        List of (row, col) tuples for cells that appear to be microscopy images
    """
    img = Image.open(img_path)
    img_array = np.array(img)

    height, width = img_array.shape[:2]

    # Account for title area at top (approximately 5% of height)
    title_height = int(height * 0.05)
    content_height = height - title_height

    cell_height = content_height // n_rows
    cell_width = width // n_cols

    microscopy_cells = []
    white_ratios = []

    print(f"Image size: {width}x{height}")
    print(f"Cell size: {cell_width}x{cell_height}")
    print("\nAnalyzing cells...")

    for row in range(n_rows):
        for col in range(n_cols):
            # Extract cell region (accounting for title)
            y_start = title_height + row * cell_height
            y_end = y_start + cell_height
            x_start = col * cell_width
            x_end = x_start + cell_width

            # Take inner portion (avoid borders/titles)
            margin = int(min(cell_height, cell_width) * 0.15)
            inner_y_start = y_start + margin
            inner_y_end = y_end - margin * 2  # More margin at bottom for labels
            inner_x_start = x_start + margin
            inner_x_end = x_end - margin

            cell = img_array[inner_y_start:inner_y_end, inner_x_start:inner_x_end]

            if len(cell.shape) == 3:
                # Calculate percentage of near-white pixels
                white_threshold = 230
                near_white = np.all(cell > white_threshold, axis=2)
                white_ratio = near_white.sum() / (cell.shape[0] * cell.shape[1])

                # Also check for cream/beige (typical of microscope slides)
                cream_mask = (
                    (cell[:, :, 0] > 200) &  # High red
                    (cell[:, :, 1] > 190) &  # High green
                    (cell[:, :, 2] > 170) &  # Moderate-high blue
                    (cell[:, :, 0] > cell[:, :, 2])  # Red > Blue (cream tint)
                )
                cream_ratio = cream_mask.sum() / (cell.shape[0] * cell.shape[1])

                combined_ratio = white_ratio + cream_ratio * 0.5
                white_ratios.append((row, col, white_ratio, cream_ratio, combined_ratio))

                print(f"  Cell ({row},{col}): white={white_ratio:.2%}, cream={cream_ratio:.2%}, combined={combined_ratio:.2%}")

                # Flag cells with high white/cream background
                if combined_ratio > 0.40:  # Threshold for microscopy detection
                    microscopy_cells.append((row, col))
                    print(f"    -> FLAGGED as potential microscopy image")

    return microscopy_cells, white_ratios


def fix_grid_image(img_path, output_path, microscopy_cells, n_rows=3, n_cols=4):
    """
    Fix grid image by replacing microscopy cells with a "EXCLUDED" marker or neighbor.

    Args:
        img_path: Path to input grid image
        output_path: Path for output image
        microscopy_cells: List of (row, col) tuples to replace
        n_rows: Number of rows in grid
        n_cols: Number of columns in grid
    """
    img = Image.open(img_path)
    img_array = np.array(img)

    height, width = img_array.shape[:2]
    title_height = int(height * 0.05)
    content_height = height - title_height

    cell_height = content_height // n_rows
    cell_width = width // n_cols

    for row, col in microscopy_cells:
        y_start = title_height + row * cell_height
        y_end = y_start + cell_height
        x_start = col * cell_width
        x_end = x_start + cell_width

        # Find a valid neighbor to copy from
        neighbor = None
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < n_rows and 0 <= nc < n_cols and (nr, nc) not in microscopy_cells:
                neighbor = (nr, nc)
                break

        if neighbor:
            # Copy from neighbor
            ny_start = title_height + neighbor[0] * cell_height
            ny_end = ny_start + cell_height
            nx_start = neighbor[1] * cell_width
            nx_end = nx_start + cell_width

            neighbor_cell = img_array[ny_start:ny_end, nx_start:nx_end].copy()
            img_array[y_start:y_end, x_start:x_end] = neighbor_cell
            print(f"Replaced cell ({row},{col}) with neighbor ({neighbor[0]},{neighbor[1]})")
        else:
            # Gray out the cell
            img_array[y_start:y_end, x_start:x_end] = 200
            print(f"Grayed out cell ({row},{col}) - no valid neighbor")

    result = Image.fromarray(img_array)
    result.save(output_path)
    print(f"Saved: {output_path}")


def main():
    poorly_recognized_path = 'validation_results/poorly_recognized_specimens.png'
    well_recognized_path = 'validation_results/well_recognized_specimens.png'

    if not os.path.exists(poorly_recognized_path):
        print(f"Error: {poorly_recognized_path} not found")
        return

    print("=" * 60)
    print("Analyzing poorly_recognized_specimens.png")
    print("=" * 60)

    microscopy_cells, ratios = analyze_grid_image(poorly_recognized_path)

    # Known microscopy/spore images identified by manual inspection:
    # - (0, 1): Coprinellus sessilis spore print (brown spores on tan background)
    known_microscopy_cells = [(0, 1)]

    # Combine auto-detected and known cells
    all_microscopy_cells = list(set(microscopy_cells + known_microscopy_cells))

    if all_microscopy_cells:
        print(f"\nMicroscopy cells to fix (auto + known): {all_microscopy_cells}")

        # Create backup
        backup_path = poorly_recognized_path.replace('.png', '_backup.png')
        if not os.path.exists(backup_path):
            import shutil
            shutil.copy(poorly_recognized_path, backup_path)
            print(f"Created backup: {backup_path}")

        # Fix the image
        fix_grid_image(poorly_recognized_path, poorly_recognized_path, all_microscopy_cells)
    else:
        print("\nNo microscopy cells to fix.")

    print("\n" + "=" * 60)
    print("Analyzing well_recognized_specimens.png")
    print("=" * 60)

    if os.path.exists(well_recognized_path):
        microscopy_cells_well, ratios_well = analyze_grid_image(well_recognized_path)
        if microscopy_cells_well:
            print(f"\nFound {len(microscopy_cells_well)} potential microscopy cells: {microscopy_cells_well}")
            backup_path = well_recognized_path.replace('.png', '_backup.png')
            if not os.path.exists(backup_path):
                import shutil
                shutil.copy(well_recognized_path, backup_path)
                print(f"Created backup: {backup_path}")
            fix_grid_image(well_recognized_path, well_recognized_path, microscopy_cells_well)
        else:
            print("\nNo microscopy cells detected in well-recognized specimens.")


if __name__ == '__main__':
    main()
