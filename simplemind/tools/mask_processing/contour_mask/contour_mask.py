#not yet a simplemind tool

import cv2
import numpy as np
import csv
import os

base = "C:/Users/sachi/Pictures/Segmentation Data"
main = '/contours_testing'
cropped_dir = base + "/cropped" # this is the directory of the segmentation masks per feature straight from SimpleMind (with one post-processing - they've been cropped to remove the outer white boundary)
save_dir = base + main + "/contours"
seg_dir = base + main + "/segmentations"
raw_dir = base + main + "/original"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for file_name in os.listdir(seg_dir):
    if file_name.endswith(('.png', '.jpg', '.jpeg')):  # Add other extensions if needed

        og_path = os.path.join(raw_dir, file_name)
        og_image = cv2.imread(og_path)
        if og_image is None:
            print(f"Error loading image: {og_image}")
            continue


        prefix = file_name.replace('.png','').replace('.jpg','').replace('.jpeg','')

        for structure in ['iris', 'cornea', 'lens']:
            cropped_path = os.path.join(cropped_dir, f"{prefix}_{structure}.png")
            if os.path.exists(cropped_path):
                image = cv2.imread(cropped_path)
                if image.shape[:2] != og_image.shape[:2]:
                    image = cv2.resize(image, (og_image.shape[1], og_image.shape[0]), interpolation=cv2.INTER_NEAREST)

                # Convert image to HSV color space for better red detection
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                # Define lower and upper ranges for red in HSV
                lower_red = np.array([0, 120, 70])  # Lower HSV values
                upper_red = np.array([10, 255, 255])  # Upper HSV values
                red = cv2.inRange(hsv, lower_red, upper_red)
                # Ensure binary mask
                _, binary = cv2.threshold(red, 127, 255, cv2.THRESH_BINARY)
                # Find contours
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Filter contours by area threshold
                if structure == 'iris':
                    min_area = 5000
                else:
                    min_area = 15000
                contours = [c for c in contours if cv2.contourArea(c) >= min_area]

                # Save contours to CSV
                with open(f'{save_dir}/{prefix}_{structure}.csv', 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['contour_id', 'x', 'y'])
                    for idx, contour in enumerate(contours):
                        for point in contour:
                            x, y = point[0]
                            writer.writerow([idx, x, y])

                # Recreate binary mask from contours.csv
                mask_from_csv = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                with open(f'{save_dir}/{prefix}_{structure}.csv', 'r') as csvfile:
                    reader = csv.DictReader(csvfile)
                    contours_points = {}
                    for row in reader:
                        cid = int(row['contour_id'])
                        x = int(row['x'])
                        y = int(row['y'])
                        if cid not in contours_points:
                            contours_points[cid] = []
                        contours_points[cid].append([x, y])
                    # Ensure each contour is closed (last point == first point)
                    for points in contours_points.values():
                        if points[0] != points[-1]:
                            points.append(points[0])
                    contours_list = [np.array(points, dtype=np.int32).reshape(-1, 1, 2) for points in contours_points.values()]
                    cv2.drawContours(mask_from_csv, contours_list, -1, 255, thickness=1)
                    cv2.imwrite(f'{save_dir}/{prefix}_{structure}.png', mask_from_csv)

                    # Resize contours if dimensions mismatch
                    if og_image.shape[:2] != image.shape[:2]:
                        scale_y = og_image.shape[0] / image.shape[0]
                        scale_x = og_image.shape[1] / image.shape[1]
                        resized_contours_list = []
                        for contour in contours_list:
                            contour = contour.astype(np.float32)
                            contour[:, 0, 0] *= scale_x
                            contour[:, 0, 1] *= scale_y
                            resized_contours_list.append(contour.astype(np.int32))
                        contours_list = resized_contours_list

                        # Save resized contours to CSV
                        with open(f'{save_dir}/{prefix}_{structure}_resized.csv', 'w', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(['contour_id', 'x', 'y'])
                            for idx, contour in enumerate(contours_list):
                                for point in contour:
                                    x, y = point[0]
                                    writer.writerow([idx, x, y])

                    cv2.drawContours(og_image, contours_list, -1, (0,255,0), thickness=3)
        
        cv2.imwrite(f'{save_dir}/{prefix}.png', og_image)

                ###############################################################################
                # og code that was improved upon:
                ###############################################################################
                # # Overlay the structure_contours onto the original image
                # overlay = og_image.copy()
                # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # iris=blue, cornea=green, lens=red
                # for mask, color in zip(structure_contours, colors):
                #     # Create a colored mask
                #     colored_mask = np.zeros_like(overlay)
                #     for c in range(3):
                #         colored_mask[:, :, c] = (mask == 255) * color[c]
                #     # Blend the colored mask onto the original image
                #     overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)
                # cv2.imwrite(os.path.join(save_dir, f"{prefix}_overlay_on_original.png"), overlay)

                ###############################################################################
                # # Define HSV bounds for blue
                # lower_blue = np.array([100, 120, 0])
                # upper_blue = np.array([140, 255, 255])
                # # Convert image to HSV and apply the mask
                # blue = cv2.inRange(hsv, lower_blue, upper_blue)
                # # Define HSV bounds for green
                # lower_green = np.array([36, 100, 0])
                # upper_green = np.array([86, 255, 255])
                # # Convert image to HSV and apply the mask
                # green = cv2.inRange(hsv, lower_green, upper_green)

                # masks = [red, green, blue]
                # colors = ['cornea', 'lens', 'iris']

                # # # Save extracted binary mask for later use
                # # cv2.imwrite('mask.png', red)

                # # # Load binary mask image (update 'mask.png' to your file path)
                # # mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)
                # for color, color_mask in zip(colors, masks):
                #     # Ensure binary mask
                #     _, binary = cv2.threshold(color_mask, 127, 255, cv2.THRESH_BINARY)

                #     # Find contours
                #     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                #     # Filter contours by area threshold
                #     min_area = 15000
                #     contours = [c for c in contours if cv2.contourArea(c) >= min_area]

                #     # # Create blank image for contours
                #     # contour_img = np.zeros_like(image)

                #     # # Draw contours on the blank image (white color, thickness=1)
                #     # cv2.drawContours(contour_img, contours, -1, (255), thickness=1)

                #     # # Save the binary image with contours
                #     # cv2.imwrite(f'contours_color_{i}.png', contour_img)

                #     # Save contours to CSV
                #     with open(f'{save_dir}/{file_name}_{color}.csv', 'w', newline='') as csvfile:
                #         writer = csv.writer(csvfile)
                #         writer.writerow(['contour_id', 'x', 'y'])
                #         for idx, contour in enumerate(contours):
                #             for point in contour:
                #                 x, y = point[0]
                #                 writer.writerow([idx, x, y])

                #     # Recreate binary mask from contours.csv
                #     mask_from_csv = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                #     with open(f'{save_dir}/{file_name}_{color}.csv', 'r') as csvfile:
                #         reader = csv.DictReader(csvfile)
                #         contours_points = {}
                #         for row in reader:
                #             cid = int(row['contour_id'])
                #             x = int(row['x'])
                #             y = int(row['y'])
                #             if cid not in contours_points:
                #                 contours_points[cid] = []
                #             contours_points[cid].append([x, y])
                #         # Ensure each contour is closed (last point == first point)
                #         for points in contours_points.values():
                #             if points[0] != points[-1]:
                #                 points.append(points[0])
                #         contours_list = [np.array(points, dtype=np.int32).reshape(-1, 1, 2) for points in contours_points.values()]
                #         cv2.drawContours(mask_from_csv, contours_list, -1, 255, thickness=1)

                #         cv2.imwrite(f'{save_dir}/{file_name}_{color}.png', mask_from_csv)
