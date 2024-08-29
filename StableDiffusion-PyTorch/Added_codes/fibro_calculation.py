from PIL import Image
import os
threshold = 10
def count_pixels(image_path, fibro_target_color, fat_target_color):
    # Open the image
    img = Image.open(image_path)
    
    # Convert image to RGB if it's not already in RGB mode
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Load the image data
    pixels = img.load()
    
    # Get image dimensions
    width, height = img.size
    
    # Initialize fibrocount
    fibrocount = 0
    fatcount = 0
    # Loop through each pixel in the image
    for x in range(width):
        for y in range(height):
            if (fibro_target_color[0] - threshold) <= pixels[x, y][0] <= (fibro_target_color[0] + threshold) and (fibro_target_color[1] - threshold) <= pixels[x, y][1] <= (fibro_target_color[1] + threshold) and (fibro_target_color[2] - threshold) <= pixels[x, y][2] <= (fibro_target_color[2] + threshold):
                fibrocount += 1
            elif (fat_target_color[0] - threshold) <= pixels[x, y][0] <= (fat_target_color[0] + threshold) and (fat_target_color[1] - threshold) <= pixels[x, y][1] <= (fat_target_color[1] + threshold) and (fat_target_color[2] - threshold) <= pixels[x, y][2] <= (fat_target_color[2] + threshold):
                fatcount += 1
    return fibrocount, fatcount

fibro_target_color = (34, 138, 139)
fat_target_color = (57, 81, 137)
folder_path = '/Users/abharian/LDM_Project/StableDiffusion-PyTorch/mnist/Generated_Results/'
images = []
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
image_files = os.listdir(folder_path)
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    img = Image.open(image_path)
    fibrocount, fatcount = count_pixels(image_path, fibro_target_color, fat_target_color)
    print("Image: ", image_file)
    print(f'Number of fibro pixels: {fibrocount}')
    print(f'Number of fat pixels: {fatcount}')
    Fat_Fibro_Sum = (fatcount + fibrocount)
    BIRAD_percent = (fibrocount / Fat_Fibro_Sum) *100
    print(f'Bi-RAD Percentage: {BIRAD_percent}')
    if BIRAD_percent < 25 :
        print("class 0")
    elif 25 <= BIRAD_percent < 50:
        print("class 1")
    elif 50 <= BIRAD_percent < 75:
        print("class 2")
    else:
        print("class 3")

#2, 2, 0, 0, 0, 0, 
# 2, 1, 2, 1, 0, 0, 
# 2, 2, 0, 1, 1, 2, 
# 1, 2, 0, 1, 0, 2, 0
