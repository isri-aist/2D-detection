from PIL import Image
import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import os
import sys
from tqdm import tqdm
from PIL import ImageEnhance, ImageOps, ImageFilter

object_classes_dict = dict()
#object_classes_dict["noodlecup"] = 0
#object_classes_dict["currycup"] = 1
#object_classes_dict["seafoodcup"] = 2
#object_classes_dict["BoxLid"] = 0
object_classes_dict["BoxNoLid"] = 0

background_images_path = "raw_data/background_samples/"
#get list of all images
background_images_list = os.listdir(background_images_path)
background_images_list = [x for x in background_images_list if x.endswith(".png") or x.endswith(".jpg") or x.endswith(".jpeg")]

random_object_images_path = "raw_data/random_objects/"
#get list of all images
random_object_images_list = os.listdir(random_object_images_path)
random_object_images_list = [x for x in random_object_images_list if x.endswith(".png") or x.endswith(".jpg") or x.endswith(".jpeg")]

objects_list_path = "raw_data/objects_to_detect/"
#get list of all objects, keep only directories
objects_list = os.listdir(objects_list_path)
objects_list = [x for x in objects_list if os.path.isdir(objects_list_path + x)]

dataset_path = "yolov8/dataset/custom_dataset/"

def generate_yolov8_dataset(background_image, x_center, y_center, bb_width, bb_height, object_rotation_rad, image_name, object_name):
    #generate a dataset for YOLOv8
    #Bounding box is anchor free
    #Bounding box format: class, x_center, y_center, width, height
    #class is the class of the object (int)
    #x_center, y_center are the coordinates of the center of the bounding box
    #width, height are the width and height of the bounding box
    
    class_id = object_classes_dict[object_name]

    x_center = x_center / background_image.width
    y_center = y_center / background_image.height

    bb_width = bb_width / background_image.width
    bb_height = bb_height / background_image.height


    # Save the resulting image
    random_number = random.uniform(0, 1)
    if random_number < 0.8:
        background_image.save(dataset_path+"train/images/" + image_name + str(object_rotation_rad) + ".jpg")
        #write label file
        with open(dataset_path+"train/labels/" + image_name + str(object_rotation_rad) + ".txt", "w") as f:
            f.write(str(class_id) + " " + str(x_center) + " " + str(y_center) + " " + str(bb_width) + " " + str(bb_height))
    elif random_number < 0.9:
        background_image.save(dataset_path+"valid/images/" + image_name + str(object_rotation_rad) + ".jpg")
        #write label file
        with open(dataset_path+"valid/labels/" + image_name + str(object_rotation_rad) + ".txt", "w") as f:
            f.write(str(class_id) + " " + str(x_center) + " " + str(y_center) + " " + str(bb_width) + " " + str(bb_height))
    else:
        background_image.save(dataset_path+"test/images/" + image_name + str(object_rotation_rad) + ".jpg")
        #write label file
        with open(dataset_path+"test/labels/" + image_name + str(object_rotation_rad) + ".txt", "w") as f:
            f.write(str(class_id) + " " + str(x_center) + " " + str(y_center) + " " + str(bb_width) + " " + str(bb_height))


def add_random_object(background_image):
    random_object_image = Image.open(random_object_images_path + random.choice(random_object_images_list)).convert("RGBA")
    #scale the object between 2 and 30% of the background image
    background_size = background_image.size
    alpha = random_object_image.split()[3]
    bbox = alpha.getbbox()
    random_object_image = random_object_image.crop(bbox)
    scale = random.uniform(0.02, 0.3)
    random_object_image = random_object_image.resize((int(background_size[0]*scale), int(background_size[1]*scale)))
    random_object_image = random_object_image.rotate(random.uniform(0, 360), expand=True)
    random_object_position = (random.randint(0, background_size[0] - random_object_image.size[0]), random.randint(0, background_size[1] - random_object_image.size[1]))
    change_color = random.randint(0, 1)
    if change_color == 1:
        random_object_image = ImageEnhance.Color(random_object_image).enhance(random.uniform(0.1, 2.5))
    change_contrast = random.randint(0, 1)
    if change_contrast == 1:
        random_object_image = ImageEnhance.Contrast(random_object_image).enhance(random.uniform(0.1, 2.5))
    change_brightness = random.randint(0, 1)
    if change_brightness == 1:
        random_object_image = ImageEnhance.Brightness(random_object_image).enhance(random.uniform(0.1, 2.5))

    background_image.paste(random_object_image, random_object_position, random_object_image)
    return background_image


#main
if __name__ == "__main__":

    for k in tqdm(range(len(objects_list))):
        raw_object_images_path = objects_list_path + objects_list[k] + "/"
        #load all images, ignore non-image files
        object_images_list = os.listdir(raw_object_images_path)
        object_images_list = [x for x in object_images_list if x.endswith(".png") or x.endswith(".jpg") or x.endswith(".jpeg")]
    
        for i in tqdm(range(len(object_images_list))):
            image_name = object_images_list[i]
            object_rotation_rad = 0
            for j in range(20):
                # Load the object image with a transparent background
                object_image = Image.open(raw_object_images_path + image_name).convert("RGBA")

                # Get the alpha channel as a separate image
                alpha = object_image.getchannel("A")

                # Cleaning: set all alpha values that are lower than 100 to 0, keep all the others
                alpha = alpha.point(lambda i: 0 if i<100 else i)

                # Set the alpha channel of the object image to the new alpha channel
                object_image.putalpha(alpha)

                # Find the non-transparent regions of the image
                bbox = alpha.getbbox()

                #crop image to bounding box
                object_image = object_image.crop(bbox)

                # Choose a random background image
                number_of_background_images = len(background_images_list)
                random_index = random.randint(0, number_of_background_images)
                if random_index == number_of_background_images:
                    #random value for each pixel
                    pixel_data = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                    background_image = Image.fromarray(pixel_data)
                else:
                    background_image = Image.open(background_images_path + background_images_list[random_index])
                    #crop if image is too big
                    if background_image.size[0] > 640 or background_image.size[1] > 640:
                        background_image = background_image.crop((0, 0, 640, 640))
                    #resize if image is too small
                    if background_image.size[0] < 640 or background_image.size[1] < 640:
                        background_image = background_image.resize((640, 640))


                background_size = background_image.size
                if background_size[0] < 640 or background_size[1] < 640:
                    print("Warning: background image is too small")

                # Resize the object image with a random size
                scale = random.uniform(0.05, 0.8)
                if object_image.size[0] > background_size[0] or object_image.size[1] > background_size[1]:
                    scale = scale * min((background_size[0] / object_image.size[0]), (background_size[1] / object_image.size[1])) #ensure object is not bigger than background
                new_size = (int(object_image.size[0] * scale), int(object_image.size[1] * scale))
                object_image = object_image.resize(new_size)

                width = object_image.size[0]
                height = object_image.size[1]

                object_image = object_image.rotate(object_rotation_rad*180/np.pi, expand=True)

                new_alpha = object_image.getchannel("A")

                # Find the non-transparent regions of the image
                bbox = new_alpha.getbbox()

                #crop image to bounding box
                object_image = object_image.crop(bbox)

                new_width, new_height = object_image.size

                #adds some random transformations (hue/saturation/lightness, contrast, brightness)
                change_color = random.randint(0, 1)
                if change_color == 1:
                    object_image = ImageEnhance.Color(object_image).enhance(random.uniform(0.5, 1.5))
                change_contrast = random.randint(0, 1)
                if change_contrast == 1:
                    object_image = ImageEnhance.Contrast(object_image).enhance(random.uniform(0.5, 1.5))
                change_brightness = random.randint(0, 1)
                if change_brightness == 1:
                    object_image = ImageEnhance.Brightness(object_image).enhance(random.uniform(0.5, 1.5))

                # Place the object image onto the background image with a random position and orientation
                object_position = (random.randint(0, background_size[0] - object_image.size[0]), random.randint(0, background_size[1] - object_image.size[1]))
                
                #add some random_object to the background before pasting the object
                num_random_objects = random.randint(0, 5)
                for l in range(num_random_objects):
                    background_image = add_random_object(background_image)

                # Place the resized object image onto the background image with the same position and orientation
                background_image.paste(object_image, object_position, object_image)

                #add a random_object to the background after pasting the object with probability 0.25
                if random.uniform(0, 1) < 0.25:
                    background_image = add_random_object(background_image)
                

                #add a random gaussian blur
                background_image = background_image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1)))

                background_image = ImageEnhance.Color(background_image).enhance(random.uniform(0.9, 1.1))
                background_image = ImageEnhance.Contrast(background_image).enhance(random.uniform(0.9, 1.1))
                background_image = ImageEnhance.Brightness(background_image).enhance(random.uniform(0.9, 1.1))

                x_center = object_position[0] + new_width / 2
                y_center = object_position[1] + new_height / 2


                generate_yolov8_dataset(background_image, x_center, y_center, new_width, new_height, object_rotation_rad, image_name, objects_list[k])

                object_rotation_rad = random.uniform(0, 2*np.pi)