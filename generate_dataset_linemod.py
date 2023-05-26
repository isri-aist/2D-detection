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
object_classes_dict["000001"] = 0
object_classes_dict["000002"] = 1
object_classes_dict["000003"] = 2
object_classes_dict["000004"] = 3
object_classes_dict["000005"] = 4
object_classes_dict["000006"] = 5
object_classes_dict["000007"] = 6
object_classes_dict["000008"] = 7
object_classes_dict["000009"] = 8
object_classes_dict["000010"] = 9
object_classes_dict["000011"] = 10
object_classes_dict["000012"] = 11
object_classes_dict["000013"] = 12
object_classes_dict["000014"] = 13
object_classes_dict["000015"] = 14

background_images_path = "raw_data/background_samples/"
#get list of all images
background_images_list = os.listdir(background_images_path)

random_object_images_path = "raw_data/random_objects/"
#get list of all images
random_object_images_list = os.listdir(random_object_images_path)

objects_list_path = "linemod/lm_train/train/"
#get list of all objects
objects_list = os.listdir(objects_list_path)

dataset_path = "yolov8/linemod_dataset/custom_dataset/"

def generate_yolov8_dataset(background_image, x_center, y_center, bb_width, bb_height, object_rotation_rad, image_name, object_name):
    #generate a dataset for YOLOv8
    #Bounding box is anchor free
    #Bounding box format: class, x_center, y_center, width, height
    #class is the class of the object (int)
    #x_center, y_center are the coordinates of the center of the bounding box
    #width, height are the width and height of the bounding box
    
    dataset_path = "yolov8/linemod_dataset/custom_dataset/"

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

def add_object_to_detect(background_image, object_image):
    #remove pixels that are all black (r is 0, g is 0, b is 0)
    object_image_np = np.array(object_image)
    object_image_np[(object_image_np == [0, 0, 0, 255]).all(axis=2)] = [0, 0, 0, 0]
    object_image = Image.fromarray(object_image_np).convert("RGBA")

    # Get the alpha channel as a separate image
    alpha = object_image.getchannel("A")

    # Find the non-transparent regions of the image
    bbox = alpha.getbbox()

    #crop image to bounding box
    object_image = object_image.crop(bbox)

    background_size = background_image.size
    if background_size[0] < 640 or background_size[1] < 640:
        print("Warning: background image is too small")

    # Resize the object image with a random size
    scale = random.uniform(0.3, 1.2)
    if object_image.size[0] > background_size[0] or object_image.size[1] > background_size[1]:
        scale = scale * min((background_size[0] / object_image.size[0]), (background_size[1] / object_image.size[1])) #ensure object is not bigger than background
    new_size = (int(object_image.size[0] * scale), int(object_image.size[1] * scale))
    object_image = object_image.resize(new_size)

    object_rotation_rad = random.uniform(0, 2*np.pi)
    object_image = object_image.rotate(object_rotation_rad*180/np.pi, expand=True)

    new_alpha = object_image.getchannel("A")

    # Find the non-transparent regions of the image
    bbox = new_alpha.getbbox()

    #crop image to bounding box
    object_image = object_image.crop(bbox)

    #adds some random transformations (hue/saturation/lightness, contrast, brightness)
    change_color = random.randint(0, 1)
    if change_color == 1:
        object_image = ImageEnhance.Color(object_image).enhance(random.uniform(0.8, 1.2))
    change_contrast = random.randint(0, 1)
    if change_contrast == 1:
        object_image = ImageEnhance.Contrast(object_image).enhance(random.uniform(0.8, 1.2))
    change_brightness = random.randint(0, 1)
    if change_brightness == 1:
        object_image = ImageEnhance.Brightness(object_image).enhance(random.uniform(0.8, 1.2))

    # Place the object image onto the background image with a random position and orientation
    object_position = (random.randint(0, background_size[0] - object_image.size[0]), random.randint(0, background_size[1] - object_image.size[1]))
    background_image.paste(object_image, object_position, object_image)

    new_width = object_image.size[0]
    new_height = object_image.size[1]
    x_center = object_position[0] + new_width / 2
    y_center = object_position[1] + new_height / 2
    object_name = objects_list[k]
    class_id = object_classes_dict[object_name]

    x_center = x_center / background_image.width
    y_center = y_center / background_image.height

    bb_width = new_width / background_image.width
    bb_height = new_height / background_image.height

    return background_image, object_position, x_center, y_center, bb_width, bb_height

#main
if __name__ == "__main__":

    for k in tqdm(range(len(objects_list))):
        raw_object_images_path = objects_list_path + objects_list[k] + "/rgb/"
        object_images_list = os.listdir(raw_object_images_path)
    
        for i in tqdm(range(len(object_images_list))):
            image_name = object_images_list[i]
            for j in range(1):
                # Load the object image with a transparent background
                object_image = Image.open(raw_object_images_path + image_name).convert("RGBA")

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


                #add some random_object to the background before pasting the object
                num_random_objects = random.randint(0, 5)
                for l in range(num_random_objects):
                    background_image = add_random_object(background_image)

                #add object to detect to the background
                background_image, object_position, x_center, y_center, bb_width, bb_height = add_object_to_detect(background_image, object_image)
                object_name = objects_list[k]
                class_id = object_classes_dict[object_name]

                x_center_list = [x_center]
                y_center_list = [y_center]
                bb_width_list = [bb_width]
                bb_height_list = [bb_height]
                class_id_list = [class_id]

                num_additional_objects_to_detect = random.randint(0, 2)
                for l in range(num_additional_objects_to_detect):
                    random_index_object = random.randint(0, len(objects_list) - 1)
                    object_list_of_images = os.listdir(objects_list_path + objects_list[random_index_object] + "/rgb/")
                    random_index_image = random.randint(0, len(object_list_of_images) - 1)
                    object_image = Image.open(objects_list_path + objects_list[random_index_object] + "/rgb/" + object_list_of_images[random_index_image]).convert("RGBA")
                    background_image, object_position, x_center, y_center, bb_width, bb_height = add_object_to_detect(background_image, object_image)
                    class_id = object_classes_dict[objects_list[random_index_object]]
                    x_center_list.append(x_center)
                    y_center_list.append(y_center)
                    bb_width_list.append(bb_width)
                    bb_height_list.append(bb_height)
                    class_id_list.append(class_id)


                #add a random_object to the background after pasting the object with probability 0.25
                if random.uniform(0, 1) < 0.25:
                    background_image = add_random_object(background_image)
                

                #add a random gaussian blur
                background_image = background_image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1)))

                background_image = ImageEnhance.Color(background_image).enhance(random.uniform(0.9, 1.1))
                background_image = ImageEnhance.Contrast(background_image).enhance(random.uniform(0.9, 1.1))
                background_image = ImageEnhance.Brightness(background_image).enhance(random.uniform(0.9, 1.1))

                random_number = random.uniform(0, 1)
                if random_number < 0.8:
                    save_path = "train/"
                elif random_number < 0.9:
                    save_path = "valid/"
                else:
                    save_path = "test/"
                # Save the resulting image
                #random id with letters and numbers
                import string
                random_id = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(5))
                background_image.save(dataset_path+save_path+"images/" + object_name + image_name[:-3] + "_" + random_id + ".jpg")
                #write label file
                with open(dataset_path+save_path+"labels/" + object_name + image_name[:-3] + "_" + random_id + ".txt", "w") as f:
                    for m in range(len(x_center_list)):
                        f.write(str(class_id_list[m]) + " " + str(x_center_list[m]) + " " + str(y_center_list[m]) + " " + str(bb_width_list[m]) + " " + str(bb_height_list[m]) + "\n")


                #             #show image with bounding box
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches
                fig, ax = plt.subplots(1)
                ax.imshow(background_image)
                for m in range(len(x_center_list)):
                    rect = patches.Rectangle((x_center_list[m]*background_image.width - bb_width_list[m]*background_image.width/2, y_center_list[m]*background_image.height - bb_height_list[m]*background_image.height/2), bb_width_list[m]*background_image.width, bb_height_list[m]*background_image.height, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                plt.show()
                exit()

                generate_yolov8_dataset(background_image, x_center, y_center, new_width, new_height, object_rotation_rad, image_name, objects_list[k])