# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a list of strings. Make sure each string is either "ODD"
# or "EVEN" (without the quotes) depending on whether the hexadecimal number in
# the image is odd or even. Take care not to make spelling or case mistakes. Make
# sure that the length of the list returned as output is the same as the number of
# filenames that were given as input. The judge may give unexpected results if this
# convention is not followed.
import numpy as np
import cv2
def remove_obfuscating_lines(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    para=70
    min_val = np.min(v)
    threshold = min_val + para
    img[v < threshold] = 255
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(gray_image, 254, 255, cv2.THRESH_BINARY_INV)

    return thresholded_image

def decaptcha( filenames ):
  import numpy as np
  import cv2
  import pickle
  model_file = 'trained_model.pkl'
  with open(model_file, 'rb') as file:
    model = pickle.load(file)

  image_list = []
  for image_path in filenames:
    image = cv2.imread(image_path)
    image1 = remove_obfuscating_lines(image)
    image_list.append(image1)
  
  images_array = np.array(image_list)
  final_images = []
  for images in images_array:
    shape = images.shape
    crop_images = images[0:150, 350:500]  
    final_images.append(crop_images)
  
  final = np.array(final_images)
  final = final.reshape(final.shape[0], -1)
  Pred_labels = model.predict(final)
  Pr_label = []
  for label in Pred_labels:
    if(label == 1):
      Pr_label.append('ODD')
    else:
      Pr_label.append('EVEN')
  
  labels = np.array(Pr_label)


  return labels