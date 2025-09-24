import cv2
import os
# Load image

def is_profile_picture(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load OpenCV's built-in Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    return len(faces) > 0


def main():
    #open a directory and check if the image is a profile picture
    directory = "datasets/profile_picture_data/profile_pictures"
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return
    # Iterate through all files in the directory
    nb_files = len(os.listdir(directory))
    nb_pp = 0
    print(f"Number of files in the directory: {nb_files}")
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            if is_profile_picture(image_path):
                nb_pp += 1
                print(f"{filename} is likely a profile picture.")
            else:
                print(f"{filename} is not a profile picture.")
    print(f"Number of profile pictures found: {nb_pp} out of {nb_files} files. This is {nb_pp/nb_files*100:.2f}% of the files.")
    print("Done processing all images.")

main()