from os import listdir, rename
from os.path import isfile, join
import tkinter as tk
from PIL import ImageTk, Image

IMAGE_FOLDER = "./images/unclassified"

images = [f for f in listdir(IMAGE_FOLDER) if isfile(join(IMAGE_FOLDER, f))]
unclassified_images = filter(lambda image: not (image.startswith("0_") or image.startswith("1_")), images)
current = None

def next_img():
    global current, unclassified_images
    try:
        current = next(unclassified_images)
    except StopIteration:
        root.quit()
    print(current)
    pil_img = Image.open(IMAGE_FOLDER+"/"+current)
    width, height = pil_img.size
    max_height = 1000
    if height > max_height:
        resize_factor = max_height / height
        pil_img = pil_img.resize((int(width*resize_factor), int(height*resize_factor)), resample=Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(pil_img)
    img_label.img = img_tk
    img_label.config(image=img_label.img)

def positive(arg):
    global current
    print("Positive")
    rename(IMAGE_FOLDER+"/"+current, IMAGE_FOLDER+"/1_"+current)
    next_img()

def negative(arg):
    global current
    print("Negative")
    rename(IMAGE_FOLDER + "/" + current, IMAGE_FOLDER + "/0_" + current)
    next_img()


if __name__ == "__main__":

    root = tk.Tk()

    img_label = tk.Label(root)
    img_label.pack()
    img_label.bind("<Button-1>", positive)
    img_label.bind("<Button-3>", negative)

    btn = tk.Button(root, text='Next image', command=next_img)

    next_img() # load first image

    root.mainloop()


