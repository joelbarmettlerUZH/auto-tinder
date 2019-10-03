import os
import person_detector
import tensorflow as tf

command = """
python retrain.py --bottleneck_dir=tf/training_data/bottlenecks --model_dir=tf/training_data/inception --summaries_dir=tf/training_data/summaries/basic --output_graph=tf/training_output/retrained_graph.pb --output_labels=tf/training_output/retrained_labels.txt --image_dir=./images/classified --how_many_training_steps=50000 --testing_percentage=20 --learning_rate=0.001
"""

IMAGE_FOLDER = "./images/unclassified"
POS_FOLDER = "./images/classified/positive"
NEG_FOLDER = "./images/classified/negative"
LOVOO_FOLDER = "./images/lovoo"



if __name__ == "__main__":
    detection_graph = person_detector.open_graph()

    images = [f for f in os.listdir(IMAGE_FOLDER) if os.path.isfile(os.path.join(IMAGE_FOLDER, f))]
    positive_images = filter(lambda image: (image.startswith("1_")), images)
    negative_images = filter(lambda image: (image.startswith("0_")), images)
    lovoo_images = [f for f in os.listdir(LOVOO_FOLDER) if os.path.isfile(os.path.join(LOVOO_FOLDER, f))]

    with detection_graph.as_default():
        with tf.Session() as sess:

            for pos in lovoo_images:
                old_filename = LOVOO_FOLDER + "/" + pos
                new_filename = POS_FOLDER + "/" + pos[:-5] + ".jpg"

                if not os.path.isfile(new_filename):
                    print(">> Moving positive image: " + pos)
                    img = person_detector.get_person(old_filename, sess)
                    if not img:
                        continue
                    img = img.convert('L')
                    img.save(new_filename, "jpeg")


            for pos in positive_images:
                old_filename = IMAGE_FOLDER + "/" + pos
                new_filename = POS_FOLDER + "/" + pos[:-5] + ".jpg"

                if not os.path.isfile(new_filename):
                    print(">> Moving positive image: " + pos)
                    img = person_detector.get_person(old_filename, sess)
                    if not img:
                        continue
                    img = img.convert('L')
                    img.save(new_filename, "jpeg")


            for neg in negative_images:
                old_filename = IMAGE_FOLDER + "/" + neg
                new_filename = NEG_FOLDER + "/" + neg[:-5] + ".jpg"
                if not os.path.isfile(new_filename):
                    print(">> Moving negative image: " + neg)
                    img = person_detector.get_person(old_filename, sess)
                    if not img:
                        continue
                    img = img.convert('L')
                    img.save(new_filename, "jpeg")
