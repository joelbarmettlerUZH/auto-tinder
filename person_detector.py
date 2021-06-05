# import numpy as np
# from   object_detection.utils import ops as utils_ops
# import tensorflow as tf
# from PIL import Image

# PERSON_CLASS = 1
# SCORE_THRESHOLD = 0.5

# def run_inference_for_single_image(image, sess):
#     ops = tf.get_default_graph().get_operations()
#     all_tensor_names = {output.name for op in ops for output in op.outputs}
#     tensor_dict = {}
#     for key in [
#         'num_detections', 'detection_boxes', 'detection_scores',
#         'detection_classes', 'detection_masks'
#     ]:
#         tensor_name = key + ':0'
#         if tensor_name in all_tensor_names:
#             tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
#                 tensor_name)
#     if 'detection_masks' in tensor_dict:
#         # The following processing is only for single image
#         detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
#         detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
#         # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
#         real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
#         detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
#         detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
#         detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
#             detection_masks, detection_boxes, image.shape[1], image.shape[2])
#         detection_masks_reframed = tf.cast(
#             tf.greater(detection_masks_reframed, 0.5), tf.uint8)
#         # Follow the convention by adding back the batch dimension
#         tensor_dict['detection_masks'] = tf.expand_dims(
#             detection_masks_reframed, 0)
#     image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

#     # Run inference
#     output_dict = sess.run(tensor_dict,
#                            feed_dict={image_tensor: image})

#     # all outputs are float32 numpy arrays, so convert types as appropriate
#     output_dict['num_detections'] = int(output_dict['num_detections'][0])
#     output_dict['detection_classes'] = output_dict[
#         'detection_classes'][0].astype(np.int64)
#     output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
#     output_dict['detection_scores'] = output_dict['detection_scores'][0]
#     if 'detection_masks' in output_dict:
#         output_dict['detection_masks'] = output_dict['detection_masks'][0]
#     return output_dict


# def open_graph():
#     detection_graph = tf.Graph()
#     with detection_graph.as_default():
#         od_graph_def = tf.GraphDef()
#         with tf.gfile.GFile('ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb', 'rb') as fid:
#             serialized_graph = fid.read()
#             od_graph_def.ParseFromString(serialized_graph)
#             tf.import_graph_def(od_graph_def, name='')
#     return detection_graph


# def load_image_into_numpy_array(image):
#     (im_width, im_height) = image.size
#     return np.array(image.getdata()).reshape(
#         (im_height, im_width, 3)).astype(np.uint8)


# def get_person(image_path, sess):
#     img = Image.open(image_path)
#     image_np = load_image_into_numpy_array(img)
#     image_np_expanded = np.expand_dims(image_np, axis=0)
#     output_dict = run_inference_for_single_image(image_np_expanded, sess)

#     persons_coordinates = []
#     for i in range(len(output_dict["detection_boxes"])):
#         score = output_dict["detection_scores"][i]
#         classtype = output_dict["detection_classes"][i]
#         if score > SCORE_THRESHOLD and classtype == PERSON_CLASS:
#             persons_coordinates.append(output_dict["detection_boxes"][i])

#     w, h = img.size
#     for person_coordinate in persons_coordinates:
#         cropped_img = img.crop((
#             int(w * person_coordinate[1]),
#             int(h * person_coordinate[0]),
#             int(w * person_coordinate[3]),
#             int(h * person_coordinate[2]),
#         ))
#         return cropped_img
#     return None