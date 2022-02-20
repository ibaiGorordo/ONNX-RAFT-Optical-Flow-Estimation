import cv2
import time
import numpy as np
import onnx
import onnxruntime

from .utils import flow_to_image

class Raft():

	def __init__(self, model_path):

		# Initialize model
		self.initialize_model(model_path)

	def __call__(self, img1, img2):

		return self.estimate_flow(img1, img2)

	def initialize_model(self, model_path):

		self.session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

		# Get model info
		self.get_input_details()
		self.get_output_details()

	def estimate_flow(self, img1, img2):

		input_tensor1 = self.prepare_input(img1)
		input_tensor2 = self.prepare_input(img2)

		outputs = self.inference(input_tensor1, input_tensor2)
		
		self.flow_map = self.process_output(outputs)

		return self.flow_map

	def prepare_input(self, img):

		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		self.img_height, self.img_width = img.shape[:2]

		img_input = cv2.resize(img, (self.input_width,self.input_height))

		# img_input = img_input/255
		img_input = img_input.transpose(2, 0, 1)
		img_input = img_input[np.newaxis,:,:,:]        

		return img_input.astype(np.float32)

	def inference(self, input_tensor1, input_tensor2):

		# start = time.time()
		outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor1, 
													   self.input_names[1]: input_tensor2})

		# print(time.time() - start)
		return outputs

	def process_output(self, output): 

		flow_map = output[1][0].transpose(1, 2, 0)

		return flow_map

	def draw_flow(self):

		# Convert flow to image
		flow_img = flow_to_image(self.flow_map)

		# Convert to BGR
		flow_img = cv2.cvtColor(flow_img, cv2.COLOR_RGB2BGR)

		# Resize the depth map to match the input image shape
		return cv2.resize(flow_img, (self.img_width,self.img_height))

	def get_input_details(self):

		model_inputs = self.session.get_inputs()
		self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

		self.input_shape = model_inputs[0].shape
		self.input_height = self.input_shape[2]
		self.input_width = self.input_shape[3]

	def get_output_details(self):

		model_outputs = self.session.get_outputs()
		self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

		self.output_shape = model_outputs[0].shape
		self.output_height = self.output_shape[2]
		self.output_width = self.output_shape[3]

if __name__ == '__main__':
	
	from imread_from_url import imread_from_url

	# Initialize model
	model_path='../models/raft_things_iter20_480x640.onnx'
	flow_estimator = Raft(model_path)

	# Read inference image
	img1 = imread_from_url("https://github.com/princeton-vl/RAFT/blob/master/demo-frames/frame_0016.png?raw=true")
	img2 = imread_from_url("https://github.com/princeton-vl/RAFT/blob/master/demo-frames/frame_0025.png?raw=true")

	# Estimate flow and colorize it
	flow_map = flow_estimator(img1, img2)
	flow_img = flow_estimator.draw_flow()

	combined_img = np.hstack((img1, img2, flow_img))

	cv2.namedWindow("Estimated flow", cv2.WINDOW_NORMAL)
	cv2.imshow("Estimated flow", combined_img)
	cv2.waitKey(0)