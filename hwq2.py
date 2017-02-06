import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import sys
import scipy.stats as st


#Function to return Gaussian filter
def gkern(kernlen=21, nsig=3):
    
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel






def convolve2D(img_mat,filter_mat):
	




	[img_height,img_width]  = img_mat.shape

	
	[filter_height, filter_width] = filter_mat.shape
	[row_offset,col_offset] = (np.multiply(0.5,filter_mat.shape)).astype(np.int)

	input_mat = np.zeros([img_height+filter_height-1,img_width+filter_width-1])
	[input_height,input_width] = input_mat.shape
	
	input_mat[row_offset:input_height-row_offset,col_offset:input_width-col_offset] = img_mat[0:img_height,0:img_width]
	#input_mat = input_mat.astype(np.uint8)

	#cv2.imshow('input_mat',input_mat)



	output_mat = np.copy(input_mat)


	 
	r = 0


	for m in xrange(0,input_height):
		c= 0
		for n in xrange(0,input_width):
			temp = 0

			for k in xrange(-row_offset,row_offset+1):
				for l in xrange(-col_offset,col_offset+1):
					if( (m-k >= 0 and m-k <input_height) and (n-l >= 0 and n-l <input_width) ):
						temp = temp + input_mat[m-k,n-l]*filter_mat[k+row_offset,l+col_offset]
					
					
			output_mat[m,n] = temp
			c = c+1
		r = r+1
	out_img = np.copy(img_mat)
	out_img = scipy.signal.convolve2d(filter_mat,img_mat)



	img_mat = img_mat.astype(np.uint8)
	out_img = out_img.astype(np.uint8)
	output_mat = output_mat.astype(np.uint8)
	#input_fft = input_fft.astype(np.uint8)
	#output_fft = output_fft.astype(np.uint8)

	cv2.imshow('InBuilt Conv2D',out_img)
	cv2.imwrite('out.jpg',out_img)

	cv2.imshow('input', img_mat)
	cv2.imwrite('inputgray.jpg',img_mat)

	cv2.imshow('my conv2D',output_mat)
	cv2.imwrite('outmine.jpg',output_mat)

	

	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return output_mat

	


def main(argv):
	input_img = cv2.imread(str(argv[0]))

	input_img = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)

	
	
	noise_sigma = input("Enter the sigma for gaussian noise:")
	noise = np.random.normal(0,noise_sigma,(input_img.shape))
	
	
	

	filter_size = input("Enter the size of the matrix:")
	filter_type = input("Filter Type: Enter 1 for Averaging. Enter 2 for Gaussian:")

	if filter_type == 1:
		filter_mat = np.ones([filter_size,filter_size])/(filter_size*filter_size)

	elif filter_type == 2:
		sigma = input("Enter variance for the Gaussian filter:")
		filter_mat = gkern(filter_size,sigma)
	else:
		print "Option not found"
		return 0

	print filter_mat


	input2 =  input_img + noise*255
	#input2 = out.astype(np.uint8)

	output = convolve2D(input2,filter_mat)
	



if __name__=="__main__":
	main(sys.argv[1:])
	cv2.destroyAllWindows();