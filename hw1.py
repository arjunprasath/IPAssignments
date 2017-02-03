import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import sys

def convolve2D(img_mat,filter_mat):
	img_mat = img_mat.astype(np.uint8)
	img_mat = cv2.cvtColor(img_mat,cv2.COLOR_BGR2GRAY)




	[img_height,img_width]  = img_mat.shape

	
	[filter_height, filter_width] = filter_mat.shape
	[row_offset,col_offset] = (np.multiply(0.5,filter_mat.shape)).astype(np.int)

	input_mat = np.zeros([img_height+filter_height-1,img_width+filter_width-1])
	[input_height,input_width] = input_mat.shape
	print input_mat.shape
	input_mat[row_offset:input_height-row_offset,col_offset:input_width-col_offset] = img_mat[0:img_height,0:img_width]
	input_mat = input_mat.astype(np.uint8)

	cv2.imshow('input_mat',input_mat)



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
					
					#print img_mat[0,0]

			#print 'm is '+str(m)+' n is '+str(n) 
			output_mat[m,n] = temp
			c = c+1
		r = r+1
	out_img = np.copy(img_mat)
	out_img = scipy.signal.convolve2d(filter_mat,img_mat)
	out_img = out_img.astype(np.uint8)
	output_mat = output_mat.astype(np.uint8)

	cv2.imshow('out_img',out_img)
	cv2.imwrite('out.jpg',out_img)
	cv2.imshow('input', img_mat)
	cv2.imshow('window',output_mat)
	cv2.imwrite('outmine.jpg',output_mat)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return output_mat

	


def main(argv):
	input_img = cv2.imread(str(argv[0]))
	#cv2.imshow("input image",input_img)
	#cv2.waitKey(0)

	filter_height = input("Enter number of rows of the filter:")
	filter_width  = input("Enter number of columns of the filter:")

	filter_mat = np.ones([filter_height,filter_width])
	

	for i in xrange(filter_height):
		for j in xrange(filter_width):
			filter_mat[i,j] = input("Enter the Element (" +str(i)+","+str(j)+") of the filter coefficient Matrix:" )
	print filter_mat

	output = convolve2D(input_img,filter_mat)



if __name__=="__main__":
	main(sys.argv[1:])
	cv2.destroyAllWindows();