import numpy as np
import cv2

#models:https://github.com/richzhang/colorization/tree/caffe/colorization/models
#points:https://github.com/richzhang/colorization/blob/caffe/colorization/resources/pts_in_hull.npy
#inspired by:https://github.com/opencv/opencv/blob/master/samples/dnn/colorization.py

prototxt_path='models/colorization_deploy_v2.prototxt'
model_path='models/colorization_release_v2.caffemodel'
kernel_path='models/pts_in_hull.npy'
image_path='lion_demo.jpg'

#net=neural network
net=cv2.dnn.readNetFromCaffe(prototxt_path,model_path)
points=np.load(kernel_path)

#basically loading all these points in our file 
#and then we are going to use them to create a blob
#these specifications are mentioned in opencv documnetation
points=points.transpose().reshape(2,313,1,1)
net.getLayer(net.getLayerId("class8_ab")).blobs=[points.astype(np.float32)]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs=[np.full([1,313],2.606,dtype="float32")]


#lab: lightness a* b*

#load the bw image
#normalize it
#change color scheme from bgr to lab
bw_image=cv2.imread(image_path)
#Why bgr and not rgb:  imread fn loads image in bgr
#normalize: values from 1 to 255 represented as 0 to 1
normalized=bw_image.astype("float32")/255.0
lab=cv2.cvtColor(normalized,cv2.COLOR_BGR2LAB)

#resize the image 224x 224
resized=cv2.resize(lab,(224,224))
#split the channels
#L: lightness
L=cv2.split(resized)[0]
#can play around with L value
L-=50
#feed that into neural network and get a and b
net.setInput(cv2.dnn.blobFromImage(L))
#output is a blob ab
ab=net.forward()[0,:,:,:].transpose((1,2,0))
#resize the ab to the original image size
ab=cv2.resize(ab,(bw_image.shape[1],bw_image.shape[0]))
#get the original lightness back
L=cv2.split(lab)[0]
#concatenate L original lightness with ab colors
colorized=np.concatenate((L[:,:,np.newaxis],ab),axis=2)

#convert from lab to bgr 
colorized=cv2.cvtColor(colorized,cv2.COLOR_LAB2BGR)
#scale it back ie un normalize it by multiplying with 255
colorized=(255.0 * colorized).astype("uint8")



# Set up the display windows with adjustable sizes
cv2.namedWindow("BW Image", cv2.WINDOW_NORMAL)
cv2.namedWindow("Colorized Image", cv2.WINDOW_NORMAL)

# Resize the display windows to your preferred size, e.g., 600x600 pixels
cv2.resizeWindow("BW Image", 600, 600)
cv2.resizeWindow("Colorized Image", 600, 600)
#outputs
cv2.imshow("BW Image",bw_image)
cv2.imshow("Colorized Image",colorized)
cv2.waitKey(0)
cv2.destroyAllWindows()

