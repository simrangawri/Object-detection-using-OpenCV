###RED-BALL DETECTION AND TRACKING

#Importing necessary libraries to be used in the code
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import random

###**********************************************************************************************
###FUNCTIONS TO BE APPLIED OVER THE VIDEO TO CHECK ITS RESULTS(TASK-2)
###funtion to add noise of different categories on our video
def noisy(noise_typ,frame):
    if noise_typ == "gauss":
        row,col,ch = frame.shape
        mean = 0
        sigma = 30**1.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        gauss_noisy = frame + gauss
        gauss_noisy = np.uint8(np.clip(gauss_noisy,0,255))
        return gauss_noisy
    elif noise_typ == "s&p":
        row,col,ch = frame.shape
        number_of_pixels = random.randint(1000,10000)
        for i in range(number_of_pixels):
            y_coord = random.randint(0,row-1)
            x_coord = random.randint(0,col-1)
            frame[y_coord][x_coord] = 255
        number_of_pixels = random.randint(1000,10000)
        for i in range(number_of_pixels):
            y_coord = random.randint(0,row-1)
            x_coord = random.randint(0,col-1)
            frame[y_coord][x_coord] = 0
        frame = np.uint8(np.clip(frame,0,255))            
        return frame
    elif noise_typ == "poisson":
        row,col,ch = frame.shape
        mean = 0
        var = 30
        sigma = var**0.1
        poisson_noisy = np.random.poisson(frame/255.0*sigma)/sigma*255
        poisson_noisy = np.uint8(np.clip(poisson_noisy,0,255))
        return poisson_noisy
    else:
        print("ERROR in noise type name")

###function to denoise the frame to get optimal result
def denoisy(frame,noise_type):
    if noise_type == "s&p":
        frame = cv2.medianBlur(frame,5)
    elif noise_type == "poisson":
        frame = cv2.blur(frame,(5,5))
    elif noise_type == "gauss":
        frame = cv2.bilateralFilter(frame,9,75,75)
    else:
        print("ERROR in noise type name")
    return frame

###function to Blur image
def blurImage(frame,kernel):
    frame = cv2.GaussianBlur(frame,(kernel,kernel),0)
    return frame

###funtion for sharpening of image
def sharpen(frame):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])    
    frame = cv2.filter2D(src=frame,ddepth=-1,kernel=kernel)
    return frame

###function for illumination variation
def varyIllumination(effect,frame):
    intensity_matrix = np.ones(frame.shape,dtype="uint8")*60
    if effect == "bright" :
        frame = cv2.add(frame,intensity_matrix)
    elif effect == "dark" :
        frame = cv2.subtract(frame,intensity_matrix)
    else:
        print("ERROR")
    return frame

###********************************************************************************************************        

###TASK(1)

#To run this program through command line interface We are implementing Argument parsing
#Creating a parser(object)
parser=argparse.ArgumentParser()
#Adding arguments into the parser
parser.add_argument("-v","--video",help="path to the (optional) video file")
parser.add_argument("-b","--buffer",type=int,default=64,help="max buffer size")
#Create an variable args which will help in parsing the argument with a function 
args=vars(parser.parse_args())

# As we have to detect the Red ball now we will give the HSV lower and upper range for red color 
redLower = (172,100,100)
redUpper = (179,255,255)
#initializing the list for tracking the ball to make contrail
pts = deque(maxlen=args["buffer"])

#if the video path is not provided then webcam gets started and will capture live video
if not args.get("video",False):
    vs = VideoStream(src=0).start()
#otherwise the video provided will be used for the further procedure
else:
    vs = cv2.VideoCapture(args["video"])
# time given to the camera to warm up and get started
time.sleep(2.0)

#loop for iterating on video frames
while True:
    #grabbing current frame
    frame = vs.read()
    # handle the frame from VideoCapture or VideoStream
    frame = frame[1] if args.get("video",False) else frame
    #Conditon for the loop when Video(all frames) reaches end
    if frame is None:
        break

###***************************************************************
    #CALLING FUNCTIONS(*uncomment the peice of command which we want to implement over the video*)
    
    #Applying Sharpening function..........................(1)
    #frame = sharpen(frame)

    #Applying Gaussian Blur with kernel value given............................(2)
    #frame = blurImage(frame,21)
    
    #Applying Illumination variation function..................................(3)
    #frame = varyIllumination("bright",frame)
    #frame = varyIllumination("dark",frame)

    #Applying noise in our frame from noise function..........................(4)
    #frame = noisy("poisson",frame)
    #frame = noisy("gauss",frame)
    frame = noisy("s&p",frame)

    #Applying De-Noising to make detection possible in noisy frame............(5)
    #frame = denoisy(frame,"poisson")
    #frame = denoisy(frame,"s&p")
    #frame = denoisy(frame,"gauss")
    
###******************************************************************    
    #resizing the frame to get optimal output(reduce time taken to read each frame)
    frame = imutils.resize(frame,width=600)
    #Blur the frame to reduce high frequency noise and allow us to focus on the structural objects(ball) inside the frame
    blurred = cv2.GaussianBlur(frame,(11,11),0)
    #Changing the color code to HSV model from RGB
    hsv = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
    #creating a mask to detect our object 
    mask = cv2.inRange(hsv,redLower,redUpper)
    #now we will apply set of erosions and dialtions to reduce the blobs in masking.
    mask = cv2.erode(mask,None,iterations=2)   #removing boundaries of foreground object
    mask = cv2.dilate(mask,None,iterations=2)  #emphasizing on features of the object
    
    #making contour for our object
    cnts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    #finding maximum contour to find the area of the largest circle possible
    if len(cnts)>0:
        c = max(cnts,key=cv2.contourArea)
        #finding minimum enclosing circle to further calculate upon moments
        ((x,y),radius)=cv2.minEnclosingCircle(c)
        #once we have the moments then we calculate the centre(centroid) of the circle
        M = cv2.moments(c)
        center = (int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"])) #formula used to find the centre with the help of moments
        
        #providing some minimum value of radius for any object to be detected
        if radius > 1:
            #we are here making two circle
            #outer one for surrounding the ball
            #inner one for centroid
            cv2.circle(frame,(int(x),int(y)),int(radius),(0,255,255),2)
            cv2.circle(frame,center,5,(0,0,255),-1)
    #appending the point for the centre in our list maintained for the tracking portion
    pts.appendleft(center)
    
###TRACKING BALL and Drawing contrail
    
    #now we are continuing upon the list we have maintained for tracking
    for i in range(1,len(pts)):
        #condition where we have reached the end
        if pts[i-1]is None or pts[i] is None:
            continue
        #compute the thickness for the contrail
        thickness = int(np.sqrt(args["buffer"]/float(i+1))*2.5)
        #draw the line of the contrail
        cv2.line(frame,pts[i-1],pts[i],(0,0,255),thickness)
    #Exeuting the frame with Detection and tracking 
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
#print statement to check the working of program
print("EXECUTION SUCESSFULL")
#condition if we have no video then stop the execution
if not args.get("video",False):
    vs.stop()
else:
    vs.release()
cv2.destroyAllWindows()
        
    
###****************************************************************************************************************



