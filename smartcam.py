#!/usr/bin/env python3
# threaded version of smartcam
from pycoral.adapters import common #https://github.com/google-coral/pycoral/blob/master/examples/detect_image.py
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from imutils.video import VideoStream
from PIL import Image
from threading import Thread
from centroidtracker import CentroidTracker #https://github.com/lev1khachatryan/Centroid-Based_Object_Tracking
from pushover import Client #https://pypi.org/p25roject/python-pushover/
from sys import exit
import imutils
import time
import cv2
import numpy as np

stop = False

red=(0,0,255) #BGR
green=(0,255,0)
blue=(255,0,0)

noImage = np.zeros((530, 300, 3), np.uint8)
cv2.putText(noImage, "No Image!", (150, 150),cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)

class VideoStreamWidget(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)#, cv2.CAP_INTEL_MFX)
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        

    def update(self):
        # Read the next frame from the stream in a different thread
        while not stop:
            if self.capture.isOpened():
                try:
                    (self.status, self.frame) = self.capture.read()
                except:
                    self.frame = noImage
                #self.frame = self.image #cv2.resize(self.image, (640, 480))
            time.sleep(.01)
            
    def get_frame(self):
        return self.frame
        
    def stop(self):
        # Display frames in main program
        self.capture.release()
        stop = True


#initialize pushover
client = Client("xxxxxxx", api_token="yyyyyy")

#the arguments
label_input='/home/atomicpi/Desktop/smartcam/tflite/python/examples/detection/models/coco_labels.txt'
model_input='/home/atomicpi/Desktop/smartcam/tflite/python/examples/detection/models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'

#label_input='/home/atomicpi/Desktop/smartcam/models/MobileDetSSD/coco_labels.txt'
#model_input='/home/atomicpi/Desktop/smartcam/models/MobileDetSSD/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite'

#label_input='/home/atomicpi/Desktop/smartcam/models/MobileNetSSDv1/coco_labels.txt'
#model_input='/home/atomicpi/Desktop/smartcam/models/MobileNetSSDv1/ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite'

#label_input='/home/atomicpi/Desktop/smartcam/models/custom_model/labels.txt'
#model_input='/home/atomicpi/Desktop/smartcam/models/custom_model/model_edgetpu.tflite'

notifyFrames=3 #consecutive frames of detection before notification
confidence=[.51, .40, .5, .6, .6, .6] #required confidence per camera, see cameraNames for camera order
tracked_labels = [0, 1, 2, 3, 5, 7] #person, bike, car, motorcycle, bus, truck
timeLimit = [60, 60, 180, 300, 60] #seconds to wait between consecutive notifications on each camera


camera_streams=[]
camera_streams.append('rtsp://user:pass@192.168.1.155/Streaming/channels/102') #frontdoor
camera_streams.append('rtsp://user:pass@192.168.1.155/Streaming/channels/202') #driveway
camera_streams.append('rtsp://user:pass@192.168.1.155/Streaming/channels/302') #pool
camera_streams.append('rtsp://user:pass@192.168.1.155/Streaming/channels/702') #trees
camera_streams.append('rtsp://user:pass@192.168.1.155/Streaming/channels/802') #Windsor
cameraNames = {
    0 : "Front Door",
    1 : "Driveway",
    2 : "Patio",
    3 : "Back Yard",
    4 : "Windsor Dr"
    }

monitored_cams = [0, 1, 2, 3] #only run object detection on some cameras, all cameras show on display



# initialize our centroid tracker and frame dimensions
#ct = CentroidTracker()
ct=[]
# Initialize the TF interpreter
interpreter = make_interpreter(model_input)
interpreter.allocate_tensors()
print(common.input_size(interpreter))

#Define ROIs within each video stream, color rectangle based on ROI, notify if inside ROI
#SEE: https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python

ROIcontours=[]
#FRONT DOOR
contour=np.array([
    [0,250],
    [300,70],
    [400,50],
    [704,30],
    [704,480],
    [0,480]],np.int32)
contour=contour.reshape((-1,1,2))
ROIcontours.append(contour)

#DRIVEWAY
contour=np.array([
    [0,355],
    [0,150],
    [110,75],
    [310,70],
    [400,80],
    [600,125],
    [610,355]],np.int32)
contour=contour.reshape((-1,1,2))
ROIcontours.append(contour)
#POOL
contour=np.array([
    [400,0],
    [535,0],
    [535,100],
    [400,100],
    ],np.int32)
contour=contour.reshape((-1,1,2))
ROIcontours.append(contour)

#TREES
contour=np.array([
    [0,480],
    [0,220],
    [260,90],
    [450,60],
    [650,90],
    [650,150],
    [704,150],
    [704,480]],np.int32)
contour=contour.reshape((-1,1,2))
ROIcontours.append(contour)

#WINDSOR
contour=np.array([
    [550,360],
    [550,200],
    [300,150],
    [550,65],
    [640,75],
    [640,360]],np.int32)
contour=contour.reshape((-1,1,2))
ROIcontours.append(contour)
draw_contours = True

#Open camera streams in new threads
print("[INFO] starting video streams...")
vs = []
for x, streams in enumerate(camera_streams):
    vs.append(VideoStreamWidget(src=camera_streams[x]))
    print(f'   Stream {x}:{cameraNames[x]} started.')
    # initialize our centroid tracker and frame dimensions
    ct.append(CentroidTracker())
time.sleep(2)
#Get first frames
images=[]
noticeTime=[]
for x,stream in enumerate(vs):
    try:
        images.append(vs[x].get_frame())
    except:
        images.append(noImage)
    noticeTime.append(time.time())

# initialize the labels dictionary
print("[INFO] parsing class labels...")
labels = {}
# loop over the class labels file
for row in open(label_input):
    # unpack the row and update the labels dictionary
    (classID, label) = row.strip().split(maxsplit=1)
    labels[int(classID)] = label.strip()

print('Tracking the following labels:')
for label in tracked_labels:
    print(labels.get(label))

#Functions to concat the camera frames for display  https://note.nkmk.me/en/python-opencv-hconcat-vconcat-np-tile/
def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = 1920 #min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

def concat_tile_resize(im_list_2d, interpolation=cv2.INTER_CUBIC):
    im_list_v = [hconcat_resize_min(im_list_h, interpolation=interpolation) for im_list_h in im_list_2d]
    return vconcat_resize_min(im_list_v, interpolation=interpolation)

def sendPushover(x, attachment):
    #noticeTime, timeLimit
    now = time.time()
    elapsed = (now-noticeTime[x])
    print(f'{time.asctime()} ***Sending {cameraNames[x]} Alert. {elapsed} seconds since last notification.')
    if elapsed > timeLimit[x]:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        success, imgFile = cv2.imencode('.jpg', images[x], encode_param)
        imAttach = imgFile.tostring()
        client.send_message("Motion Detected", title=f'{cameraNames[x]} Alert', priority = 1, sound="classical", attachment=imAttach)
        noticeTime[x] = time.time()

# loop over the frames from the video stream
while True:
    fps = time.time()
    # grab the frame from the threaded video stream and resize it
    for x,stream in enumerate(vs):
        try:
            images[x]=vs[x].get_frame()
        except:
            images[x]=noImage
        # prepare the frame for object detection by converting (1) it
        # from BGR to RGB channel ordering and then (2) from a NumPy
        # array to PIL image format
        #images[x] = cv2.cvtColor(images[x], cv2.COLOR_BGR2RGB)
        if x in monitored_cams:

            #pre-scale in OpenCV for speed
            try:
                imagesP = cv2.resize(images[x], dsize=common.input_size(interpreter), interpolation=cv2.INTER_LINEAR)
            except:
                print("Couldn't resize for some reason (line 243)")
                imagesP = cv2.resize(noImage, dsize=common.input_size(interpreter), interpolation=cv2.INTER_LINEAR)
            imagesP = Image.fromarray(images[x])
            scale = common.set_resized_input(interpreter, imagesP.size, lambda size: imagesP.resize(size, Image.NEAREST)) #this line crashes

            # make predictions on the input frame
            interpreter.invoke()
            results = detect.get_objects(interpreter, confidence[x], scale[1])

        else: #return empty list if camera isn't being monitored
            results = []

        if draw_contours:
            cv2.polylines(images[x],[ROIcontours[x]],isClosed=True,color=blue,thickness=1)
        # loop over the results

        if x not in monitored_cams:
            cv2.putText(images[x], "Detection OFF", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
        #TODO: Consolidate for loops in ln 215-260 into single for loop
        rects = []
        for r in results:
            if r.id in tracked_labels:
                # extract the bounding box and box and predicted class label
                box = r.bbox

                #print(box)
                (startX, startY, endX, endY) = box
                box_center = ((endX+startX)//2,(endY+startY)//2)
                label = labels.get(r.id)
                im_height, im_width, channels = images[x].shape
                if ((endX-startX)*(endY-startY)//(im_height*im_width)) < 0.33: #screen out objects bigger than 1/3 the screen
                    notTooBig = True
                else:
                    notTooBig = False

                #check if midpoint of bounding box is inside ROI
                if cv2.pointPolygonTest(ROIcontours[x],box_center,False) > 0 and notTooBig:
                    cv2.rectangle(images[x], (startX, startY), (endX, endY), red, 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    text = "{}: {:.1f}%".format(label, r.score * 100)
                    cv2.putText(images[x], text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
                    rects.append(box)
                # draw the bounding box and label on the image
                else:
                    cv2.rectangle(images[x], (startX, startY), (endX, endY), green, 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    text = "{}: {:.1f}%".format(label, r.score * 100)
                    cv2.putText(images[x], text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)

        # update our centroid tracker using the computed set of bounding
        # box rectangles
        objects = ct[x].update(rects)

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {} - {}".format(objectID, ct[x].frameCount[objectID])
            cv2.putText(images[x], text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            text = "{:.1f}%".format((endX-startX)*(endY-startY)/(im_height*im_width) * 100)
            cv2.putText(images[x], text, (centroid[0] - 10, centroid[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #send notice if same object has been tracked for notifyFrames
            if ct[x].frameCount[objectID] > notifyFrames:
                cv2.circle(images[x], (centroid[0], centroid[1]), 4, (0, 255, 255), -1)
                if ct[x].noticeSent[objectID] == False:
                    sendPushover(x,images[x])
                    ct[x].noticeSent[objectID] = True
            else:
                cv2.circle(images[x], (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # show the output frame and wait for a key press
    try:
        im_tile_resize = concat_tile_resize([[images[0], images[1]],
                                        [images[2], images[3], images[4]]],interpolation=cv2.INTER_NEAREST)
        cv2.putText(im_tile_resize, str(int(1//(time.time()-fps))), (5,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
    except:
        print("Error resizing on line 317")
    #cv2.putText(im_tile_resize, str(int(model.get_inference_time())), (1,22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
    cv2.imshow("Frame", im_tile_resize)
    #fps.update()
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
for x, streams in enumerate(camera_streams):
    print(f'stopping {x}')
    vs[x].stop()
