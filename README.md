# CoralMultiCamExample
Using a Coral Edge TPU to run object detection/tracking on multiple RTSP cameras.  Python is not my first langauge. This is something I cobbled together as a proof of concept following a few Udemy intro courses on Python, computer vision, and machine learning. A long with a lot of Coral Edge TPU and OpenCV examples online.  Someone please fork this and make it better!<p>
Script loads RTSP feeds from five cameras and displays them all in one window. Runs object tracking against people and vehicles within a defined region of interest and notifies user via Pushover if an object of interest within a certain size range is tracked for a specified number of frames.  I wrote it because of my previous attepts and making my dumb IP cameras smarter would send continuous notifications if someone parked in the driveway for example.<p>
Thanks to the Edge TPU, it can run the inference on 4 cameras (displaying 5) at about 7-9 fps using the low-res substreams from the cameras on an Atomic Pi (Intel Atom CPU).<p>

![Screenshot](https://github.com/mattncsu/CoralMultiCamExample/blob/main/smartcam.png)
