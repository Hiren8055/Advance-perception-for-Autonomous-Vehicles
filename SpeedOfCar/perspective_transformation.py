import cv2
import numpy as np

TOP_VIEW_IMAGE_DIMESNION = (416, 416) # inv map output-image size (w, h) = (x, y)

FRONT_VIEW_IMAGE_DIMESNION = (416, 416) # (w, h) = (x, y)
# if transformation is changed then localization points will also change
# [(196.4206181381147, 265.87761926867154), (74.09867828295815, 416.0), (236.31060680787974, 265.87761926867154), (369.75272301350503, 416.0)]
FRONT_VIEW_POINTS = [(196.4206181381147, 265.87761926867154), (74.09867828295815, 416.0), (236.31060680787974, 265.87761926867154), (369.75272301350503, 416.0)]

# Dependent constants
TOP_VIEW_POINTS = [(0, 0),
          	   (0, TOP_VIEW_IMAGE_DIMESNION[1]),
          	   (TOP_VIEW_IMAGE_DIMESNION[0], 0), 
        	   (TOP_VIEW_IMAGE_DIMESNION[0], TOP_VIEW_IMAGE_DIMESNION[1])]

M = cv2.getPerspectiveTransform( np.float32(FRONT_VIEW_POINTS), np.float32(TOP_VIEW_POINTS) )


def convertBack(x, y, w, h):
    """
    Converts detections output into x-y coordinates
    :x, y: position of bounding box
    :w, h: height and width of bounding box
    """
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def inv_map(frame):
    """
    Transforms given image to top-view image (used for visual debug)
    :frame: front-view image
    :returns: transformation matrix and transformed image
    """
    image = cv2.warpPerspective(frame, M, TOP_VIEW_IMAGE_DIMESNION, flags=cv2.INTER_LINEAR)
    #cv2.imshow('itshouldlookfine!', image)
    return image, M

def get_inv_coor(detections):
    """
    Converts front-view coordinates (of cone) to top-view coordinates
    :detections: front-view coordinates
    :M: transformation matrix
    :returns: top-view coordinates of cones and person
    """
    mybox = []
    person = []
    for detection in detections:
        x, y, w, h = detection
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        # print(type(detection[0]))
        #person.append( ( (xmin+xmax)//2,(ymax) ) )
        a = np.array([[((xmax+xmin)//2), (ymax//1)]], dtype='float32')
        a = np.array([a])
        pointsOut = cv2.perspectiveTransform(a, M)
        box = int(pointsOut[0][0][0]), int(pointsOut[0][0][1])
        # print(detection[0])
        # if(detection[0].decode() == 'person'):
        #    person.append(box)
        # elif(box[1]>0):
        mybox.append(box)

    mybox = sorted(mybox, key=lambda k: (k[1], k[0])).copy()
    # print(mybox[::-1],'\n')

    # return person, mybox[::-1]
    return person, mybox


# _ , topview_coor = get_inv_coor()