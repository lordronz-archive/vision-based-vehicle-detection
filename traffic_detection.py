import cv2
import numpy as np
import imutils
from coco import coco

def main():
    net = cv2.dnn.readNetFromDarknet('./yolov4-tiny.cfg','./yolov4-tiny.weights')

    classes = coco

    np_argmax = np.argmax
    np_random_uniform = np.random.uniform

    font = cv2.FONT_HERSHEY_PLAIN

    cv2_dnn_blobFromImage = cv2.dnn.blobFromImage
    cv2_dnn_NMSBoxes = cv2.dnn.NMSBoxes
    cv2_resize = cv2.resize
    cv2_imshow = cv2.imshow
    cv2_waitKey = cv2.waitKey
    net_getUnconnectedOutLayersNames = net.getUnconnectedOutLayersNames
    net_forward = net.forward
    net_setInput = net.setInput
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    vid_in = './traffic.mp4'
    vid_out = './output.avi'

    COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")

    VEHIC_CNT_COLOR = [0, 0, 255]

    vs = cv2.VideoCapture(vid_in)
    writer = None
    W, H = None, None
    # try to determine the total number of frames in the video file
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        print("[INFO] {} total frames in video".format(total))
    # an error occurred while trying to determine the total
    # number of frames in the video file
    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1

    while True:
        grabbed, frame = vs.read()

        if not grabbed:
            break

        # img = cv2_resize(img, (960, 720))

        if not W or not H:
            H, W = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)


        layer_outputs = net_forward(ln)

        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                score = detection[5:]
                class_id = np_argmax(score)
                confidence = score[class_id]
                if confidence > .4 and class_id in range(1, 9):
                    center_x = int(detection[0] * W)
                    center_y = int(detection[1] * H)
                    w = int(detection[2] * W)
                    h = int(detection[3] * H)

                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    boxes.append((x,y,w,h))
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2_dnn_NMSBoxes(boxes, confidences, .4, .4)

        if len(indexes):
            for i in indexes.flatten():
                x,y,w,h = boxes[i]
                label = classes[class_ids[i]]
                confidence = round(confidences[i],2)
                color = [int(c) for c in COLORS[class_ids[i]]]
                cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)

        cv2.putText(frame, f'{len(indexes)} vehicles', (W//10, H//10) ,font,2,VEHIC_CNT_COLOR,2)

        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(vid_out, fourcc, 30,
                (frame.shape[1], frame.shape[0]), True)


        writer.write(frame)

    writer.release()
    vs.release()

if __name__ == '__main__':
    main()
