import cv2, math, copy
import numpy as np
from datetime import datetime


def getManhattan(c1, c2):
    """
    return manhattan distance between c1 and c2
    :param c1: Coord [y,x] to compare with c2
    :param c2: Coord [y,x] to compare with c1
    :return: Manhattan distance between c1 and c2
    """
    return math.sqrt((c2[1] - c1[1]) ** 2 + (c2[0] - c1[0]) ** 2)


def findRedPoints(image, th=20000):
    """
    search red dot coords from bit_wised array
    :param image: numpy array that consisted of 0 and 1
    :param th: threshold that detecting red dot
    :return: return red coord such as [[y1,x1],[y2,x2],...,[yn,xn]]
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # calculate window size
    # if the window size is getting bigger, it needs less calculation
    w_kernel = 40
    h_kernel = 40

    # stack recognized red dot's points
    point_stack = []

    # first step window sliding
    # search red dots using window
    for i in range(0, hsv_image.shape[0], h_kernel):
        for j in range(0, hsv_image.shape[1], w_kernel):

            # sum of kernel values
            # means when kernel has values, that kernel includes red dot
            kernel_value = np.sum(hsv_image[i:i + h_kernel, j:j + w_kernel])

            # if kernel has values and sum of the value is more than 20,000 (threshold)
            # means red dots on that paper needs enough size to recognize
            if kernel_value != 0 and kernel_value > th:

                # trigger to escape double for loop
                found = False

                # second step window sliding
                # if the first step window sliding recognize red dots, search the precise coords of the dots
                for k in range(0, h_kernel, 4):
                    for l in range(0, w_kernel, 4):

                        # same as kernel_value
                        white_check = np.sum(hsv_image[i + k:i + k + 4, j + l:j + l + 4])

                        # if second step window has value (detect red dots), append coords to array and escape the
                        # double for loop
                        if white_check != 0:
                            found = True
                            ignore = False

                            if point_stack:
                                for p in point_stack:
                                    # distance between exist dots and new dots are less than 20 pixel
                                    if getManhattan([j + l + 1, i + k + 1], p) < 100:
                                        ignore = True
                                        break

                            if not ignore:
                                point_stack.append([j + l + 1, i + k + 1])  # [y,x]
                            break

                    if found:
                        break
    return point_stack


def sortRectangle(ps: list):
    """
    sort array for create accurate rectangle
    :param ps: array of points
    :return: sorted array
    """
    top = ps[:2]
    bottom = ps[2:]
    top.sort(key=lambda x: x[0])
    bottom.sort(key=lambda x: -x[0])
    return top + bottom


def wrap_image(img: list, from_pts: list, to_pts: list, w: int, h: int):
    """
    return perspect image form origin images using points
    :param img: Original image
    :param from_pts: list of red dot points
    :param to_pts: target image points
    :param w: target image width
    :param h: target image height
    :return: return cv2 image with (w,h)
    """
    perspect_mat = cv2.getPerspectiveTransform(from_pts, to_pts)
    dst = cv2.warpPerspective(img, perspect_mat, (h, w), cv2.INTER_CUBIC)
    return dst


def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        command = (x // 60)

        if command == 0:  # Lock
            CFG['Lock'] = not(CFG['Lock'])
        elif command == 1:  # Rotate the Result screen
            CFG['Rotate'] = (CFG['Rotate'] + 1) % 4
        elif command == 2:  # View origin frame
            CFG['View Origin'] = not(CFG['View Origin'])
        elif command == 3:
            CFG['Save Image'] = True


# Settings
CFG = {
    'Display rate': 0.55,  # Rate of frame size / original size
    'Threshold': 2000,  # Threshold to detect red dots
    'TEST': False,  # if true : Set mode to test mode
    'Rotate': 0,
    'Lock': False,
    'View Origin': False,
    'Save Image': False
}

if __name__ == '__main__':
    capture = cv2.VideoCapture(0)

    # If camera does not exist or not working, raise error
    if not capture.isOpened():
        raise Exception('Camera does not exist')

    # setting the threshold of HSV to recognize red dots
    # lower_red = np.array([-30, 100, 100])
    # upper_red = np.array([30, 255, 255])

    # Control pannel
    btn_image = cv2.imread('./Buttons.png')
    cv2.imshow('Control Panel', btn_image)
    cv2.setMouseCallback('Control Panel', onMouse)

    # lower_red = np.array([0, 50, 50])
    # upper_red = np.array([10, 255, 255])

    # Saving red dots position
    dots_cash = []
    perspect_cash = []
    width_cash = 0
    height_cash = 0

    origin_window_toggle = False

    while True:
        ret, frame = capture.read()
        original_frame = copy.deepcopy(frame)

        if not ret or cv2.waitKey(30) >= 0:
            break

        # Get original image frame size
        frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        img_mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        img_result = cv2.bitwise_and(frame, frame, mask=img_mask)

        points = sortRectangle(findRedPoints(img_result, CFG['Threshold']))
        # points = findRedPoints(img_result, CFG['Threshold'])
        perspect_frame = ''

        for point in points:
            cv2.line(frame, point, point, (0, 0, 255), 10)

        if len(points) == 4 and not(CFG['Lock']):
            cv2.polylines(frame, [np.array(points)], True, (255, 0, 255), 2)

            # ===== TEST =====
            if CFG['TEST']:
                for i, p in enumerate(points):
                    cv2.putText(frame, str(i + 1), p, 1, 3, (98, 17, 0), 2, cv2.LINE_AA)
            # ===== TEST =====

            #  Convert to perspect image

            width = int((getManhattan(points[0], points[1]) + getManhattan(points[2],points[3])) / 2)
            height = int((getManhattan(points[0], points[3]) + getManhattan(points[1], points[2])) / 2)

            width_cash, height_cash = width, height

            pts1 = np.float32(points)
            pts2 = np.float32([(0, 0), (width, 0), (width, height), (0, height)])

            dots_cash = pts1
            perspect_cash = pts2

            perspect_frame = wrap_image(original_frame, pts1, pts2, height, width)

            if CFG['Rotate'] == 1:
                perspect_frame = cv2.rotate(perspect_frame, cv2.ROTATE_90_CLOCKWISE)
            elif CFG['Rotate'] == 2:
                perspect_frame = cv2.rotate(perspect_frame, cv2.ROTATE_180)
            elif CFG['Rotate'] == 3:
                perspect_frame = cv2.rotate(perspect_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # ===== TEST =====
            if CFG['TEST']:
                cv2.putText(perspect_frame, f'width : {width}, Height : {height}', (20, 20), 1, 1, (98, 17, 0), 2, cv2.LINE_AA)
            # ===== TEST =====

            cv2.imshow('Result', perspect_frame)

        if CFG['Lock']:
            perspect_frame = wrap_image(original_frame, dots_cash, perspect_cash, height_cash, width_cash)

            if CFG['Rotate'] == 1:
                perspect_frame = cv2.rotate(perspect_frame, cv2.ROTATE_90_CLOCKWISE)
            elif CFG['Rotate'] == 2:
                perspect_frame = cv2.rotate(perspect_frame, cv2.ROTATE_180)
            elif CFG['Rotate'] == 3:
                perspect_frame = cv2.rotate(perspect_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            cv2.imshow('Result', perspect_frame)

        if CFG['Save Image']:
            now = datetime.now()
            cv2.imwrite(f'{int(now.timestamp())}.png', perspect_frame)
            print(f"Image saved '{int(now.timestamp())}.png'")
            CFG['Save Image'] = False

        if CFG['View Origin']:
            cv2.imshow('Origin Image', cv2.resize(frame, (int(frame_width * CFG['Display rate']), int(frame_height * CFG['Display rate']))))
            origin_window_toggle = True
        elif not(CFG['View Origin']) and origin_window_toggle:
            cv2.destroyWindow('Origin Image')
            origin_window_toggle = False

        # ===== TEST =====
        if CFG['TEST']:
            cv2.imshow('Final frame',cv2.resize(hsv, (int(frame_width * CFG['Display rate']), int(frame_height * CFG['Display rate']))))
            cv2.imshow('Frame',cv2.resize(frame, (int(frame_width * CFG['Display rate']), int(frame_height * CFG['Display rate']))))
        # ===== TEST =====
