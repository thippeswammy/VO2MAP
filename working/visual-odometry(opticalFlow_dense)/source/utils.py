import cv2
import numpy as np


def drawFrameFeatures(frame, prevPts, currPts, frameIdx):
    currFrameRGB = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    prevPts = np.asarray(prevPts, dtype=np.float32)
    currPts = np.asarray(currPts, dtype=np.float32)
    # print(f"prevPts shape: {prevPts.shape}, type: {prevPts.dtype}")
    # print(f"currPts shape: {currPts.shape}, type: {currPts.dtype}")
    if prevPts.shape != currPts.shape or prevPts.ndim != 2 or prevPts.shape[1] != 2:
        print(f"[ERROR] Invalid shapes: prevPts={prevPts.shape}, currPts={currPts.shape}")
        return
    for i in range(len(currPts)):
        curr_pt = tuple(map(int, currPts[i]))
        prev_pt = tuple(map(int, prevPts[i]))
        cv2.circle(currFrameRGB, curr_pt, radius=3, color=(200, 100, 0), thickness=-1)
        cv2.line(currFrameRGB, prev_pt, curr_pt, color=(200, 100, 0), thickness=1)
    cv2.putText(currFrameRGB, f"Frame: {frameIdx}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200))
    cv2.putText(currFrameRGB, f"Features: {len(currPts)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200))
    cv2.imshow("Frame with keypoints", currFrameRGB)


def drawTrajectory(trajMap, trackedPoints, groundtruthPoints, frame_no, len_trajMap):
    offset_draw = int(len_trajMap / 2)
    offset_draw_small = int(len_trajMap / 8)

    # Draw estimated trajectory (blue)
    x_est = int(trackedPoints[-1, 0] + offset_draw)
    z_est = int(trackedPoints[-1, 2] + offset_draw_small)
    if 0 <= x_est < len_trajMap and 0 <= z_est < len_trajMap:
        cv2.circle(trajMap, (x_est, z_est), 1, (255, 0, 0), 5)

    # Draw ground truth trajectory (white)
    x_gt = int(groundtruthPoints[-1, 0] + offset_draw)
    z_gt = int(groundtruthPoints[-1, 2] + offset_draw_small)
    if 0 <= x_gt < len_trajMap and 0 <= z_gt < len_trajMap:
        cv2.circle(trajMap, (x_gt, z_gt), 1, (255, 255, 255), 2)

    trajMap1 = cv2.flip(trajMap, 0)
    cv2.imshow('Trajectory', cv2.resize(trajMap1, (1000, 700)))


def drawTrajectory_vo(trajMap, trackedPoints, groundtruthPoints, frame_no, len_trajMap):
    offset_draw = int(len_trajMap / 2)
    offset_draw_small = int(len_trajMap / 8)

    # Draw estimated trajectory (blue)
    x_est = int(trackedPoints[-1, 0] + offset_draw)
    z_est = int(-trackedPoints[-1, 2] + offset_draw_small)
    if 0 <= x_est < len_trajMap and 0 <= z_est < len_trajMap:
        cv2.circle(trajMap, (x_est, z_est), 1, (255, 0, 0), 5)

    # Draw ground truth trajectory (white)
    x_gt = int(groundtruthPoints[-1, 0] + offset_draw)
    z_gt = int(groundtruthPoints[-1, 2] + offset_draw_small)
    if 0 <= x_gt < len_trajMap and 0 <= z_gt < len_trajMap:
        cv2.circle(trajMap, (x_gt, z_gt), 1, (255, 255, 255), 2)

    trajMap1 = cv2.flip(trajMap, 0)
    cv2.imshow('Trajectory', cv2.resize(trajMap1, (1000, 700)))
