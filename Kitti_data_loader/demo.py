import cv2

from Kitti_data_loader.utils.kitti_loader import KITTILoader

# Load dataset
dataset = KITTILoader(r"D:\downloadFiles\2011_10_03_drive_0042_sync\2011_10_03\2011_10_03_drive_0042_sync", time_unit="ns")
data_ = dataset.get_data(0)
img, img_rel, oxts, oxts_rel = data_
cv2.imshow("img", img)
cv2.waitKey(0)
print("Image relative timestamp (ns):", img_rel)
print("OXTS relative timestamp (ns):", oxts_rel)
print("GPS lat/lon/alt:", oxts["lat"], oxts["lon"], oxts["alt"])

data_ = dataset.get_data(dataset.__len__() - 1)
img, img_rel, oxts, oxts_rel = data_
cv2.imshow("img", img)
cv2.waitKey(0)
print("Image relative timestamp (ns):", img_rel)
print("OXTS relative timestamp (ns):", oxts_rel)
print("GPS lat/lon/alt:", oxts["lat"], oxts["lon"], oxts["alt"])
