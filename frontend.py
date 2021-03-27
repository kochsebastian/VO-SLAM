import numpy as np
import cv2

import itertools

from scipy.spatial import cKDTree

from skimage.measure import ransac
from frontend_help import  poseRt, fundamentalToRt,  EssentialMatrixTransform

RANSAC_RESIDUAL_THRES = 0.02
RANSAC_MAX_TRIALS = 100

def extract_features(img):
  extractor = cv2.ORB_create() # check other descriptors
  
  # detection
  pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)

  # extraction
  kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
  kps, des = extractor.compute(img, kps)

  # return pts and des
  return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des



def match_frames(f1, f2, kps1, kps2, d1, d2):
  bf = cv2.BFMatcher(cv2.NORM_HAMMING)
  matches = bf.knnMatch(d1, d2, k=2)

  # Lowe's ratio test
  ret = []
  idx1, idx2 = [], []
  idx1s, idx2s = set(), set()

  for m,n in matches:
    if m.distance < 0.75*n.distance:
      p1 = kps1[m.queryIdx]
      p2 = kps2[m.trainIdx]

      # be within orb distance 32
      if m.distance < 32:
        # keep around indices
        if m.queryIdx not in idx1s and m.trainIdx not in idx2s:
          idx1.append(m.queryIdx)
          idx2.append(m.trainIdx)
          idx1s.add(m.queryIdx)
          idx2s.add(m.trainIdx)
          ret.append((p1, p2))

 
  ret = np.array(ret)
  idx1 = np.array(idx1)
  idx2 = np.array(idx2)

  # fit matrix
  model, inliers = ransac((ret[:, 0], ret[:, 1]),
                          EssentialMatrixTransform,
                          min_samples=8,
                          residual_threshold=RANSAC_RESIDUAL_THRES,
                          max_trials=RANSAC_MAX_TRIALS)
  print("Matches:  %d -> %d -> %d -> %d" % (len(d1), len(matches), len(inliers), sum(inliers)))
  return idx1[inliers], idx2[inliers], fundamentalToRt(model.params)


if __name__ == "__main__":


  cap = cv2.VideoCapture('rgbd_dataset_freiburg2_desk-rgb.avi')

  for i in itertools.count(0):
      
      ret, new_frame = cap.read()
       


      #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      new_kps, new_des = extract_features(new_frame)
      if i > 0:
        idx1, idx2, Rt = match_frames(old_frame,new_frame,old_kps,new_kps,old_des,new_des)
      
      for f in new_kps:
        cv2.circle(new_frame,(int(f[0]),int(f[1])),3,255,-1)
      cv2.imshow('frame',new_frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
      old_frame = new_frame.copy()
      old_des = new_des.copy()
      old_kps = new_kps.copy()

  cap.release()
  cv2.destroyAllWindows()
