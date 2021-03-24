import numpy as np
import cv2


def extract_features(img):
  extractor = cv2.ORB_create() # check other descriptors
  
  # detection
  pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)

  # extraction
  kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
  kps, des = extractor.compute(img, kps)

  # return pts and des
  return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des

if __name__ == "__main__":


  cap = cv2.VideoCapture('./test.mp4')

  while(True):
      
      ret, frame = cap.read()

      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      features, res = extract_features(gray)
      
      cv2.imshow('frame',res)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  cap.release()
cv2.destroyAllWindows()
