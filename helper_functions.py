import Detector_Descriptor_initializer as DDI
import Matcher

def Alignfunction(Detector_name,Descriptor_name,Matcher_name, ratio_test_threshold,imReference,imtoAlign):
    #I plan to wrap this around a try exception block like if some name is not their it will throw exception.
    
    #Convert images to grayscale
    im1Gray = cv2.cvtColor(imtoAlign,cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(imReference,cv2.COLOR_BGR2GRAY)

    #Selecting the type of detector and descriptor
    detector, descriptor = DDI.Initialize_detector_descriptor(Detector_name,Descriptor_name)
    
    #Detecting keypoints and finding descriptors for the corresponding keypoints.
    keypoints1 = detector.detect(im1Gray,None)
    keypoints2 = detector.detect(im2Gray,None)
    keypoints1,descriptors1 = descriptor.compute(im1Gray,keypoints1)
    keypoints2,descriptors2 = descriptor.compute(im2Gray,keypoints2)
    
    #Selecting which matcher to use
    Matcher = Matcher.Initialize_Matcher(Matcher_name)
    #Matches = Matcher.match(descriptors1,descriptors2,None)
    Matches = Matcher.knnMatch(descriptors1,descriptors2,k= 2)
    
    #Applying the ratio test and getting Good_Matches from Matches
    Good_Matches = Ratio_Test(Matches,ratio_test_threshold)
    
    #Extract location of good matches
    #points1 = np.zeros((len(Good_Matches),2), dtype=np.float32)
    #points2 = np.zeros((len(Good_Matches),2), dtype=np.float32)
    
    #for i, match in enumerate(Good_Matches):
        #points1[i, :] = keypoints1[match.queryIdx].pt
        #points2[i, :] = keypoints2[match.trainIdx].pt
    
    #Find Homography
    #h_parameters, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    
    #Using the Homography parameters to align imtoAlign
    #height, width, channels = imReference.shape
    #imRegistered = cv2.warpPerspective(imtoAlign, h_parameters, (width, height))
    src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in Good_Matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in Good_Matches ]).reshape(-1,1,2)
    h_parameters, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    height,width,channels = imReference.shape
    imRegistered = cv2.warpPerspective(imtoAlign,h_parameters,(width,height))
    

    
    return imRegistered, h_parameters




def Ratio_Test(Matches, threshold=0.7):
    Good_matches = []
    for m,n in Matches:
        if m.distance < threshold*n.distance:
            Good_matches.append(m)
    return Good_matches


def check_parameters():
    return
