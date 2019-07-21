def Initialize_detector_descriptor(detectorname,descriptorname):
    #For detector part.
    switcher1 = {
        "ORB_detect": orb_initializer,
        "SIFT_detect": sift_initializer,
        "AKAZE_detect": akaze_initializer,
        "FAST_detect" : fast_initializer,
        "SURF_detect" : surf_initializer,
        "BRISK_detect" : brisk_initializer,
        "STAR_detect" : star_initializer
    }
    
    #For descriptor part
    switcher2 = {
        "ORB_descript": orb_initializer,
        "SIFT_descript": sift_initializer,
        "AKAZE_descript": akaze_initializer,
        "SURF_descript": surf_initializer,
        "BRISK_descript": brisk_initializer,
        "FREAK_descript" : freak_initializer,
        "BRIEF_descript" : brief_initializer
    }
    
    
    #Get the function name from dictionary for detector & descriptor
    detect_functn = switcher1.get(detectorname, default)
    descript_functn = switcher2.get(descriptorname, default)
    
    #Execute the function thus initializeing the required descriptor and detector.
    detector = detect_functn()
    descriptor = descript_functn()
    
    return detector,descriptor
    
        
    
##############################################################################
def orb_initializer():
    orb = cv2.ORB_create()
    #Also initialize the parameters for the detector/descriptor here
    return orb


def sift_initializer():
    sift = cv2.xfeatures2d.SIFT_create()
    #Also initialize the parameters for the detector/descriptor here
    return sift

def akaze_initializer():
    akaze = cv2.AKAZE_create()
    #Also initialize the parameters for the detector/descriptor here
    return akaze

def fast_initializer():
    fast = cv2.FastFeatureDetector_create()
    #Also initialize the parameters for the detector/descriptor here
    return fast

def surf_initializer():
    surf = cv2.xfeatures2d.SURF_create()
    return surf

def brisk_initializer():
    brisk = cv2.BRISK_create()
    return brisk

def star_initializer():
    star = cv2.xfeatures2d.StarDetector_create()
    return star

def freak_initializer():
    freak = cv2.xfeatures2d.FREAK_create()
    return freak

def brief_initializer():
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    return brief

def default():
    sift = cv2.xfeatures2d.SIFT_create()
    #Also initialize the parameters for the detector/descriptor here
    return sift
