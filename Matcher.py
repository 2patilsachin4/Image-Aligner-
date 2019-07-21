##############################################################################3
##############################################################################
def Initialize_Matcher(Matcher_name):
    switcher = {
        "BRUTEFORCE_L1_NORM" : bruteforce_L1_initializer
        "BRUTEFORCE_L2_NORM" : bruteforce_L2_initializer
        "BRUTEFORCE_NORM_HAMMING" : bruteforce_norm_hamming_initializer
        "BRUTEFORCE_NORM_HAMMING2" : bruteforce_norm_hamming2_initializer
        "FLANNBASED" : flannbased_initializer
    }   
    #Get the matcher function    
    initializer_function = switcher.get(Matcher_name, default)
        
    #Execute the function
    Matcher = initializer_function()
    return Matcher


 def bruteforce_L1_initializer():
    Matcher = cv2.BFMatcher(cv2.NORM_L1,crossCheck = False)
    return Matcher


def bruteforce_L2_initializer():
    Matcher = cv2.BFMatcher(cv2.NORM_L2,crossCheck = False)
    return Matcher

def bruteforce_norm_hamming_initializer():
    Matcher = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck = False)
    return Matcher

def bruteforce_norm_hamming2_initializer():
    Matcher = cv2.BFMatcher(cv2.NORM_HAMMING2,crossCheck = False)
    return Matcher

def flannbased_initializer():
    #Set flann parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    Matcher = cv2.FlannBasedMatcher(index_params,search_params)
    return Matcher

def default():
    Matcher = cv2.BFMatcher(cv2.NORM_L1,crossCheck = False)
    return Matcher
#########################################################################
