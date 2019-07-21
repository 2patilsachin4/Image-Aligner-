#Import the required libraries
import cv2
import numpy as np
import helper_functions

def Image_Aligner(Detector_name, Descriptor_name, Matcher_name,ratio_test_threshold,referenceimgname, alignmentimgname):
    
    #Check whether the detector & descriptor pair is valid or not.
    #Valid = check_parameters()
    #if Valid == False:
        #print("The Descriptor and Detector pairing do not match")
        #return
        
    
    #Read reference image
    print("Reading referenceimage: ", referenceimgname)
    imReference = cv2.imread(referenceimgname, cv2.IMREAD_COLOR)
    
    #Reading image to be aligned
    print("Reading image to align : ", alignmentimgname)
    imtoAlign = cv2.imread(alignmentimgname, cv2.IMREAD_COLOR)
    
    print("Alligning Images ...")
    
    #Aligning the images by applying alignment function.
    #Registered image will be put in imRegistered and homography in h_parameters.
    imRegistered, h_parameters = helper_functions.Alignfunction(Detector_name,Descriptor_name,Matcher_name,imReference,imtoAlign)
    
    #Writing the aligned image to disk
    outFilename = "Aligned.jpg"
    print("Saving aligned image: ", outFilename)
    cv2.imwrite(outFilename, imRegistered)
    
    #Displaying the Aligned image & Reference image side by side for comparison.
    #here print fucntion will come edit it.
    
    #Printing homography parameters
    print("Estimated homography: \n", h_parameters)
    
if __name__ == '__main__':
    Detector_name = input("Enter the detector name: ")
    Descriptor_name = input("Enter the descriptor name: ")
    Matcher_name = input("Enter the matcher name: ")
    ratio_test_threshold = int(input("Enter the ratio_test_threshold value to use: "))
    ref_image = input("Enter the name of reference image in .jpg extension: ")
    align_image = input("Enter the name of image to be aligned in .jpg extension: ")
    Image_Aligner(Detector_name, Descriptor_name, Matcher_name,ratio_test_threshold,ref_image, align_image)    
