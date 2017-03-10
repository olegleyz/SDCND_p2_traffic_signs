import numpy as np
from random import randint
import cv2

def shifting(img):
    """
    """
    rows,cols = img.shape[:2]
    shift_x = randint(0,8) - 4
    shift_y = randint(0,8) - 4
    M = np.float32([[1,0,shift_x],[0,1,shift_y]])
    return cv2.warpAffine(img,M,(cols,rows))

def rotation(img):
    """
    """
    rows,cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),randint(0,30)-15,1)
    #M = cv2.getRotationMatrix2D((cols/2,rows/2),randint(0,90)+45,1)
    #M = cv2.getRotationMatrix2D((cols/2,rows/2),randint(45,360-45),1)
    return cv2.warpAffine(img,M,(cols,rows))

def affine(img):
    """
    """
    rows,cols = img.shape[:2]
    if randint(0,1):
        pts1 = np.float32([[50,50],[200,50],[50,200]])*32/400
        pts2 = np.float32([[10,100],[200,50],[100,250]])*32/400
    else:
        pts1 = np.float32([[50,50],[200,50],[50,200]])*32/400
        pts2 = np.float32([[60,0],[200,50],[0,150]])*32/400    
    M = cv2.getAffineTransform(pts1,pts2)
    return cv2.warpAffine(img,M,(cols,rows))

def perspective(img):
    """
    """
    rows,cols = img.shape[:2]
    scale = randint(0,4)
    pts1 = np.float32([[scale,scale],[32-scale,scale],[scale,32-scale],[32-scale,32-scale]])
    pts2 = np.float32([[0,0],[32,0],[0,32],[32,32]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    return cv2.warpPerspective(img,M,(32,32))

def geom_transf(img):
    """
    """
#    choice = randint(0,3)
    choice = 2
    if choice == 0:
        return shifting(img)
    elif choice == 1:
        return perspective(img)
    elif choice == 2:
        return rotation(img)
    elif choice == 3:
        return affine(img)
#    return affine(rotation(perspective(shifting(img))))

# Data augmentation
def data_augmentation(signs_freq_list, n_classes, signs_dic, X_train, y_train):
    # Here I set the size of generated dataset.
    num = 1
    sum_freq = sum(signs_freq_list)
    max_freq = max(signs_freq_list)
    new_signs_count = num * max_freq
    size_au = max_freq * n_classes - sum_freq + (num - 1) * max_freq * n_classes

    X_train_au = np.empty([size_au,32,32,3], dtype='uint8')
    y_train_au = np.empty([size_au], dtype='uint8')

    ind=0
    for i in range(n_classes):
        for j in range(new_signs_count - signs_freq_list[i]):
            index_ = randint(0, len(signs_dic[i])-1)
            index = signs_dic[i][index_]        
            X_train_au[ind] = geom_transf(X_train[index])
            y_train_au[ind] = i
            ind += 1
    print (X_train_au.shape[0],"images were successfully generated")
    X_train_au = np.concatenate((X_train, X_train_au),axis=0)
    y_train_au = np.concatenate((y_train, y_train_au),axis=0)
    return X_train_au, y_train_au

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def clahe_transform (img):
    """
    Funtion takes an image as input and return a new image with
    applied adaptive histogram equalization
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    return cl1

def gray_arr(arr):
    arr_gray = np.empty([len(arr),32,32,1], dtype='uint8')
    for i in range(len(arr)):
        arr_gray[i] = np.reshape(grayscale(arr[i]), (32,32,1))
    return arr_gray

def equ_arr(arr):
    equ_arr = np.zeros_like(arr)
    for i in range(len(arr)):
        equ_arr[i] = np.reshape(cv2.equalizeHist(arr[i]), (32,32,1))
    return equ_arr

def clahe_arr(arr):
    clahe_arr = np.zeros_like(arr)
    for i in range(len(arr)):
        clahe_arr[i] = np.reshape(clahe_transform(arr[i]), (32,32,1))
    return clahe_arr

def norm_arr(arr):
    return (arr - 128)/128

def show():
    """
    Function visualize a traffic sign and all grayscale transformations
    """
    img = cv2.cvtColor(X_train[0], cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(img)
    clh = clahe_transform(img)
    equ_clh = clahe_transform(equ)
    clh_equ = cv2.equalizeHist(clh)
    names = ['Original','Grayscale','Equalization','CHLOE','EQU->CHLOE','CHLOE->EQU']
    plt.figure(figsize=(32,32))
    for i, var in enumerate([X_train[0],img,equ,clh,equ_clh,clh_equ]):
        ax=plt.subplot(161 + i)
        plt.grid(False)
        plt.imshow(var)
        ax.set_title(names[i],fontsize=30)

        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        
def data_pp(X, gray=0, norm=0):
    """
    Data preprocessing function. Takes an array of images, grayscale 
    and normalization flags as input and performs data transformation
    """
    X_pp = np.empty([len(X),32,32,1])
    # transform input to Gray
    if (gray==1):
        X_pp = np.empty([len(X),32,32,1])
        for i in range(len(X)):
            X_pp[i] = np.reshape(grayscale(X[i]), (32,32,1))
    else:
        X_pp = np.copy(X)
    #normalize input (X - 128)/128
    if (norm==1):
        X_pp = norm_arr(X_pp)
    return X_pp

def data_preprocessing(X_train,X_test):
    X_train_pp = data_pp(X_train, gray=1, norm=1)
    X_test_pp = data_pp(X_test, gray=1, norm=1)
    assert X_train_pp.shape[0] == X_train.shape[0]
    assert X_test_pp.shape[0] == X_test.shape[0]
    print ("Data preprocessed successfully")
    return X_train_pp, X_test_pp