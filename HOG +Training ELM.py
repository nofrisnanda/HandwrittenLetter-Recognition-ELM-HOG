from cv2 import *
import os
import random
import numpy as np
import sys
import math
from numpy.linalg import inv
import xlsxwriter

np.set_printoptions(threshold=sys.maxsize)

listGambar = []

def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = gray
    width = 60
    height = 60
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    image = resized
    m = cv2.moments(image)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*width*skew], [0, 1, 0]])
    image = cv2.warpAffine(image, M, (width, height), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return image


# def deskew(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     image = gray
#     width = 40
#     height = 40
#     dim = (width, height)
#     resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
#     image = resized
#     return image

def matrix_target(target):
    j = 0
    sign = [599,1199,1799,2399,2999,3599,4199,4799,5399,5999,6599,7199,7799,8399,8999,9599,10199,10799,11399,11999,12599,13199,13799,14399,14999]
    for i in range (len(target)):
        target[i][j] += 1.0
        if i in sign:
            j += 1
    return target

def regression_matrix(input_array,input_hidden_weights,bias):
    input_array = np.array(input_array)
    input_hidden_weights = np.array(input_hidden_weights)
    bias = np.array(bias)
    regression_matrix = np.add(np.dot(input_array,input_hidden_weights),bias)
    return regression_matrix

def hidden_layer_matrix(regression_matrix):
    sigmoidal = [[0.0 for i in range(0,no_of_hidden_neurons)]for j in range(0,no_of_inputs)]
    for i in range(0,no_of_inputs):
        for j in range(0,no_of_hidden_neurons):
            sigmoidal[i][j] = (1.0)/(1+math.exp(-(regression_matrix[i][j])))
    return sigmoidal

def hht(h,i,c):
    konstanta = np.true_divide(i,c)
    hht1 = np.dot(h.transpose(), h)
    hht = np.add(konstanta, hht1)
    return hht

def pseudoinverse(hht,htrans):
    pseudo = np.dot(inv(hht), htrans)
    return pseudo

def masukexcel(data):
    workbook = xlsxwriter.Workbook('output data train.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0
    for row in range(len(data)):
        for col in range(len(data[0])):
            worksheet.write(row,col,data[row][col])
    workbook.close()

def masukexcel_bias(data):
    workbook = xlsxwriter.Workbook('bias.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0
    for row in range(len(data)):
        for col in range(len(data[0])):
            worksheet.write(row, col, data[row][col])
    workbook.close()

def masukexcel_inputweight(data):
    workbook = xlsxwriter.Workbook('input weight.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0
    for row in range(len(data)):
        for col in range(len(data[0])):
            worksheet.write(row, col, data[row][col])
    workbook.close()

for folder in range(26):
    fol = str(folder)
    directory = 'E:/1. Nofri/TA/program/huruf1/training_' + fol
    listening = os.listdir(directory)
    for infile in listening:
        path = 'E:/1. Nofri/TA/program/huruf1/training_' + fol + '/' + infile
        img = deskew(cv2.imread(path))

        winSize = (60, 60)
        blockSize = (30, 30)
        blockStride = (15, 15)
        cellSize = (30, 30)
        nbins = 9
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 1
        nlevels = 64
        signedGradients = True

        hog = cv2.HOGDescriptor(winSize , blockSize, blockStride, cellSize, nbins, derivAperture,
                                winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels,
                                signedGradients)
        descriptor = hog.compute(img)
        listpiksel = []
        for a in range(len(descriptor)):
            listpiksel.append(float(descriptor[a]))
            # print(len(descriptor))
        listGambar.append(listpiksel)
        print("folder ke ", folder, "current file : ", infile)

no_of_inputs = len(listGambar)
input_dim = len(listpiksel)
no_of_input_neurons = input_dim
no_of_hidden_neurons = 3000
no_of_output_neurons = 26
input_array = np.array(listGambar)
id = np.identity(no_of_hidden_neurons, dtype=float)
# weight = random.uniform(0,1)
# biases = random.uniform(0,1)

input_hidden_weights = [[random.uniform(0,1) for i in range(0,no_of_hidden_neurons)]for j in range(0,no_of_input_neurons)]
bias = [[random.uniform(0,1) for i in range(0,no_of_hidden_neurons)]for j in range(0,no_of_inputs)]
mtarget = [[0.0 for i in range(0,no_of_output_neurons)]for j in range(0,no_of_inputs)]
mat_target = matrix_target(mtarget)
hidden_matrix = np.array(hidden_layer_matrix(regression_matrix(input_array,input_hidden_weights,bias)))
htranspose = hidden_matrix.transpose()

# hht2 = hht(hidden_matrix, id, 1000)
hht3 = np.dot(htranspose,hidden_matrix)

pseudo = np.array(pseudoinverse(hht3, htranspose))

output_weight = np.array(np.dot(pseudo, mat_target))
#
masukexcel(output_weight)
masukexcel_inputweight(input_hidden_weights)
masukexcel_bias(bias)
# print(len(hidden_matrix))
# print(len(hidden_matrix[0]))
# print(len(pseudo))
# print(len(pseudo[0]))
# print(len(output_weight))
# print(len(output_weight[0]))
# print(len(input_array))
# print(len(input_array[0]))







