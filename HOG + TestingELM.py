from cv2 import *
import numpy as np
import sys
import math
import xlrd
import xlsxwriter
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

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

def bobotoutput():
    output_weight = [[0.0 for i in range(0, no_of_output_neurons)] for j in range(0, no_of_hidden_neurons)]
    Workbook = xlrd.open_workbook("output data train.xlsx")
    Worksheet = Workbook.sheet_by_index(0)
    for i in range(0,no_of_hidden_neurons):
        for j in range(0,no_of_output_neurons):
            output_weight[i][j] = Worksheet.cell_value(i,j)
    return output_weight

def bobotinput():
    output_weight = [[0.0 for i in range(0, no_of_hidden_neurons)] for j in range(0, no_of_input_neurons)]
    Workbook = xlrd.open_workbook("input weight.xlsx")
    Worksheet = Workbook.sheet_by_index(0)
    for i in range(0,no_of_input_neurons):
        for j in range(0,no_of_hidden_neurons):
            output_weight[i][j] = Worksheet.cell_value(i,j)
    return output_weight
#
def inputbias():
    output_weight = [[0.0 for i in range(0, no_of_hidden_neurons)] for j in range(0, no_of_inputs)]
    Workbook = xlrd.open_workbook("bias.xlsx")
    Worksheet = Workbook.sheet_by_index(0)
    for i in range(0,no_of_inputs):
        for j in range(0,no_of_hidden_neurons):
            output_weight[i][j] = Worksheet.cell_value(i,j)
    return output_weight

def masukexcel_prediksi(data):
    workbook = xlsxwriter.Workbook('yprediksi.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0
    for row in range(len(data)):
        for col in range(len(data[row])):
            worksheet.write(row,col,data[row][col])
    workbook.close()

def masukexcel_yprediksi(data,predik):
    workbook = xlsxwriter.Workbook('yprediksi.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0
    for row in range(len(data)):
        worksheet.write(row,0, data[row][col])
        worksheet.write(row,2, predik[row][col])
    workbook.close()

def prediksi_kelas(data):
    kelas = [[0 for i in range(1)]for j in range(0,len(data))]
    indeks1 = 0
    indeks2 = 0
    for indeks1 in range(len(data)):
        kelas[indeks1][0] += np.argmax(data[indeks1])
    return kelas

def label_testing(target):
    k = 0
    sign = [99,199,299,399,499,599,699,799,899,999,1099,1199,1299,1399,1499,1599,1699,1799,1899,1999,2099,2199,2299,2399,2499]
    for i in range(len(target)):
        target[i][0] += k
        if i in sign:
            k += 1
    return target

# def label_testing(target):
#     k = 0
#     sign = [15,31,47,63,79,95,111,127,143,159,175,191,207,223,239,255,271,287,303,319,335,351,367,383,399]
#     for i in range(len(target)):
#         target[i][0] += k
#         if i in sign:
#             k += 1
#     return target

for folder in range(26):
    fol = str(folder)
    directory = 'E:/1. Nofri/TA/program/testing_2a/testing_' + fol
    listening = os.listdir(directory)
    for infile in listening:
        path = 'E:/1. Nofri/TA/program/testing_2a/testing_' + fol + '/' + infile
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
label = [[0.0 for i in range(0,no_of_output_neurons)]for j in range(0,no_of_inputs)]

input_hidden_weights = np.array(bobotinput())
bias = np.array(inputbias())
input_hidden_weights = [[input_hidden_weights[j][i] for i in range(0,no_of_hidden_neurons)]for j in range(0,no_of_input_neurons)]
bias = [[bias[j][i] for i in range(0,no_of_hidden_neurons)]for j in range(0,no_of_inputs)]
label_output = label_testing(label)

hidden_matrix = np.array(hidden_layer_matrix(regression_matrix(input_array,input_hidden_weights,bias)))

output_weight_training = np.array(bobotoutput())

matrix_y_prediksi = np.dot(hidden_matrix,output_weight_training)

prediksi = prediksi_kelas(matrix_y_prediksi)
y_prediiksi = []
y_true = []
for i in range(len(prediksi)):
    y_prediiksi.append(float(prediksi[i][0]))
for i in range(len(label_output)):
    y_true.append(float(label_output[i][0]))


# masukexcel_yprediksi(prediksi,label_output)

cnf_matrix = confusion_matrix(y_true, y_prediiksi)
micro = precision_recall_fscore_support(y_true, y_prediiksi, average='macro')
macro = precision_recall_fscore_support(y_true, y_prediiksi, average='micro')
# weighted = precision_recall_fscore_support(y_true, y_prediiksi, average='weighted')

masukexcel_yprediksi(label_output, prediksi)
# print("prediksi = ", prediksi)
print("macro score = ", macro)
print("micro score = ", micro)
# print("weighted score = ", weighted)
