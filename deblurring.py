import numpy as np
import math
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


def multiplyTwoMatricies(A, B):
     dim1 = np.shape(A)
     size1 = np.size(A)
     m1 = dim1[0]
     n1 = size1//m1
     dim2 = np.shape(B)
     size2 = np.size(B)
     m2 = dim2[0]
     n2 = size2//m2
     result = np.zeros((m1, n2))
     for i in range(0, m1):
         for j in range(0, n2):
             for k in range(0, m2):
                 result[i][j] += (A[i][k] * B[k][j])
     return result

def getNorm(A):
    norm = 0
    for i in range(0, len(A)):
        norm += A[i]*A[i]
    return math.sqrt(norm)

def getNormalisedVector(vector):
    length = getNorm(vector)
    for i in range(0, len(vector)):
        vector[i] = vector[i]/length
    return vector

def getTranspose(A):
    dim3 = np.shape(A)
    size3 = np.size(A)
    temp1 = dim3[0]
    temp2 = size3//dim3[0]    
    result = np.zeros((temp2, temp1))
    for i in range(0, temp1):
        for j in range(0, temp2):
            result[j][i] = A[i][j]
    return result

def multiplyScalarToVector(scale, vect):
     for i in range(0, len(vect)):
         vect[i] = scale*vect[i]
     return vect

def power_method(A, num_iter=1000, tol=1e-10):
    n = A.shape[0]
    b_k = np.random.rand(n)
    b_k = b_k / np.linalg.norm(b_k)

    for _ in range(num_iter):
        b_k1 = np.dot(A, b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        b_k1 = b_k1 / b_k1_norm

        if np.linalg.norm(b_k1 - b_k) < tol:
            break

        b_k = b_k1

    eigenvalue = np.dot(b_k.T, np.dot(A, b_k))
    return eigenvalue, b_k

def getEigenValuesPower(A, k=10):
    A_copy = A.copy()
    n = A.shape[0]
    eigenvalues = []
    eigenvectors = []
    for _ in range(k):
        eigval, eigvec = power_method(A_copy)
        eigenvalues.append(eigval)
        eigenvectors.append(eigvec)
        A_copy = A_copy - eigval * np.outer(eigvec, eigvec)
    return np.array(eigenvalues), np.array(eigenvectors).T 


def getSingularValues(A):
     result = np.zeros(len(A))
     for i in range(0, len(A)):
         if(A[i] < 0):
             A[i] = (-1) * A[i]
         result[i] = math.sqrt(A[i])
     return result

def calculateSVD(eValues, eVectors, sv):
    U = np.zeros((n,n))
    V = np.zeros((m,m))
    sigma = np.zeros((n,m))

    for i in range(0, len(eValues)):
        V[:, i] = getNormalisedVector(eVectors[:, i])

    for i in range(0, len(sv)):
        sigma[i][i] = sv[i]

    for i in range(0, m):
        temp = multiplyTwoMatricies(b, eVectors[:, i].reshape(m, 1)).reshape(1, m)
        U[:, i] = multiplyScalarToVector(1/sv[i], temp)
    return U, sigma, getTranspose(V)

img = Image.open("image.png")
img2 = ImageOps.grayscale(img)

b = np.array(img2)

print('The dimension of the image is: ', b.shape)
n,m = b.shape
area = m*n

bT = getTranspose(b)

S = np.dot(b, bT)

eValues1, eVectors = np.linalg.eig(S)
eValues, eVectors = getEigenValuesPower(S, k=16)
singValues = getSingularValues(eValues1)

U, Sigma, VT = calculateSVD(eValues, eVectors, singValues)

k = 1
resultantBlurredMatrixApproximated = U[:,:k] @ Sigma[0:k,:k] @ VT[:k,:]

k = 10
resultantDeblurredMatrixApproximated = U[:,:k] @ Sigma[0:k,:k] @ VT[:k,:]
resultantDeblurredMatrixApproximated = U[:,:k] @ Sigma[0:k,:k] @ VT[:k,:]

k = 1000
resultantDeblurredMatrixApproximatedFinal = U[:,:k] @ Sigma[0:k,:k] @ VT[:k,:]

f, axes = plt.subplots(2,2)
plt.suptitle('Results')
axes[0][0].imshow(img)
axes[0][1].imshow(resultantBlurredMatrixApproximated, cmap='gray',vmin=0, vmax=255)
axes[1][0].imshow(resultantDeblurredMatrixApproximated, cmap='gray',vmin=0, vmax=255)
axes[1][1].imshow(resultantDeblurredMatrixApproximatedFinal, cmap='gray',vmin=0, vmax=255)
plt.show()
