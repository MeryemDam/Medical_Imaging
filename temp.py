# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

#medical imaging
#%%
#importations
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#%%
#charging images

img1 = mpimg.imread("IRMcoupe17-t1.jpg")
img2=mpimg.imread("IRMcoupe17-t2.jpg")


#%%
plt.figure(1)
plt.title('image tumeur 1')
plt.imshow(img1,cmap='Greys',  interpolation='nearest')
plt.show()
plt.title('image tumeur 2')
plt.imshow(img2,cmap='Greys',  interpolation='nearest')
plt.show()
#%%histogrammes
plt.figure()
plt.hist(img1)
plt.title('hist1')
plt.show()

plt.figure()
plt.hist(img2)
plt.title('hist2')
plt.show()

#%%
#le seuillage
import numpy as np
image_seuil1=np.where(img1<80,255,0)
plt.imshow(image_seuil1 ,cmap='Greys',  interpolation='nearest')

#%%

image_seuil2=np.where(img2>80,0,255)
plt.imshow(image_seuil2 ,cmap='Greys',  interpolation='nearest')
#%%
from skimage import morphology

# Dilatation
erosion_img1 = morphology.binary_erosion(image_seuil1)

plt.figure()
plt.imshow(erosion_img1, cmap=plt.cm.gray)
plt.axis('off')
plt.title('Image érosion 1')
#%%
erosion_img2 = morphology.binary_erosion(image_seuil2)

plt.figure()
plt.imshow(erosion_img2, cmap=plt.cm.gray)
plt.axis('off')
plt.title('Image érosion 2')

#%%
dilation_img1 = morphology.binary_dilation(erosion_img1, morphology.disk(4))

plt.figure()
plt.imshow(dilation_img1, cmap=plt.cm.gray)
plt.title('Image dilatée 1')
#%%
dilation_img2 = morphology.binary_dilation(erosion_img2, morphology.disk(4))

plt.figure()
plt.imshow(dilation_img2, cmap=plt.cm.gray)
plt.title('Image dilatée 2')


#%%labeliser l'image
#choisir la deuxième zone sauf 0
import skimage.morphology as sk
image_label_1=sk.label(dilation_img1,8)
plt.title('image labellisée 1')
plt.imshow(image_label_1,cmap='Greys',  interpolation='nearest')

#%%
image_label_2=sk.label(dilation_img2,8)
plt.title('image labellisée 2')
plt.imshow(image_label_2,cmap='Greys',  interpolation='nearest')


#%%

# Display the image and plot all contours found
from skimage import measure
contours1 = measure.find_contours(image_label_1, 0.9)
fig, ax = plt.subplots()
ax.imshow(img1, interpolation='nearest', cmap=plt.cm.gray)

for n, contour in enumerate(contours1):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()
#%%

contours2 = measure.find_contours(image_label_2, 0.9)
fig, ax = plt.subplots()
ax.imshow(img2, interpolation='nearest', cmap=plt.cm.gray)

for n, contour in enumerate(contours2):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()

#%%
#calculer la différence entre les deux images
diff=len(contours1[0][:])-len(contours2[0][:])
difference=diff/len(contours1[0][:])

print(difference*100)



#%% Methode de FCM

from skimage import io
from scipy.ndimage.measurements import label
from scipy.ndimage import sobel
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import exposure,measure
import numpy as np
import random
import operator
import math

#%% Load images

image1 = io.imread("IRMcoupe17-t1.jpg")
image2 =io.imread("IRMcoupe17-t2.jpg")
plt.figure(1)
plt.title("image originale")
io.imshow(image1)
plt.show()
io.imshow(image2)
plt.show()

#%% Fuzzy C-means Clustering Algorithm

def centroides(image,M,c,m):
    img = np.ravel(image)
    n,d = image.shape
    N = len(img)
    v = np.zeros(c)
    for i in range(c):
        som1=0
        som2=0
        for k in range(N):
            som1+=(((M[k,i])**m)*img[k])
            som2+=(M[k,i])**m
        v[i]=som1/som2
    
    return(v)

def ajuste_deg_app(image,M,c,m,v) :
    
    img = np.ravel(image)
    n,d = image.shape
    N = len(img)
    d = np.random.rand(N,c)
    for i in range(c):
        for k in range(N):
            d[k,i]=np.linalg.norm(img[k]-v[i])
            som=0
            for j in range(c):
                d[k,j]=np.linalg.norm(img[k]-v[j])
                som+=(d[k,i]/d[k,j])**(2/(m-1))
            M[k,i]=1/som

    return(M)
    
def fuzzy_c_means(image,M,c,m,erreur):
    n,d = image.shape
    N = n*d
    mat_memo = np.random.rand(N,c)
    i = 0
    while(np.max(abs(M-mat_memo))>erreur or i == 30):
        print('i :',i)
        print('max diff : ',np.max(abs(M-mat_memo)))
        mat_memo = np.copy(M)
        v = centroides(image,M,c,m)
        M = ajuste_deg_app(image,M,c,m,v)
        i+=1
    
    y = np.zeros(N)
    for i in range(N):
        y[i] = np.where(M[i] == np.max(M[i]))[0][0]
    img_classe = y.reshape((n,d))
    return(M,v,img_classe)
        
def Binarisation(image,seuil):
    I=np.copy(image)
    nl, nc = I.shape
    npn = 0
    for i in range(nl):
        for j in range(nc):
            if I[i,j]<seuil:
                I[i,j] = 0
                npn = npn + 1
            else:
                I[i,j] = 1
    return(I)
    
#%% TEST : Fuzzy C-means Clustering sur les deux images

m=2
c=3
n1,d1 = image1.shape
n2,d2 = image2.shape
N1=n1*d1
N2=n2*d2
M1 = np.random.rand(N1,c) 
M2 = np.random.rand(N2,c)

erreur = 1e-3 

M1,v1,img_classe1 = fuzzy_c_means(image1,M1,c,m,erreur)
M2,v2,img_classe2 = fuzzy_c_means(image2,M2,c,m,erreur)

#%% PLOT : image et son image labelisée

plt.figure(2)
plt.title("image originale")
io.imshow(image1)
plt.show()
plt.title("image1 classée")
io.imshow(img_classe1,cmap='gray')
plt.show()

plt.figure(3)
plt.title("image originale")
io.imshow(image2)
plt.show()
plt.title("image2 classée")
io.imshow(img_classe2,cmap='gray')
plt.show()

plt.figure(4)
plt.title("classe 1")
io.imshow(M2[:,0].reshape((n1,d1)))
plt.show()
plt.title("classe 2")
io.imshow(M2[:,1].reshape((n1,d1)))
plt.show()
plt.title("classe 3")
io.imshow(M2[:,2].reshape((n1,d1)))
plt.show()

#%% pourcentage d'augmentation de la tumeur:

seuil = 0.5
ind1 = np.where(v1 == np.max(v1))
ind2 = np.where(v2 == np.max(v2))

image1_bin = Binarisation(M1[:,ind1].reshape((n1,d1)),seuil)
image2_bin = Binarisation(M2[:,ind2].reshape((n2,d2)),seuil)

labeled_img1, num_features_img1 = label(image1_bin)
labeled_img2, num_features_img2 = label(image2_bin)

plt.figure(5)
plt.imshow(labeled_img1,cmap=plt.cm.gray)
plt.show()
plt.imshow(labeled_img2,cmap=plt.cm.gray)
plt.show()

taille_labels1 = []
for k in range(num_features_img1):
    nb_pixel_lab=0
    for i in range(n1):
        for j in range(d1):
            if labeled_img1[i,j] == k:
                nb_pixel_lab+=1
    taille_labels1.append(nb_pixel_lab)

taille_labels1.remove(np.max(taille_labels1))
taille_tumeur1 = np.max(taille_labels1)

taille_labels2 = []
for k in range(num_features_img2):
    nb_pixel_lab=0
    for i in range(n2):
        for j in range(d2):
            if labeled_img2[i,j] == k:
                nb_pixel_lab+=1
    taille_labels2.append(nb_pixel_lab)
    
taille_labels2.remove(np.max(taille_labels2))
taille_tumeur2 = np.max(taille_labels2)

#%%

k1 = taille_labels1.index(taille_tumeur1)
k2 = taille_labels2.index(taille_tumeur2)

mat1 = np.zeros((n1,d1))
mat2 = np.zeros((n2,d2))

for i in range(n1):
    for j in range(d1):
        if labeled_img1[i,j] == k1+1:
            mat1[i,j]=1
        

for i in range(n2):
    for j in range(d2):
        if labeled_img2[i,j] == k2+1:
            mat2[i,j]=1
            
plt.figure(6)
plt.imshow(mat1,cmap=plt.cm.gray)
plt.show()
plt.imshow(mat2,cmap=plt.cm.gray)
plt.show()

#%%   
nb_pixel_img1 = 0
for i in range(n1):
    for j in range(d1):
        if mat1[i,j] == 1:
            nb_pixel_img1 += 1

nb_pixel_img2 = 0
for i in range(n2):
    for j in range(d2):
        if mat2[i,j] == 1:
            nb_pixel_img2 += 1
            
difference = nb_pixel_img2 - nb_pixel_img1

difference = nb_pixel_img2 - nb_pixel_img1
print('augmentation : ',(difference/nb_pixel_img1)*100,'%')

#%% Méthode 3
import math


def ecart_type(M):
    sigma=math.sqrt(np.var(M))
    return sigma
def moyenne (M):
    moyenne= np.mean(M)
    return moyenne

def loi_normale(ecart_type, moyenne,y):
    return (1/math.sqrt(2*math.pi*ecart_type))*math.exp(-(y-moyenne)**2/2*ecart_type**2) 
#%%

#tirer les moyennes et l'écart type de chaque image
    
partie_back1=img1[0:10,0:30]
partie_cerveau1=img1[50:100,100:200]
partie_tumeur1=img1[130:150,100:150]

#calculer la moyenne et l'écart type de chaque partie

m_img1=moyenne(partie_back1)
m_img2=moyenne(partie_cerveau1)
m_img3=moyenne(partie_tumeur1)

sigma_img1=ecart_type(partie_back1)
sigma_img2=ecart_type(partie_cerveau1)
sigma_img3=ecart_type(partie_tumeur1)

moyenne=[m_img1,m_img2,m_img3]
sigma=[sigma_img1,sigma_img2,sigma_img3]
#%%
U=[]
Y1=np.ravel(img1)
m=len(Y1)
n=len(moyenne)
proba=np.zeros((m,n))

#%%
for i in range (m):
    for j in range (n):
        proba[i,j]=loi_normale(sigma[j],moyenne[j],Y1[i])
        if proba [i,j]==0:
            proba[i,j]=10**-8
        

U1=np.zeros((m,n))       
for i in range (m):
    for j in range (n):
        U1[i,j]=-math.log(proba[i,j])
    
#définir U2

#%% Méthode 3: Contour Actif
from skimage.segmentation import active_contour
from skimage.filters import gaussian  
s = np.linspace(0, 2*np.pi, 160)
x = 160 + 35*np.cos(s)
y =130 + 50*np.sin(s)
init = np.array([x, y]).T 
snake1 = active_contour(gaussian(dilation_img1, 2),
                       init, alpha=0.015, beta=5, gamma=0.001)
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(img1, cmap=plt.cm.gray)

ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
ax.plot(snake1[:, 0], snake1[:, 1], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img1.shape[1], img1.shape[0], 0])



snake2 = active_contour(gaussian(dilation_img2, 2),
                       init, alpha=0.015, beta=5, gamma=0.001)
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(img2, cmap=plt.cm.gray)

ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
ax.plot(snake2[:, 0], snake2[:, 1], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img2.shape[1], img2.shape[0], 0])





       
        
    




    