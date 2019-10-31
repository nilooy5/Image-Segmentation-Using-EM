# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import cv2
import matplotlib.pyplot as plt
import os 
from os.path import join
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
from skimage.color import rgb2gray
from skimage.color import label2rgb
from skimage.filters import gaussian
from sklearn.cluster import KMeans


plt.close('all')
clear = lambda: os.system('clear')
clear()
np.random.seed(110)

colors = [[1,0,0],[0,1,0],[0,0,1],[0,0.5,0.5],[0.5,0,0.5]]

imgNames = ['water_coins','jump','tiger']#{'balloons', 'mountains', 'nature', 'ocean', 'polarlights'};
segmentCounts = [2,3,4,5]

result={}

for imgName in imgNames:
    segResult={}
    for SegCount in segmentCounts:
        # Load the imageusing OpenCV        
        imgInputPath=join(''.join(['Input/',imgName,'.png']))
        img = mpimg.imread(imgInputPath) 
        print('Using Matplotlib Image Library: Image is of datatype ',img.dtype,'and size ',img.shape) # Image is of type float 

        # Load the Pillow-- the Python Imaging Library
        img = np.asanyarray(Image.open(imgInputPath) )
        print('Using Pillow (Python Image Library): Image is of datatype ',img.dtype,'and size ',img.shape) # Image is of type uint8  
                
        
        #%% %Define Parameters
        nSegments = SegCount   # of color clusters in image
        nPixels =  img.shape[0]*img.shape[1]   # Image can be represented by a matrix of size nPixels*nColors
        maxIterations = 20; #maximum number of iterations allowed for EM algorithm.
        nColors = 3;
        #%% Determine the output path for writing images to files
        outputPath = join(''.join(['Output/',str(SegCount), '_segments/', imgName , '/']));
        if not(os.path.exists(outputPath)):
            os.makedirs(outputPath)
            mpimg.imsave(outputPath+'0.png',img)
        """ save input image as *0.png* under outputPath-- 3 points""" #save using Matplotlib image library
        #%% Vectorizing image for easier loops- done as im(:) in Matlab
        pixels = img
        pixels =  pixels.reshape(nPixels,nColors,1) #""" Reshape pixels as a nPixels X nColors X 1 matrix-- 5 points"""
        
       
        #%%
        """ Initialize pi (mixture proportion) vector and mu matrix (containing means of each distribution)
            Vector of probabilities for segments... 1 value for each segment.
            Best to think of it like this...
            When the image was generated, color was determined for each pixel by selecting
            a value from one of "n" normal distributions. Each value in this vector 
            corresponds to the probability that a given normal distribution was chosen."""
        
        
        """ Initial guess for pi's is 1/nSegments. Small amount of noise added to slightly perturb 
           GMM coefficients from the initial guess"""
           
        pi = 1/nSegments*(np.ones((nSegments, 1),dtype='float'))
        increment = np.random.normal(0,.0001,1)
        for seg_ctr in range(len(pi)):
            if(seg_ctr%2==1):
                pi[seg_ctr] = pi[seg_ctr] + increment
            else:
                pi[seg_ctr] = pi[seg_ctr] - increment
        
        #%%
        """Similarly, the initial guess for the segment color means would be a perturbed version of [mu_R, mu_G, mu_B],
           where mu_R, mu_G, mu_B respectively denote the means of the R,G,B color channels in the image.
           mu is a nSegments X nColors matrrix,(seglabels*255).np.asarray(int) where each matrix row denotes mean RGB color for a particcular segment"""
           
        mu = 1/nSegments*(np.ones((nSegments, nColors),dtype='float'))
        #%%
        """Initialize mu to 1/nSegments*['ones' matrix (whose elements are all 1) of size nSegments X nColors] -- 5 points"""  #for even start
        #add noise to the initialization (but keep it unit)
        for seg_ctr in range(nSegments):
            if(seg_ctr%2==1):
                increment = np.random.normal(0,.0001,1)
            for col_ctr in range(nColors):
                 if(seg_ctr%2==1):
                    mu[seg_ctr,col_ctr] = np.mean(pixels[:,col_ctr]) + increment
                 else:
                    mu[seg_ctr,col_ctr] = np.mean(pixels[:,col_ctr]) - increment;              
        

        # #%% EM-iterations begin here. Start with the initial (pi, mu) guesses        
        
        mu_last_iter = mu;
        pi_last_iter = pi;
        
        
        for iteration in range(maxIterations):
            """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
               % -----------------   E-step  -----estimating likelihoods and membership weights (Ws)
               %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""

            print(''.join(['Image: ',imgName,' nSegments: ',str(nSegments),' iteration: ',str(iteration+1), ' E-step']))
            # Weights that describe the likelihood that pixel denoted by "pix_import scipy.miscctr" belongs to a color cluster "seg_ctr"
            Ws = np.ones((nPixels,nSegments),dtype='float')  # temporarily reinitialize all weights to 1, before they are recomputed

            """ logarithmic form of the E step."""
            
            for pix_ctr in range(nPixels):
                # Calculate Ajs
                logAjVec = np.zeros((nSegments,1),dtype='float')
                for seg_ctr in range(nSegments):
                    x_minus_mu_T  = np.transpose(pixels[pix_ctr,:]-(mu[seg_ctr,:])[np.newaxis].T)
                    x_minus_mu    = ((pixels[pix_ctr,:]-(mu[seg_ctr,:])[np.newaxis].T))
                    logAjVec[seg_ctr] = np.log(pi[seg_ctr]) - .5*(np.dot(x_minus_mu_T,x_minus_mu))
                
                # Note the max
                logAmax = max(logAjVec.tolist()) 
                # Calculate the third term from the final eqn in the above link
                thirdTerm = 0;
                for seg_ctr in range(nSegments):
                    thirdTerm = thirdTerm + np.exp(logAjVec[seg_ctr]-logAmax)
                
                # Here Ws are the relative membership weights(p_i/sum(p_i)), but computed in a round-about way 
                
                for seg_ctr in range(nSegments):
                    logY = logAjVec[seg_ctr] - logAmax - np.log(thirdTerm)
                    Ws[pix_ctr][seg_ctr] = np.exp(logY)
                

            """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % -----------------   M-step  --------------------
               %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
            
            print(''.join(['Image: ',imgName,' nSegments: ',str(nSegments),' iteration: ',str(iteration+1), ' M-step: Mixture coefficients',str(pi.squeeze()[:])]))
            #%% temporarily reinitialize mu and pi to 0, before they are recomputed
            mu = np.zeros((nSegments,nColors),dtype='float') # mean color for each segment
            pi = np.zeros((nSegments,1),dtype='float') #mixture coefficients

            
            for seg_ctr in range(nSegments):

                denominatorSum = 0;
                # for pix_ctr in range(nPixels):
                #     #"""Update RGB color vector of mu[seg_ctr] as current mu[seg_ctr] + pixels[pix_ctr,:] times Ws[pix_ctr,seg_ctr] -- 5 points"""
                #     mu[seg_ctr,:]= mu[seg_ctr,:]+np.squeeze(pixels[pix_ctr,:])*Ws[pix_ctr][seg_ctr]
                #     denominatorSum = denominatorSum + Ws[pix_ctr][seg_ctr]
                
                """Compute mu[seg_ctr] and denominatorSum directly without the 'for loop'-- 10 points.
                   If you find the replacement instruction, comment out the for loop with your solution"
                   Hint: Use functions squeeze, tile and reshape along with sum"""
               
               
                mu[seg_ctr,:]= np.sum(np.multiply(np.squeeze(pixels),np.tile(Ws[:,seg_ctr],(3,1)).T),axis=0)      
                denominatorSum = np.sum(Ws[:,seg_ctr],axis=0)


                ## Update mu
                mu[seg_ctr,:] =  mu[seg_ctr,:]/ denominatorSum;
                ## Update pi
                pi[seg_ctr] = denominatorSum / nPixels; #sum of weights (each weight is a probability) for given segment/total num of pixels   
        

            #print(np.transpose(pi))

            muDiffSq = np.sum(np.multiply((mu - mu_last_iter),(mu - mu_last_iter)))
            piDiffSq = np.sum(np.multiply((pi - pi_last_iter),(pi - pi_last_iter)))

            if (muDiffSq < .0000001 and piDiffSq < .0000001): #sign of convergence
                print('Convergence Criteria Met at Iteration: ',iteration, '-- Exiting code')
                break;
            

            mu_last_iter = mu;
            pi_last_iter = pi;


            ##Draw the segmented image using the mean of the color cluster as the 
            ## RGB value for all pixels in that cluster.
            segpixels = np.array(pixels)
            cluster = 0
            for pix_ctr in range(nPixels):
                cluster = np.where(Ws[pix_ctr,:] == max(Ws[pix_ctr,:]))
                vec     = np.squeeze(np.transpose(mu[cluster,:])) 
                segpixels[pix_ctr,:] =  vec.reshape(vec.shape[0],1)
                
            """ Save segmented image at each iteration. For displaying consistent image clusters, it would be useful to blur/smoothen the segpixels image using a Gaussian filter.  
                Prior to smoothing, convert segpixels to a Grayscale image, and convert the grayscale image into clusters based on pixel intensities"""
            
            segpixels = np.reshape(segpixels,(img.shape[0],img.shape[1],nColors)) ## reshape segpixels to obtain R,G, B image

            
            

            """convert segpixels to uint8 gray scale image and convert to grayscale-- 5 points""" #convert to grayscale
            
            segpixels = rgb2gray(segpixels.astype('uint8'))
            
            # print(segpixels.shape)
            
            """ Use kmeans from sci-kit learn library to cluster pixels in gray scale segpixels image to *nSegments* clusters-- 10 points"""
            
            kmeans = KMeans(n_clusters=nSegments).fit(np.reshape(segpixels,(-1,1)))
            
            # print(kmeans.labels_.shape)
            
            """ reshape kmeans.labels_ output by kmeans to have the same size as segpixels -- 5 points"""
            
            seglabels = np.reshape(kmeans.labels_,(segpixels.shape[0],segpixels.shape[1]))
            
            #print(seglabels.shape)
            
            "Use np.clip, Gaussian smoothing with sigma =2 and label2rgb functions to smoothen the seglabels image, and output a float RGB image with pixel values between [0--1]-- 20 points"""
            
            seglabels = np.clip(gaussian(label2rgb(seglabels,colors=colors),sigma=2,multichannel=True),a_min=0,a_max=1)
            
            segResult[SegCount]=seglabels

            mpimg.imsave(''.join([outputPath,str(iteration+1),'.png']),seglabels) #save the segmented output

    result[imgName]=segResult

""" Display the 20th iteration (or final output in case of convergence) segmentation images with nSegments = 2,3,4,5 for the three images-- this will be a 3 row X 4 column image matrix-- 15 points"""  
    
fig = plt.figure(figsize = (4, 4))

# plot number
i = 1

for imgName in imgNames :
    for segCount in segmentCounts :
        
        outputPath = join(''.join(['Output/',str(segCount), '_segments/', imgName , '/']))
        
        flist = []
        
        imgfiles = os.listdir(outputPath)
        
        for file in imgfiles:
            file = int(file[:-4])
            flist.append(file)
            
        flist.sort()
        
        lastImg = flist[-1]
        
        img = mpimg.imread(outputPath + str(lastImg) + '.png')
        im = fig.add_subplot(len(imgNames), len(segmentCounts), i)
        im.axis('off')
        
        if i<=4:
            im.set_title('nSeg = '+str(segCount),fontdict={'fontsize': 8, 'fontweight': 'medium'})
        
        plt.imshow(img)
        i = i + 1
        
plt.show()
fig.savefig('result.png')


""" Comment on the results obtained, and discuss your understanding of the Image Segmentation problem in general-- 10 points """  

"""
The images were segmented using EM Maximization, which assumes that we know the 
no. of segments in the image apriori. The segmentation for each image was done for different assumptions of the no. of segments in the image.

It was observed that the segmentation of coins image, with the assumption of two segments converged within the first 5 iterations.This was expected 
as the coin image had quite consistent differentiation in terms of color and contrast between the white background and the bronze coins.
The assumption of two segments thus facilitated quick segmentation convergence.

The jump image, assuming 2 segments was segmented into the foreground ice and background sky quite satisfactorily, the ice in green and sky in red (in segmented image).
The jump image,with the assumption of 3 segments, converged at the 16th iteration.The segmentation results can be explained by the variation in the color
intensities of the original image. Very light colored portions(ice in the original image,white colored) was perfectly segmented from the background and the person.
The light and dark colors of the background were unfortunately segmented into 2 eventhough it was the same sky. Dark portions of the person appeared segmented with the
dark portions of the sky, while light colored portions of the person appeared segmented with the light colored part of the sky.

The tiger image was the most difficult task to segment provided the variation in colors within each entity(tiger stripes, Variation in grass color,etc).
Nevertheless the entities tiger,grass,ground and water was segmented,with the presence of noise in each segmented entity.

The Image Segmentation implemented above is by applying EM Maximization on a mutli modal gaussian distribution, assuming we know the no of modes in the underlying distribution.
We randomly initialize random weights to the modes and the means of each mode. For each pixel, we then find the probablity it belongs to each of the gaussians. This is done in the E-Step.
We then update the mean and weight of each gaussian, utilizing the new probabilities of each pixel belonging to a gaussian. 
This constitutes the M-Step. The repetition of the above two steps till convergence results in each gaussian approximating a segment in the image.
We then use kmeans to cluster the pixel intensities, assuming the no of clusters to be the no of segments assumed to be present in the image.
The label of each pixel is updated to be the cluster centroid label and are colored same.
We finally clip the pixel intensities to be in range 0-1 and apply a gaussian smoothening filter with standard deviation =2 to obtain the final segmented
image.

Note : The converged segmented images for each category,for different assumptions of segments is saved as results.png in the folder.

"""