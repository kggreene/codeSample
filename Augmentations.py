import cv2, math, os, numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from sklearn.preprocessing import normalize

class Augmentations():
    #A collection of some functions for augmenting AB image pairs 
    #in DragonPaint, DragonDraw and similar experiments.
    
    #Includes affine transformations and elastic (Gaussian filter) distortion.
    
    #For DragonPaint, if there is a random value, e.g. amount of rotation,
    #we need to use the same random value for A and for B, which the "pair"
    #functions, e.g. randRotationPair, do.
    
    #AFFINE TRANSFORMATIONS
    def affineTPair(self, imgA, imgB, M):
        #Holds the default values we want to use in multiple 
        #functions, e.g. randRotationPair and randxyScaledSkewedPair.
        size=len(imgA)
        return (cv2.warpAffine(img, M,(size, size), 
                               borderMode=cv2.BORDER_CONSTANT, 
                               borderValue=(255,255,255)) for 
                               img in (imgA, imgB))
    
    #ROTATION
    def padForRotation(self, img):
        #Helper/option for randRotationPair for rotation of square 
        #without cropping. Adds a white border around original image big 
        #enough that rotation never crops original image. Will need to 
        #resize to return to original dimensions.
        borderSize = int(len(img)*((math.sqrt(2) - 1)/2))
        return cv2.copyMakeBorder(img, 
                                  top=borderSize, 
                                  bottom=borderSize, 
                                  left=borderSize, 
                                  right=borderSize, 
                                  borderType= cv2.BORDER_CONSTANT, 
                                  value=(255,255,255)) 
    
    def randRotationPair(self, 
                         imgA, 
                         imgB, 
                         characterType='flower', 
                         pad = False):
        #Use for an AB pair.
        #Chooses random rotation between 0 and maxRotation.
        #Rotates both input images same rotation amount. 
        #To shrink to avoid possible cropping, use pad = True.

        #Choose max rotation appropriate for character type.    
        maxRotation={'flower':360,'dragon':10}
        #Use the same random rotation value for imgA and imgB.
        theta = np.random.randint(maxRotation[characterType])
        sizeA=len(imgA)
        if pad:
            #Pad borders, rotate and resize file to original dimensions.
            #Picture inside will be shrunk and rotated.
            padA, padB = (self.padForRotation(img) for img in (imgA, imgB))
            sizePad=len(padA)
            rotM = cv2.getRotationMatrix2D((sizePad/2,sizePad/2),theta,1)
            return (cv2.resize(padImg, (sizeA, sizeA)) for padImg in 
                    self.affineTPair(padA, padB, rotM))
        else:
            rotM = cv2.getRotationMatrix2D((sizeA/2,sizeA/2),theta,1)            
            return self.affineTPair(imgA, imgB, rotM)
        
    #REFLECTION
    def mirrorFlipPair(self, imgA, imgB):
        #flips both input images across y axis
        return (cv2.flip(img, 1) for img in (imgA, imgB))
                
    #TRANSLATION
    def randTranslationPair(self, imgA, imgB):
        #Small translation in x and y. Might crop image
        transMax = 20
        transX = np.random.randint(-transMax, transMax)
        transY = np.random.randint(-transMax, transMax)
        transM = np.float32([[1, 0, transX], [0, 1, transY]])
        return self.affineTPair(imgA, imgB, transM) 
    
    #SCALE SKEW
    def randxyScaledSkewedPair(self, imgA, imgB):
        #scales x, y independently
        #note this also translates, might crop
        #parameters chosen by experimentation
        xscale = np.random.uniform(.75,1)
        yscale = np.random.uniform(.75,1) 
        skewFactor = np.random.uniform(0,.3)
        skewM = np.float32([[xscale,0,0],[skewFactor,yscale,0]])
        return self.affineTPair(imgA, imgB, skewM)
    
    #ELASTIC DISTORTION 
    def elasticDistortionPair(self, 
                              imgA, 
                              imgB, 
                              alpha, 
                              sigma, 
                              random_state=None):
        #Assumes both are 3D arrays (i.e. color not grayscale.) Using 
        #Gaussian blur to add domain appropriate variation to handwritten 
        #numbers (MNIST) was an idea from Simard, et. al. 2002. Provides 
        #a change to the character of the line without changing pose, 
        #a domain appropriate variation for drawings as well.  
        if random_state is None:
            random_state = np.random.RandomState(None)
        shape = imgA.shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), 
                             sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), 
                             sigma, mode="constant", cval=0) * alpha
        dz = np.zeros_like(dx)
        x, y, z = np.meshgrid(np.arange(shape[0]), 
                              np.arange(shape[1]), 
                              np.arange(shape[2]))
        indices = (np.reshape(y+dy, (-1, 1)), 
                   np.reshape(x+dx, (-1, 1)), 
                   np.reshape(z, (-1, 1)))
        distA = map_coordinates(imgA, indices, order=1, mode='reflect')
        distB = map_coordinates(imgB, indices, order=1, mode='reflect')
        return distA.reshape(imgA.shape), distB.reshape(imgB.shape)   
    
    def elasticDistortionPairsLists(self, imgA, imgB):
        #Alphas and sigmas were chosen by experimentation. Makes an A list 
        #and a B list of four elastic distortions for an AB pair, using 
        #alpha, sigma = alphas[i], sigmas[i]
        alphas = [50, 200, 800, 1500]
        sigmas = [3, 4, 7, 8]
        return zip(*[self.elasticDistortionPair(imgA, imgB, alpha, sigma) 
                          for alpha, sigma in zip(alphas, sigmas)])

    #READ ALL IN FOLDER, TRANSFORM, WRITE TO NEW FOLDER
    def transformFolders(self, 
                         readFolderA, 
                         readFolderB, 
                         writeFolderA, 
                         writeFolderB, 
                         transformFunction, 
                         code, 
                         returnsLists = False):
        #Makes new AB pairs.                                                   
        #Takes in names of folders to read from and to write to, a function or 
        #series to use for transformation and a code to mark new files.
        #returnsLists == True means transformFunction returns lists instead of 
        #single images, e.g. elasticTransformPairsLists.
        for file in os.listdir(readFolderA):
            #The image files in readFolderA and readFolderB should always have 
            #the same names. Add code to check that here if you have reason to 
            #think they don't (like an error or a different number of files in 
            #the two folders) and only use the files that do pair up.
            if os.path.basename(file).lower() != 'thumbs.db':
                A = cv2.imread('%s//%s'%(readFolderA, str(file)))
                B = cv2.imread('%s//%s'%(readFolderB, str(file)))
                Anew, Bnew = transformFunction(A,B)
                if returnsLists:
                    for i in range(len(Anew)):
                        newName = self.addCodeToName(str(file), code, i)
                        for writeFolder, newFile in zip((writeFolderA, 
                                                         writeFolderB),
                                                        (Anew[i], Bnew[i])):
                            cv2.imwrite('%s//%s'%(writeFolder, newName), 
                                        newFile) 
                else:
                        newName = self.addCodeToName(str(file), code)
                        for writeFolder, newFile in zip((writeFolderA, 
                                                         writeFolderB),
                                                        (Anew, Bnew)):
                            cv2.imwrite('%s//%s'%(writeFolder, newName), 
                                        newFile) 
    
    def addCodeToName(self, oldName, code, num = None):
        #Inserts string code plus optional number (for list index)into file 
        #name. We need to add a code to the transformed images to give distinct
        #names because several different images will be derived from the same
        #original, e.g. 'flower1.png', and ultimately they'll end up together 
        #in the same folder. It's also helpful if we're manually deciding which 
        #functions/series/parameters gave us good or bad transformations.
        strFile = oldName[:-4]
        ext = oldName[-4:]
        if num!=None:
            return strFile+code+str(num)+ext
        else:
            return strFile+code+ext
            
    #SERIES OF TRANSFORMATIONS
    #Work in progress. Useful series or series I'm experimenting with stored 
    #here. I'm looking to decide which functions, series/compositions and 
    #parameters are most useful to build an augmentation pipeline and perhaps 
    #a single function call that I can use for DragonPaint and all similar 
    #projects/experiments to generate the AB augmentations.
    
    def augFtnComposition(self, imgA, imgB, *functionList):
    #Composition of AB pair augmentation functions that take AB pair (and
    #maybe parameters) and return A'B' pair. Enter paired functions and 
    #parameter values as tuples. 
    #E.g. transA, transB = augFtnComp(A,B,(f,[v1,v2,v3]),(g,[w1,w2]),(h,None))
        Aold = imgA
        Bold = imgB
        for f, paramList in functionList:
            if paramList:
                Anew, Bnew = f(Aold, Bold, *paramList)
            else:
                Anew, Bnew = f(Aold, Bold)  
            Aold, Bold = Anew, Bnew
        return Anew, Bnew
    
    def seriesRSF(self, imgA, imgB):
        #Rotates, scaleskews, flip.
        return self.augFtnComposition(imgA, 
                                      imgB, 
                                      (self.randRotationPair,None), 
                                      (self.randxyScaledSkewedPair, None), 
                                      (self.mirrorFlipPair, None))  
    
    def seriesSPRT(self, imgA, imgB):
        #Scaleskews, pads and rotates, translates.
        return self.augFtnComposition(imgA, 
                                      imgB,  
                                      (self.randxyScaledSkewedPair, None),
                                      (self.randRotationPair, ['flower', True]),
                                      (self.randTranslationPair, None))       
    
    def seriesSPRTE(self, imgA, imgB):
        #Scaleskews, pads and rotates, translates.
        return self.augFtnComposition(imgA, 
                                      imgB,  
                                      (self.randxyScaledSkewedPair, None),
                                      (self.randRotationPair, ['flower', True]),
                                      (self.randTranslationPair, None),
                                      (self.elasticDistortionPair,[50,3])) 
                                      
    #TESTS
    def showTransFile(self, 
                      imgName, 
                      transFtn, 
                      *transFtnParams):
        #Test function for transforming a single image sitting in current 
        #directory (or wherever is passed in as imgName), e.g. 'flower1.png'. 
        #Displays old and new files in windows. Uses same image for A and B 
        #in the pair transformation. If transFtn needs parameters, pass them 
        #as well, e.g. 
        #self.showTransFile(testImgName,augment.elasticDistortionPair, 50, 3)
        img = cv2.imread(imgName)
        imgNewA, imgNewB = transFtn(img, img, *transFtnParams)
        self.showFiles(img, imgNewA)
        return None
    
    def showFiles(self, *imgs):
        for i in range(len(imgs)):
            cv2.imshow('img '+str(i), imgs[i])
        cv2.waitKey(0)
        #Hit any key to close windows
        cv2.destroyAllWindows()
        return None
    
    def testFunctions(self, testType):
    #Tests to use during development to make sure functions are working.
        if testType == 1:
            #Test one file, one function. Uncomment the one you want to see.
            #Make sure you have the files/folders for test data.
            testImgName='Flower1.png'
            img=cv2.imread(testImgName)
            #self.showTransFile(testImgName,self.randRotationPair, 'flower', True)
            #self.showTransFile(testImgName,self.randRotationPair, 'flower')
            #self.showTransFile(testImgName,self.mirrorFlipPair)
            #self.showTransFile(testImgName,self.randxyScaledSkewedPair)
            #self.showTransFile(testImgName,self.randTranslationPair)
            #self.showTransFile(testImgName,self.elasticDistortionPair, 50, 3)
            #self.showTransFile(testImgName,self.seriesRSF) 
            #self.showTransFile(testImgName,self.seriesSPRT)
            self.showTransFile(testImgName,self.seriesSPRTE)
            #self.showFiles(img, img)
            #self.showFiles(*self.augFtnComposition(
                    #img,img,(self.randRotationPair, ['flower', True]), 
                    #(self.randxyScaledSkewedPair, None),
                    #(self.mirrorFlipPair, None)))
        if testType == 2:
            #Test transformations on all files in a folder. Will transform and 
            #write transformed images to different folder pair. Need the four 
            #folders below with a few images in readA and readB (with the same 
            #names in the two read folders) and writeA and writeB empty.
            readA = 'readA'
            readB = 'readB'
            writeA = 'writeA'
            writeB = 'writeB'
            self.transformFolders(readA, readB, writeA, writeB, 
                                  self.elasticDistortionPairsLists, 'E', True)
        return None

#CODE FOR CALLING FUNCTIONS/TESTING AUGMENTATIONS
#A sample test. Make sure you have Flower1.png in working directory.
#For other tests, uncomment a different test in testFunctions.
augment = Augmentations()
augment.testFunctions(1)            
