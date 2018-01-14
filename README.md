# codeSample 
### PythonPaint Augmentations

##### Code sample:
This code sample is part of the code for a machine learning training set augmentation generator for DragonPaint (automated drawing and coloring of cartoon characters) and related projects.

The augmentation goal is to leverage extremely small original data sets (e.g. 30 drawings, only one colored, for B&W sketch to color) to get to the estimated 400-1000 training pairs we need to follow Isola, et. al.'s GANs image to image translation work. 


![alt text](./Dragon200x200.bmp)

##### DragonPaint: 
DragonPaint uses geometric rules and other augmentations to leverage extremely small original data sets to create augmented training set generation for automating coloring and drawing with machine learning. 

##### DragonPaint - B&W Sketch to Color:
Presented at the PAPIs machine learning conference, October, 2017, and at Boston Python, December, 2018, the first DragonPaint project trained a GANs model to color cartoon characters of two character types, dragons and flowers, in a consistent way across type by combining geometric rules, rule breaking transformations and machine learning. 

We used geometric rules to create the colored version in AB sketch/colored pairs for 30-40 original "rule conforming" drawings of each type (background = largest component = white, body/center = next largest = orange...) After using components and rule conforming drawings to create AB pairs, we created "rule breaking" drawings with colored mates to add to the training set by applying rule breaking transformations AB -> A'B' or AB -> A'B. Examples include erasing parts of lines in an A drawing and pairing it with the colored version B generated before the erasing, or cropping/rescaling A and B so they broke the rule that the background must be bigger than the body. 

Having created augmented training sets in the 400-1000 range, we applied Isola, et. al's GANs image to image translation work and Hesse's TensorFlow implementation to successfully color flowers and dragons with the trained model, including several types that the geometric rules could not (e.g. drawings with poorly connected lines or flowers with small centers.)

##### DragonPaint - Layers of Complexity/Spikify:
The second DragonPaint project will investigate the stages of complexity in the creation of a drawing and see whether we can train the same AB imge to image translation model to draw petals and spikes by using temporal information and saving drawings in stages (e.g. draw the center, save the drawing as an A, add the petals to that center, save it as the paired B.)

##### DragonPaint Presentations
PAPIs ML conference talk, October, 2017 (slides at https://drive.google.com/open?id=1XtB26GEqcZI-nPldiM92hiByG1SH0-X1)
Boston Python meetup, December, 2017

##### Resources
*Image to Image Translation with Conditional Adversarial Networks*, Isola, et. al. https://arxiv.org/abs/1611.07004
TensorFlow Pix2Pix implementation by Christopher Hesse https://github.com/affinelayer/pix2pix-tensorflow
*Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis*, Simard, et. al. https://www.microsoft.com/en-us/research/publication/best-practices-for-convolutional-neural-networks-applied-to-visual-document-analysis/
