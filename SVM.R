##### Support Vector Machines -------------------
## Example: Optical Character Recognition ----

#When OCR software first processes a document, 
#it divides the paper into a
#matrix such that each cell in the grid contains a single glyph, 
#which is just a term referring to a letter, symbol, or number. 
#Next, for each cell, the software will attempt to match the glyph to a 
#set of all characters it recognizes. 
#Finally, the individual characters would be combined back together into words, 
#which optionally could be spell-checked against a dictionary in the document's language.

#we'll assume that we have already developed the algorithm to
#partition the document into rectangular regions each consisting of a single character.
#We will also assume the document contains only alphabetic characters in English.
#So,we'll simulate a process that involves matching glyphs to one of the 26 letters, 
#A through Z.

#The dataset contains 20,000 examples of 26 English alphabet capital letters as printed
#using 20 different randomly reshaped and distorted black and white fonts.

#when the glyphs are  scanned into the computer, they are converted into pixels 
#and 16 statistical attributes are recorded.
#The attributes measure such characteristics as the horizontal and vertical
#dimensions of the glyph, the proportion of black (versus white) pixels, and the
#average horizontal and vertical position of the pixels.
#Differences in the concentration of black pixels across various areas of the box
#should provide a way to differentiate among the 26 letters of the alphabet.

## Step 2: Exploring and preparing the data ----
# read in data and examine structure
letters <- read.csv("letterdata.csv")
str(letters)
#some of the ranges for these integer variables appear fairly wide.
#This indicates that we need to normalize or standardize the data. 
# we can skip this step for now,because the R package that
#we will use for fitting the SVM model will perform the rescaling automatically.

# divide into training and test data

#the first 16,000 records (80 percent) to build the model and
#the next 4,000 records (20 percent) to test.

letters_train <- letters[1:16000, ]
letters_test  <- letters[16001:20000, ]

## Step 3: Training a model on the data ----
# begin by training a simple linear SVM
install.packages('kernlab')
library(kernlab)

#To provide a baseline measure of SVM performance, 
#let's begin by training a simple linear SVM classifier.
#Then, we can call the ksvm() function on the training data and 
#specify the linear kernel using the vanilladot option,

letter_classifier <- ksvm(letter ~ ., data = letters_train,
                          kernel = "vanilladot")

# look at basic information about the model
letter_classifier

## Step 4: Evaluating model performance ----
# predictions on testing dataset

letter_predictions <- predict(letter_classifier, letters_test)

#This returns a vector containing a predicted letter for each row of values in the test data. 
#Using the head() function, we can see that the first six predicted letters
#were U, N, V, X, N, and H

head(letter_predictions)

#To examine how well our classifier performed, we need to compare the predicted
#letter to the true letter in the testing dataset.

table(letter_predictions, letters_test$letter)

# look only at agreement vs. non-agreement
# construct a vector of TRUE/FALSE indicating correct/incorrect predictions

#We can simplify our evaluation instead by calculating the overall accuracy. 
#This considers only whether the prediction was correct or incorrect,and ignores the type of error.

#The following command returns a vector of TRUE or FALSE values, indicating
#whether the model's predicted letter agrees with (that is, matches) the actual
#letter in the test dataset

agreement <- letter_predictions == letters_test$letter

#we see that the classifier correctly identified the letter in 3,357 out of the 4,000 test records

table(agreement)
prop.table(table(agreement))

## Step 5: Improving model performance ----

#Our previous SVM model used the simple linear kernel function. By using a more
#complex kernel function, we can map the data into a higher dimensional space, and
#potentially obtain a better model fit.
#let's use Gaussian RBF kernel
set.seed(12345)
letter_classifier_rbf <- ksvm(letter ~ ., data = letters_train, kernel = "rbfdot")
letter_predictions_rbf <- predict(letter_classifier_rbf, letters_test)

agreement_rbf <- letter_predictions_rbf == letters_test$letter
table(agreement_rbf)
prop.table(table(agreement_rbf))
