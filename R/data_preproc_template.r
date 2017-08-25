library(caTools)

dataset = read.csv('../moar_data/Data.csv')

## Split the dataset into the training and test sets:
set.seed(123)
## sample.split(dependent_var (output), what_percentage_is_used_for_training)
split = sample.split(dataset$Purchased, SplitRatio = 0.8) ## split the sample into the training and test, taking 80% of the cases as the training set.
## => Returns an array of true or false values.  True if that row was used for training, false if used for test.

training_set = subset(dataset, split == TRUE) ## Select all the data from the data set where it matches split at the same index as TRUE ?  I think?
test_set = subset(dataset, split == FALSE) ## Ditto, but where false.


## training_set[,2:3] = scale(training_set[,2:3])
## test_set[,2:3] = scale(test_set[,2:3])
