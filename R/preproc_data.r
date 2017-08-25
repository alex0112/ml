library(caTools)

dataset = read.csv('../moar_data/Data.csv')

## If an Age is not available (i.e. not there, not a number etc.), then replace it with the mean of all of the defined values in that column.
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age
                     )

## Same, but for Salary column
dataset$Salary = ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Salary
                     )

## Encode categorical data for Country column:
dataset$Country = factor(dataset$Country,
                         levels = c('France', 'Spain', 'Germany'),
                         labels = c(1, 2, 3)
                         )
## Encode categorical data for dependent var:
dataset$Purchased = factor(dataset$Purchased,
                         levels = c('No', 'Yes'),
                         labels = c(0, 1)
                         )


## Split the dataset into the training and test sets:
set.seed(123)
## sample.split(dependent_var (output), what_percentage_is_used_for_training)
split = sample.split(dataset$Purchased, SplitRatio = 0.8) ## split the sample into the training and test, taking 80% of the cases as the training set.
## => Returns an array of true or false values.  True if that row was used for training, false if used for test.

training_set = subset(dataset, split == TRUE) ## Select all the data from the data set where it matches split at the same index as TRUE ?  I think?
test_set = subset(dataset, split == FALSE) ## Ditto, but where false.


training_set[,2:3] = scale(training_set[,2:3])
test_set[,2:3] = scale(test_set[,2:3])

print("dataset:")
print(dataset)
print("*************************************************")
print("test set:")
print(training_set)
print("*************************************************")
print("test set:")
print(test_set)



