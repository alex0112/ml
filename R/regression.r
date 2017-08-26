library(caTools)
library(ggplot2)

dataset = read.csv('../moar_data/Salary_Data.csv')

## Split the dataset into the training and test sets:
set.seed(123)
## sample.split(dependent_var (output), what_percentage_is_used_for_training)
split = sample.split(dataset$Salary, SplitRatio = 2/3) ## split the sample into the training and test, taking 80% of the cases as the training set.
## => Returns an array of true or false values.  True if that row was used for training, false if used for test.

training_set = subset(dataset, split == TRUE) ## Select all the data from the data set where it matches split at the same index as TRUE ?  I think?
test_set = subset(dataset, split == FALSE) ## Ditto, but where false.


## Simple linear regression

## Fit:
regressor = lm(
    formula = Salary ~ YearsExperience,
    data = training_set
)

## Predict:
y_pred = predict(regressor, new_data = test_set)

## Visualize the data:
ggplot() +
    geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
               color = 'red') +
    geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, new_data = training_set)),
              color = 'blue') +
    ggtitle('Salary vs XP (training)') +
    xlab('XP') +
    ylab('$$$')

