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
dataset$Country = factor(dataset$Country
                         levels = c('France', 'Spain', 'Germany'),
                         labels = c(1, 2, 3)
                         )
## Encode categorical data for dependent var:
dataset$Country = factor(dataset$Purchased
                         levels = c('No', 'Yes'),
                         labels = c(0, 1)
                         )




print(dataset)

