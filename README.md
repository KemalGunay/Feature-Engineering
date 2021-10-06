# Feature-Engineering

Required for a machine learning pipeline data preprocessing and variable engineering script
needs to be prepared.

When the dataset is passed through this script, the modeling starts.
expected to be ready.

## Dataset Story
The data set is the data set of the people who were in the Titanic shipwreck.
It consists of 768 observations and 12 variables.
The target variable is specified as "Survived";
1: one's survival,
0: indicates the person's inability to survive.

## Variables
 **PassengerId:** ID of the passenger
* **Survived:** Survival status (0: not survived, 1: survived)
* **Pclass:** Ticket class (1: 1st class (upper), 2: 2nd class (middle), 3: 3rd class(lower))
* **Name:** Name of the passenger
* **Sex:** Gender of the passenger (male, female)
* **Age:** Age in years
* **Sibsp:** Number of siblings/spouses aboard the Titanic
     * Sibling = Brother, sister, stepbrother, stepsister
     * Spouse = Husband, wife (mistresses and fiances were ignored)
 **Parch:** Number of parents/children aboard the Titanic
     * Parent = Mother, father
     * Child = Daughter, son, stepdaughter, stepson
     * Some children travelled only with a nanny , therefore Parch = 0 for them.
 * **Ticket:** Ticket number
 * **Fare:** Passenger fare
 * **Cabin:** Cabin number
 * **Embarked:** Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
 
 
 **REFERENCE:** Data Science and ML Boot Camp, 2021, Veri Bilimi Okulu (https://www.veribilimiokulu.com/)
 
