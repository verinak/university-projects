#taking dataset path as input from user and using it to create a dataframe
datasetPath <- readline("Enter the full path of the csv file: ")

library(dplyr)

projectData <- read.csv(datasetPath)


###1st step: assessing the data

dim(projectData)
names(projectData)
head(projectData)
str(projectData)
summary(projectData)


###2nd step: data visualization

#To combine all the plots in one Dashboard we use par() function
par(mfrow=c(2,2))

#I-creating a dataframe with cash and credit totals and representing it with a pie chart
typeTotal <- group_by(projectData, paymentType)
typeTotal <- summarise(typeTotal,paymentTotal=sum(total))
head(typeTotal)


#the pie chart is used to represent here the Cash and Credits totals
#Notice:
#this chart is more used when we have 2 or 3 parameters or variables not with many parameters

pie(
  x=typeTotal$paymentTotal,
  labels=typeTotal$paymentType,
  main="Cash and Credit totals"
)

#II-Retrieving age and total spending data and comparing it using a scatter plot
Age_By_Total_Spending<-group_by(projectData,age)
head(Age_By_Total_Spending)
Age_By_Total_Spending<-summarise(Age_By_Total_Spending, totalSalary=sum(total))
Age_By_Total_Spending
plot(
  x=Age_By_Total_Spending$age,
  y=Age_By_Total_Spending$totalSalary,
  main="Age and Sum of Total Spending",
  xlab="Age",
  ylab="Total Spending", col="deepskyblue"
)

#III-Creating a dataframe to hold the spending data grouped by city and arranged by 
#total descending, then representing the data using a barplot
citySpending <- select(projectData, total, city)
head(citySpending)
summary(citySpending)

cityTotal <- group_by(citySpending, city)
cityTotal <- summarize(cityTotal, sum(total))
names(cityTotal)[2] <- "spendings"
names(cityTotal)
cityTotal <- arrange(cityTotal, desc(spendings))
cityTotal

barplot(
  height=cityTotal$spendings,
  names=cityTotal$city,
  col="mistyrose",
  main="city total spending",
  xlab="City",
  ylab="Total Spending"
)

#IV-Using a boxplot to display the distribution of total spending
boxplot(
  x = projectData$total,
  col = "pink",
  main = "Distribution of total spending",
  xlab = "Total"
)


###3rd step: clustering using kmeans algorithm

#storing total spendings and age data in matrix
kmeansData <- cbind(projectData$total, projectData$age)
colnames(kmeansData) <- c("Total spendings", "Age")
rownames(kmeansData) <- projectData$customer
kmeansData

#reading number of clusters from user
numberOfClusters <- as.numeric(readline("Enter number of clusters: "))

if(numberOfClusters >= 2 && numberOfClusters <= 4) {
  #kmeans algorithm
  Total <- kmeans(kmeansData, centers=numberOfClusters)
  print(Total)
  clusteringData <- cbind(Total$cluster, projectData$age, projectData$total)
  colnames(clusteringData) <- c("Cluster Number", "Age", "Total")
  clusteringData
  
} else {
  print("Enter number between 2 and 4")
}


###4th part: generating association rules using apriori algorithm

#selecting transactions with more than one item to use in generating association rules
dataApriori <- filter(projectData, count > 1)
dim(dataApriori)
head(dataApriori)


library(arules)

#taking minimum support and minimum confidence as user input
min_support<-readline("Enter minimum support: ")

if(min_support>0.001 & min_support<1) {
  min_conf<-readline("Enter minimum confidence: ")
}

if(min_conf>0.001 & min_conf<1){
  #apriori algorithm
  transactions=strsplit(as.vector(dataApriori$items), ',')
  head(transactions)
  apriori_rules<-apriori(transactions,parameter=list(supp=as.numeric(min_support),
                                                conf=as.numeric(min_conf),minlen=2))
  summary(apriori_rules)
  inspect(apriori_rules)
  }

