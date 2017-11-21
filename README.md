# caraccident


Some exercises with PANDAS 
Questions below

The UK has a dataset(https://data.gov.uk/dataset/road-accidents-safety-data) on vehicle
accidents. Please dowload the "2014 All STATS19 data (accident, casualties and vehicle tables) for
2005 to 2014." Information on the variables can be found at the bottom of the page under additional
links. In addition, the form which is used to record data by police officers can be found
here(http://docs.adrn.ac.uk/888043/mrdoc/pdf/888043_stats19-road-accident-injury-statisticsreport-
form.pdf).
1- What fraction of accidents occur in urban areas? Report the answer in decimal form.
2- When is the most dangerous time to drive? Find the hour of the day that has the highest
occurance of fatal accidents, normalized by the total number of accidents that occured in
that hour. For your answer, submit the corresponding frequency of fatal accidents to all
accidents in that hour. Note: round accident times down. For example, if an accident occured
at 23:55 it occured in hour 23.
3- There appears to be a linear trend in the number of accidents that occur each year. What is
that trend? Return the slope in units of increased number of accidents per year.
4- Do accidents in highspeedlimit areas have more casualties? Compute the Pearson correlation 
coefficient between the speed limit and the ratio of the number of casualties to accidents 
for each speed limit. Bin the data by speed limit.
5- How many times more likely are you to be in an accident where you skid, jackknife, or
overturn (as opposed to an accident where you don't) when it's raining or snowing compared
to nice weather with no high winds? Ignore accidents where the weather is unknown or
missing.
6- How many times more likely are accidents involving male car drivers to be fatal compared to
accidents involving female car drivers? The answer should be the ratio of fatality rates of
males to females. Ignore all accidents where the driver wasn't driving a car.
7- We can use the accident locations to estimate the areas of the police districts. Represent
each as an ellipse with semiaxes
given by a single standard deviation of the longitude and
latitude. What is the area, in square kilometers, of the largest district measured in this
manner?
8- How fast do the number of car accidents drop off with age? Only consider car drivers who
are legally allowed to drive in the UK (17 years or older). Find the rate at which the number of
accidents exponentially decays with age. Age is measured in years. Assume that the number
of accidents is exponentially distributed with age for driver's over the age of 17.
