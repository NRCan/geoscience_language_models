# Load libraries
```
#load libraries
library(tidyverse)
```



# Load word vectors in Word2Vec format
```
#load data using tidyverse functions
myvecs 		<- read_table("FILEPATH HERE", skip = 1, col_names = FALSE)
mydf 		<- as.data.frame(myvecs)
```



# Create a list of vocabularies to filter
```
myvocab <- c("claystone", "clay", "sandstone", "sand", "silt", "siltstone", "gravel", "conglomerate")
```



# Filter preferred vocabularies
```
mydf_filter <-	filter(mydf, mydf[,1] %in% myvocab)
```



# Prepare data for pca
```
forpca <-	mydf_filter %>%
		dplyr::select(-X1) %>%
		as.data.frame()
```



# Calculate PCA

```
mypca <-	prcomp(forpca) 
plot(mypca)
biplot(mypca)
summary(mypca)
```
