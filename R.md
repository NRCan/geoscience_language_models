# Load libraries
'''
#load libraries
library(tidyverse)
'''



# Load word vectors in Word2Vec format
'''
#load data using tidyverse functions
embeddings 	<- read_table("FILEPATH HERE", skip = 1, col_names = FALSE)
df 		<- as.data.frame(embeddings)
'''



# Create a list of vocabularies to filter
'''
mycols <- c(
"claystone",
"clay",
"sandstone",
"sand",
"silt",
"siltstone",
"gravel",
"conglomerate")
'''



# Filter preferred vocabularies
'''
df_select <-	filter(df, df[,1] %in% mycols)
'''



# Prepare data for pca
'''
forpca <-	myselect %>%
		dplyr::select(-X1) %>%
		as.data.frame()
'''



# Calculate PCA

'''
mypca <-	prcomp(forpca) 
plot(mypca)
biplot(mypca)
summary(mypca)
'''
