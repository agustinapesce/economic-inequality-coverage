library("stm")
library("tm")
library(splines)

library(ggplot2)
library(jsonlite)
library(dplyr)

#----------DF IMPORT------------------------------------------


python_df <- read.csv("../2_stm_prepro/articles_speeches_dataset_r.csv", 
               stringsAsFactors = FALSE)

names(python_df)
#head(python_df)

table(python_df$dataset)
#Congress      nyt 
#3777        10681

table(python_df$source)
#Democrat: 2592
#Republican: 1177
#Independent: 6
#nyt: 10681


stm_data <- python_df[, c("id_", "author", "year", "date", "year_n",  
                          "source", "dataset", "text", "tokens_R")]

#---------Corpus & vocabulary-----------------------------------

#Preprocessing was done in Python in stm_preprocessing.ipynb with SpaCy
#(lowercase, digits, punctuation, stopwords, lemmatization)
processed <- textProcessor(stm_data$tokens_R, metadata = stm_data,  
                           lowercase = FALSE,
                           removenumbers = FALSE,
                           removepunctuation = FALSE,
                           removestopwords = FALSE,
                           stem = FALSE)

#Remove unfrequent words (in less than 1% of documents)
lower_bound <- ceiling(0.01 * nrow(stm_data)) 
out <- prepDocuments(processed$documents, processed$vocab,
                     processed$meta, lower.thresh = lower_bound)

#> nrow(stm_data)
#[1] 14458

#Removing 110890 of 115727 terms (836689 of 4995882 tokens) due to frequency 
#Your corpus now has 14458 documents, 4837 terms and 4159193 tokens.

#Metadata in out$meta
out$meta$dataset <- as.factor(out$meta$dataset)
out$meta$year_n <- as.integer(out$meta$year_n)

#write.csv(out$meta, "out_meta_stm.csv", row.names = FALSE)

#out_meta <- read.csv("out_meta_stm.csv", 
#               stringsAsFactors = FALSE)

#-------- Function to extract documents annotations for the created models for both datasets ----
# Function to process STM model and export topic prevalence data
process_topic_prevalence <- function(stm_model, out, k, seed) {
  # Ensure theta_matrix is a data frame and has topic column names
  theta_df <- as.data.frame(stm_model$theta)
  colnames(theta_df) <- seq_len(ncol(theta_df))  # Rename columns as 1, 2, 3, ...
  
  # Add metadata to theta_matrix
  combined_df <- cbind(year = out$meta$year, 
                       dataset = out$meta$dataset, 
                       theta_df)
  
  # Data frame for the "Congress" dataset
  congress_df <- combined_df %>%
    filter(dataset == "Congress") %>%  # Filter rows for "Congress"
    group_by(year) %>%
    summarise(across(where(is.numeric), mean, na.rm = TRUE))  # Summarize numeric columns
  
  # Data frame for the "nyt" dataset
  nyt_df <- combined_df %>%
    filter(dataset == "nyt") %>%  # Filter rows for "nyt"
    group_by(year) %>%
    summarise(across(where(is.numeric), mean, na.rm = TRUE))  # Summarize numeric columns
  
  # Export both data frames to separate CSV files
  congress_filename <- paste0("stm_fitted_models/stm_topic_prev_congress_k", k, "_s", seed, ".csv")
  nyt_filename <- paste0("stm_fitted_models/stm_topic_prev_nyt_k", k, "_s", seed, ".csv")
  
  write.csv(congress_df, congress_filename, row.names = FALSE)
  write.csv(nyt_df, nyt_filename, row.names = FALSE)
  
  cat("Saved topic prevalence data for Congress and nyt datasets\n")
}

#---------Model fitting-------------------

ks = c(40,45,50,55,60,65,70,75)
seeds = c(3) #seeds dont change results with spectral initialization
#seeds = c(3,50,100)

#help(ns)
years_seq <- seq(1, 45, by = 1)
quantile_knots <- as.numeric(quantile(years_seq, probs = c(0.25, 0.5, 0.75)))

years_seq_2 <- seq(1980, 2024, by = 1)
quantile_knots_2 <- quantile(years_seq_2, probs = c(0.25, 0.5, 0.75))
#1991, 2002, 2013

#Loop over 7 Ks and 3 runs seeds
for (k in ks) {
  for (seed in seeds) {
    set.seed(seed)  # Set seed for reproducibility

    model <- stm(
      documents = out$documents, 
      vocab = out$vocab,
      data = out$meta,
      prevalence =~ dataset * ns(year_n, knots = quantile_knots, #df =5 (df - 1 - intercept = knots)
                                 Boundary.knots = c(1, 45)),
      #prevalence =~ dataset * s(year_n), 
      content =~ dataset,
      K = k,
      seed = seed,
      max.em.its = 75,
      init.type = "Spectral" #c("Spectral" (spectral algorithm for latent Dirichlet allocation, method of moments, recomended) , 
                             #"LDA", (collapsed Gibbs sampler)
                             #"Random",  (random starting values)
                             #"Custom", (user specified starting values)
    )
    
    filename <- paste0("stm_fitted_models/stm_K", k, "_seed", seed, ".rds")
    saveRDS(model, file = filename)
    cat("Saved model:", filename, "\n")
    
    # Save the plot as a PDF
    plot_filename <- paste0("plots/top_topics_K", k, "_seed", seed, ".pdf")
    pdf(plot_filename, width = 7, height = 15)
    plot(model, type = "summary", xlim = c(0, 0.3))
    dev.off()
    cat("Saved plot:", plot_filename, "\n")
    
    # Generate labelTopics output
    topic_labels <- labelTopics(model, c(1:k))  # Label all topics
    
    # Save labelTopics output to a text file
    labels_filename <- paste0("plots/topics_labels_K", k, "_seed", seed, ".txt")
    capture.output(print(topic_labels), file = labels_filename)
    cat("Saved topic labels:", labels_filename, "\n")
    
    # Process and save topic prevalence data
    process_topic_prevalence(model, out, k, seed)
  }
}

#K 40-75 models with model:
#prevalence =~ dataset * ns(year_n, df = 5, Boundary.knots = c(1, 45)),
#content =~ dataset

#--------- Extract words from the selected model to create intrusion task validation -----------------

stm_model <- readRDS("stm_fitted_models/stm_K70_seed3.rds")

### CSV for the intrusion task
k = 70
topic_labels <- labelTopics(stm_model, c(1:k))
write.csv(topic_labels$topics, "intrusion_words.csv", row.names = FALSE)
#go to get_data_from_stm.ipynb in intrusion_task folder


#------------------------------------------------------------------------------

#------------------- Topic Models Exploration through documents reading --------------------------------------

#Find 15 most relevant documents for a topic with findThoughts
#out$meta
#out_meta
topic_n = 3
thoughts <- findThoughts(stm_model, 
                          n = 15, 
                          texts = out_meta$text,
                          topics = c(topic_n))$docs

file_conn <- file("thoughts.txt")
writeLines(sapply(seq_along(thoughts), function(i) {
  paste0("Element ", i, ":\n", thoughts[[i]], "\n")
}), file_conn)
close(file_conn)

#If all docs correspond to a dataset check examples for the other dataset
dataset = "nyt"
dataset = "Congress"

# Subset theta matrix (prevalences) and metadata
subset_indices <- which(out_meta$dataset == dataset)

subset_theta <- stm_model$theta[subset_indices, ]
subset_meta <- out_meta[subset_indices, ]

# Get the topic proportions for the assessed topic
topic_pro <- subset_theta[, topic_n]

# Find the indices of the top 10 highest values within the subset
top_10_indices <- order(topic_pro, decreasing = TRUE)[1:10]

# Retrieve the metadata of the top documents
top_docs_meta <- subset_meta[top_10_indices, ]
top_docs_theta <- subset_theta[top_10_indices, ]

# Display relevant information
top_docs_info <- data.frame(
  ID = top_docs_meta$id_,
  Year = top_docs_meta$year,
  Text = top_docs_meta$text,
  Dataset = top_docs_meta$dataset,
  prop = topic_pro[top_10_indices]
)

#see year and proportions columns
top_docs_info$Year
top_docs_info$prop

# View the resulting data
i = 3
top_docs_info$Year[i]
top_docs_info$prop[i]
top_docs_info$Text[i]

######
# Selfmade way to get the indices of the 10 highest values manually and look at variables
topic_n = 52
topic_docs <- stm_model$theta[, topic_n]
top_10_docs <- order(topic_docs, decreasing = TRUE)[1:10]

doc_n = 5468
out$meta$id_[doc_n]
out$meta$year[doc_n]
out$meta$text[doc_n]
out$meta$date[doc_n]

#----------- Combine theta_matrix with metadata to export prevalences for python processing -----------------

#unique(out$meta$dataset) #Congress nyt
theta_matrix <- stm_model$theta

# Ensure theta_matrix is a data frame and has topic column names
theta_df <- as.data.frame(theta_matrix)
colnames(theta_df) <- seq_len(ncol(theta_df))  # Rename columns as 1, 2, 3, ...

# Add metadata to theta_matrix
combined_df <- cbind(year = out_meta$year, dataset = out_meta$dataset, theta_df)

# Data frame for the "Congress" dataset
congress_df <- combined_df %>%
  filter(dataset == "Congress") %>%  # Filter rows for "Congress"
  group_by(year) %>%
  summarise(across(where(is.numeric), mean, na.rm = TRUE))  # Summarize numeric columns

# Data frame for the "nyt" dataset
nyt_df <- combined_df %>%
  filter(dataset == "nyt") %>%  # Filter rows for "nyt"
  group_by(year) %>%
  summarise(across(where(is.numeric), mean, na.rm = TRUE))  # Summarize numeric columns

# Export both data frames to separate CSV files
write.csv(congress_df, "stm_topic_prev_congress.csv", row.names = FALSE)
write.csv(nyt_df, "stm_topic_prev_nyt.csv", row.names = FALSE)


#------------------- Rank 1 calculations (histogram with documents most prevalent topic) ----------------

theta_matrix <- stm_model$theta

#boiler_col <- c(1,2,10,21,29,31,37,40,41,46,47,48,50)
#theta_matrix_noboiler <- theta_matrix[, -boiler_col]
dim(theta_matrix)
#length(boiler_col)
#dim(theta_matrix_noboiler)

# Map the original column indices (before removing columns) to the plot labels
#original_topic_names <- setdiff(1:ncol(theta_matrix), boiler_col)  # Original topic indices, excluding the removed ones
#original_topic_names <- setdiff(1:ncol(theta_matrix), boiler_col)
original_topic_names <- seq(1:ncol(theta_matrix))


# Step 1: Find the index of the topic with the highest value for each document
primary_topic_indices <- apply(theta_matrix, 1, function(x) which.max(x))
#primary_topic_indices <- apply(theta_matrix_noboiler, 1, function(x) which.max(x))

# Step 2: Calculate how many times each topic appears in any document (including primary and non-primary)
#topic_occurrence_count <- colSums(theta_matrix_noboiler > 0)
topic_occurrence_count <- colSums(theta_matrix > 0)

# Step 3: Calculate how many times each topic is the primary topic
primary_topic_count <- table(primary_topic_indices)

# Step 4: Make sure primary_topic_count has the same length as topic_occurrence_count
# Create a vector of zeros for topics that don't have a primary topic
rank_1_percentage <- rep(0, ncol(theta_matrix))  # Initialize to zeros
#rank_1_percentage <- rep(0, ncol(theta_matrix_noboiler))

# Fill in the rank_1_percentage with the proper values from primary_topic_count
rank_1_percentage[as.numeric(names(primary_topic_count))] <- primary_topic_count / topic_occurrence_count[as.numeric(names(primary_topic_count))] * 100

# Step 5: Sort topics by rank_1_percentage in descending order
sorted_indices <- order(rank_1_percentage, decreasing = TRUE)
rank_1_percentage_sorted <- sort(rank_1_percentage, decreasing = TRUE)
original_topic_names
original_topic_names_sorted <- original_topic_names[sorted_indices]

# Prepare the data for ggplot
rank_1_percentage_df <- data.frame(
  Original_Topic = original_topic_names_sorted,
  Percentage = rank_1_percentage_sorted  # Corresponding percentages
)
#rank_1_percentage_df$Original_Topic <- as.factor(rank_1_percentage_df$Original_Topic)
rank_1_percentage_df$Original_Topic <- factor(rank_1_percentage_df$Original_Topic, levels = rank_1_percentage_df$Original_Topic)

ggplot(rank_1_percentage_df, aes(x = Original_Topic, y = Percentage)) +
  geom_bar(stat = "identity", fill = "lightblue", width = 0.8) +  # Horizontal bars with wider width
  coord_flip() +  # Flip the axes to make bars horizontal
  labs(
    title = "Percentage of Times Each Topic is Primary",
    x = "Topics",
    y = "Percentage (%)"
  ) +
  theme_minimal() +  # Cleaner theme
  theme(
    axis.text.x = element_text(size = 12),  # Increase x-axis text size
    axis.text.y = element_text(size = 10),  # Increase y-axis text size
    plot.title = element_text(size = 14, face = "bold")  # Style the title
  )
