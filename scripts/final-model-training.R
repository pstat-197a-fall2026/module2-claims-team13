
library(tidyverse)
library(keras3)

set_random_seed(110122)

load("data/claims-clean.RData")

all_text <- claims_clean$text_clean

binary_labels <- as.integer(claims_clean$bclass) - 1
multi_labels <- as.integer(claims_clean$mclass) - 1


#binary classification model

binary_preprocess_final <- layer_text_vectorization(
  standardize = NULL,
  split = "whitespace",
  ngrams = 2,
  max_tokens = NULL,
  output_mode = "tf_idf"
)

binary_preprocess_final %>% adapt(all_text)

binary_final_model <- keras_model_sequential() %>%
  binary_preprocess_final() %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 25, activation = "relu") %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 1) %>%
  layer_activation(activation = "sigmoid")

binary_final_model %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = "binary_accuracy"
)

binary_final_model %>% fit(
  all_text,
  binary_labels,
  epochs = 2
)


#multiclass classification model

multi_preprocess_final <- layer_text_vectorization(
  standardize = NULL,
  split = "whitespace",
  ngrams = 2,
  max_tokens = NULL,
  output_mode = "tf_idf"
)

multi_preprocess_final %>% adapt(all_text)

multi_final_model <- keras_model_sequential() %>%
  multi_preprocess_final() %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 25, activation = "relu") %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 5) %>%
  layer_activation(activation = "softmax")

multi_final_model %>% compile(
  loss = "sparse_categorical_crossentropy",
  optimizer = "adam",
  metrics = "sparse_categorical_accuracy"
)

multi_final_model %>% fit(
  all_text,
  multi_labels,
  epochs = 9
)


#save final models

dir.create("results", showWarnings = FALSE)

save_model(
  binary_final_model,
  "results/binary-final-model.keras"
)

save_model(
  multi_final_model,
  "results/multi-final-model.keras"
)
