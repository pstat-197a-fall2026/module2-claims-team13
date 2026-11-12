
library(tidyverse)
library(tidymodels)
library(keras3)
library(textrecipes)
library(kernlab)

set_random_seed(110122)

load("data/claims-clean.RData")

names(claims_clean)
table(claims_clean$bclass)
table(claims_clean$mclass)




#split data

set.seed(110122)

partitions <- claims_clean %>%
  initial_split(prop = 0.8, strata = mclass)

train_data <- training(partitions)
valid_data <- testing(partitions)

train_text <- train_data %>% pull(text_clean)
valid_text <- valid_data %>% pull(text_clean)


#binary classification labels
train_labels <- as.integer(train_data$bclass) - 1
valid_labels <- as.integer(valid_data$bclass) - 1

#multiclass classification labels 
train_mlabels <- as.integer(train_data$mclass) - 1
valid_mlabels <- as.integer(valid_data$mclass) - 1


#text preprocessing layer
binary_preprocess <- layer_text_vectorization(
  standardize = NULL,
  split = "whitespace",
  ngrams = 2,
  max_tokens = NULL,
  output_mode = "tf_idf"
)


#preprocessing layer using training text only
binary_preprocess %>% adapt(train_text)



binary_model <- keras_model_sequential() %>%
  binary_preprocess() %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 25, activation = "relu") %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 1) %>%
  layer_activation(activation = "sigmoid")


binary_model %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = "binary_accuracy"
)

#early stopping
early_stop <- callback_early_stopping(
  monitor = "val_loss",
  patience = 2,
  restore_best_weights = TRUE
)

#train model
binary_history <- binary_model %>%
  fit(
    train_text,
    train_labels,
    validation_data = list(valid_text, valid_labels),
    epochs = 20,
    callbacks = list(early_stop)
  )

#evaluate model
binary_model %>% evaluate(valid_text, valid_labels)


#generate predictions
binary_prob <- binary_model %>%
  predict(valid_text)

binary_pred <- ifelse(
  as.vector(binary_prob) >= 0.5,
  levels(claims_clean$bclass)[2],
  levels(claims_clean$bclass)[1]
)

#confusion matrix
binary_results <- tibble(
  truth = valid_data$bclass,
  estimate = factor(
    binary_pred,
    levels = levels(claims_clean$bclass)
  )
)

conf_mat(binary_results, truth = truth, estimate = estimate)



#text preprocessing layer
multi_preprocess <- layer_text_vectorization(
  standardize = NULL,
  split = "whitespace",
  ngrams = 2,
  max_tokens = NULL,
  output_mode = "tf_idf"
)


#preprocessing layer using training text only
multi_preprocess %>% adapt(train_text)



multi_model <- keras_model_sequential() %>%
  multi_preprocess() %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 25, activation = "relu") %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 5) %>%
  layer_activation(activation = "softmax")


multi_model %>% compile(
  loss = "sparse_categorical_crossentropy",
  optimizer = "adam",
  metrics = "sparse_categorical_accuracy"
)


#early stopping
multi_early_stop <- callback_early_stopping(
  monitor = "val_loss",
  patience = 2,
  restore_best_weights = TRUE
)

#train model
multi_history <- multi_model %>%
  fit(
    train_text,
    train_mlabels,
    validation_data = list(valid_text, valid_mlabels),
    epochs = 20,
    callbacks = list(multi_early_stop)
  )

#evaluate model
multi_model %>% evaluate(valid_text, valid_mlabels)


#generate predictions
multi_prob <- multi_model %>%
  predict(valid_text)

multi_pred <- levels(claims_clean$mclass)[
  max.col(multi_prob)
]

#confusion matrix
multi_results <- tibble(
  truth = valid_data$mclass,
  estimate = factor(
    multi_pred,
    levels = levels(claims_clean$mclass)
  )
)

conf_mat(multi_results, truth = truth, estimate = estimate)






#binary SVM model

binary_svm_recipe <- recipe(bclass ~ text_clean, data = train_data) %>%
  step_tokenize(text_clean) %>%
  step_tokenfilter(text_clean, max_tokens = 5000) %>%
  step_tfidf(text_clean)

svm_spec <- svm_linear(
  cost = 1
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

binary_svm_workflow <- workflow() %>%
  add_recipe(binary_svm_recipe) %>%
  add_model(svm_spec)

binary_svm_fit <- binary_svm_workflow %>%
  fit(data = train_data)


#generate predictions
binary_svm_pred <- predict(
  binary_svm_fit,
  new_data = valid_data
) %>%
  bind_cols(valid_data %>% select(bclass))

#evaluate model
accuracy(
  binary_svm_pred,
  truth = bclass,
  estimate = .pred_class
)

#confusion matrix
conf_mat(
  binary_svm_pred,
  truth = bclass,
  estimate = .pred_class
)




#multiclass SVM model

multi_svm_recipe <- recipe(mclass ~ text_clean, data = train_data) %>%
  step_tokenize(text_clean) %>%
  step_tokenfilter(text_clean, max_tokens = 5000) %>%
  step_tfidf(text_clean)

multi_svm_workflow <- workflow() %>%
  add_recipe(multi_svm_recipe) %>%
  add_model(svm_spec)

multi_svm_fit <- multi_svm_workflow %>%
  fit(data = train_data)


#generate predictions
multi_svm_pred <- predict(
  multi_svm_fit,
  new_data = valid_data
) %>%
  bind_cols(valid_data %>% select(mclass))

#evaluate model
accuracy(
  multi_svm_pred,
  truth = mclass,
  estimate = .pred_class
)

#confusion matrix
conf_mat(
  multi_svm_pred,
  truth = mclass,
  estimate = .pred_class
)
