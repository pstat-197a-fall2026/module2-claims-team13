
require(tidyverse)
require(keras3)
require(tensorflow)
library(reticulate)

load('data/claims-test.RData')
load('data/claims-clean.RData')

source('scripts/preprocessing.R')

keras_layers <- import("keras.layers")

binary_model <- keras_layers$TFSMLayer(
  "results/binary-final-model",
  call_endpoint = "serve"
)

multi_model <- keras_layers$TFSMLayer(
  "results/multi-final-model",
  call_endpoint = "serve"
)

tf <- import("tensorflow")

# apply preprocessing pipeline
clean_df <- claims_test %>%
  mutate(
    text_clean = map_chr(
      text_tmp,
      ~ tryCatch(parse_fn(.x), error = function(e) NA_character_)
    ),
    text_clean = na_if(str_trim(text_clean), "")
  ) %>%
  select(.id, text_clean)

# grab input
valid_rows <- which(!is.na(clean_df$text_clean))

x <- clean_df %>%
  slice(valid_rows) %>%
  pull(text_clean)

# compute predictions
binary_labels <- levels(claims_clean$bclass)
multi_labels <- levels(claims_clean$mclass)

binary_classes <- rep(binary_labels[1], nrow(clean_df))
multi_classes <- rep(multi_labels[1], nrow(clean_df))

x_tensor <- tf$constant(x, dtype = tf$string)

binary_preds <- binary_model(x_tensor) %>%
  as.array() %>%
  as.numeric()

multi_preds <- multi_model(x_tensor) %>%
  as.array()

binary_classes[valid_rows] <- ifelse(
  binary_preds > 0.5,
  binary_labels[2],
  binary_labels[1]
)

multi_classes[valid_rows] <- multi_labels[
  max.col(multi_preds)
]

# export (KEEP THIS FORMAT IDENTICAL)
pred_df <- clean_df %>%
  mutate(
    bclass.pred = factor(binary_classes, levels = binary_labels),
    mclass.pred = factor(multi_classes, levels = multi_labels)
  ) %>%
  select(.id, bclass.pred, mclass.pred)

save(pred_df, file = 'results/preds-group13.RData')
