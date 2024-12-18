

library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
library(bonsai)
library(lightgbm)


# train <- vroom("/kaggle/input/allstate-claims-severity/train.csv")
# test <- vroom("/kaggle/input/allstate-claims-severity/test.csv")

train <- vroom("train.csv")
test <-vroom("test.csv")

my_recipe <- recipe(loss ~ ., data = train) %>%
  step_rm(id) %>%
  step_zv(all_predictors()) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(loss)) %>%
  step_normalize(all_numeric_predictors()) # getting failed to converge messages without

# boosted model -----------------------------------------------------------

boosted_model <- boost_tree(tree_depth=tune(),
                            trees=tune(),
                            learn_rate=tune()) %>%
  set_engine("lightgbm") %>%
  set_mode("regression")

wf_boosted <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(boosted_model)

tuning_grid <- grid_regular(tree_depth(),
                            trees(),
                            learn_rate(),
                            levels = 3)

folds <- vfold_cv(train, v = 3, repeats=1)

CV_results <- wf_boosted %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(mae))

bestTune <- CV_results %>%
  select_best()

final_wf <- wf_boosted %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

preds <- predict(final_wf, new_data=test)

submission <- preds %>%
  mutate(id = test$id) %>%
  mutate(loss = .pred) %>%
  select(id, loss)

vroom_write(submission, "submission_boostedA.csv", delim = ",")


# random forest -----------------------------------------------------------

rf_model <- rand_forest(mtry = 3,
                        min_n = 2,
                        trees = 400) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")

rf_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(rf_model)

# tuning_grid <- grid_regular(mtry(range = c(1,length(train))),
min_n(),
levels = 3)

# folds <- vfold_cv(train, v = 3, repeats = 1)

# CV_results <- rf_wf %>%
#tune_grid(resamples=folds,
#grid=tuning_grid,
#metrics=metric_set(mae))

#bestTune <- CV_results %>%
#select_best()

final_wf <- rf_wf %>% 
  finalize_workflow(bestTune) %>% 
  fit(data = amazon_train)


# bart --------------------------------------------------------------------

BART_model <- parsnip::bart(trees = 100) %>% 
  set_engine("dbarts") %>% 
  set_mode("regression")

BART_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(BART_model) %>% 
  fit(data = train)

preds <- predict(BART_wf,
                 new_data = test)

submission <- preds %>%
  mutate(id = test$id) %>%
  mutate(loss = .pred) %>%
  select(id, loss)

vroom_write(submission, "submission_bart.csv", delim = ",")


