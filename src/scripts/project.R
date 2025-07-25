
library(glue)
library(randomForest)
library(tidyverse) # for ggplot
library(caret)
library(tree)

set.seed(123)


cwd <- "/home/diego/Downloads/FinalReport/src" # change this to your cwd where the dataset is in
df <- read_csv(glue("{cwd}/data/lithium-production.csv")) 

# Postprocessing
df_global <- df %>%
  group_by(Year) %>%
  summarise(Production = sum(lithium_production_kt, na.rm = TRUE)) %>%
  filter(Year >=  2019 & Year <= 2024)

df_global$Year_c <- df_global$Year - 2019

rmse_list = rep(0, 10)

for (i in 1:10){
  train_index <- sample(1:nrow(df_global), size = 0.8 * nrow(df_global))
  train_data <- df_global[train_index, ]
  test_data <- df_global[-train_index, ]
  
  lm_model <- lm(Production ~ Year_c, data = train_data)
  
  predictions <- predict(lm_model, newdata = test_data)
  
  rmse <- sqrt((mean(test_data$Production - predictions)^2))
  rmse_list[i] <- rmse
}

glue("Mean Test RMSE: {mean(rmse_list)}")


train_ctrl <- trainControl(method = "cv", number = 5)
lm_model <- train(Production ~ Year_c,
                  data = df_global,
                  method = "lm",
                  trControl = train_ctrl)


library(tree)
RMSE_decision_tree = rep(0, 10)
for (i in 1:10){
  train_idx = sample(1:nrow(df_global), size = 0.8 * nrow(df_global))
  train_data <- df_global[train_idx, ]
  test_data <- df_global[-train_idx, ]
  
  tree_model <- tree(Production ~ Year_c, data = df_global, subset = train_idx)
  
  yhat = predict(tree_model, newdata=test_data)
  RMSE_decision_tree[i] = sqrt(mean(as.numeric(test_data$Production) - as.numeric(yhat))^2) 
}
mean(RMSE_decision_tree)

train_ctrl <- trainControl(method = "cv", number = 5)
ntree_vals <- c(100,200,300,400,500,600,700,800,900,1000)
rmse_vals_ntree <- numeric(length(ntree_vals))

for(i in seq_along(ntree_vals)){
  rf_tmp <- train(
    Production ~ Year_c,
    data = df_global,
    method = "rf",
    trControl = trainControl(method = "cv", number = 5),
    ntree = ntree_vals[i]
  )
  rmse_vals_ntree[i] <- min(rf_tmp$results$RMSE)
}

(rmse_ntree_df <- data.frame(ntree = ntree_vals, RMSE = rmse_vals_ntree))


best_ntree <- rmse_ntree_df[which.min(rmse_ntree_df$RMSE), ]$ntree
rf_model <- train(Production ~ Year_c, data = df_global, method = "rf",
                  trControl = train_ctrl, ntree = best_ntree)


resamps <- resamples(list(LM = lm_model, RF = rf_model))


df_global$pred_lm <- predict(lm_model, df_global)
df_global$pred_rf <- predict(rf_model, df_global)

ggplot(df_global, aes(x = Year)) +
  geom_point(aes(y = Production), size = 3) +
  geom_line(aes(y = pred_lm, color = "Linear"), linetype = "dashed") +
  geom_line(aes(y = pred_rf, color = "Random Forest")) +
  scale_color_manual(values = c("Linear" = "blue", "Random Forest" = "darkgreen")) +
  labs(
    title = "Global Lithium Production: Actual vs Predicted",
    y = "Production (kilotonnes)",
    color = "Model"
  ) +
  theme_minimal()


cv_df <- resamps$values %>%
  select(contains("RMSE")) %>%
  pivot_longer(cols = everything(), names_to = "Model", values_to = "RMSE") %>%
  mutate(Model = case_when(
    str_detect(Model, "^LM") ~ "Linear Regression",
    str_detect(Model, "^RF") ~ "Random Forest",
    TRUE ~ Model
  ))


# Boxplot of RMSE values across models
ggplot(cv_df, aes(x = Model, y = RMSE, fill = Model)) +
  geom_boxplot() +
  labs(title = "Slide 4: Model Performance (5-Fold Cross-Validation)",
       x = "Model",
       y = "Root Mean Squared Error (RMSE)") +
  theme_minimal() +
  theme(legend.position = "none")