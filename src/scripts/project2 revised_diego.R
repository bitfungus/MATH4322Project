# run this 1st, Install packages
#install.packages("tidyverse")
#install.packages("caret", dependencies = TRUE)
#install.packages("rpart")
#install.packages("randomForest")


# set working directory where the csv data is at
setwd("/home/diego/temp/")

# Read the CSV data
library(tidyverse)
df <- read_csv("lithium-production.csv")
head(df)

# clean and filter the data
df_global <- df %>%
  group_by(Year) %>%
  summarise(Production = sum(lithium_production_kt, na.rm = TRUE)) %>%
  filter(Year >= 1990 & Year <= 2024)

# Dont get why this is here? manually adding an anamoly?
# df_global <- bind_rows(df_global, tibble(Year = 2024, Production = 240))

# Confirm it all looks right
print(df_global)

# run this 7th, create centered feature for modeling
df_global$Year_c <- df_global$Year - 1990
print(df_global)

# Set up 5-fold CV
library(caret)
set.seed(123)
train_ctrl <- trainControl(method = "cv", number = 5)

# LINEAR REGRESSION MODEL
lm_mod <- train(Production ~ Year_c,
                data = df_global,
                method = "lm",
                trControl = train_ctrl)
print(lm_mod)

# DECISION TREE MODEL
library(rpart)
dt_mod <- train(Production ~ Year_c,
                data = df_global,
                method = "rpart",
                trControl = train_ctrl)
print(dt_mod)

# RF MODEL
library(randomForest)
rf_mod <- train(Production ~  Year_c,
                data = df_global,
                method = "rf",
                trControl = train_ctrl,
                ntree = 500)
print(rf_mod)


POLY_ITERATIONS <- 3

adj_r2_arr = c(rep(POLY_ITERATIONS,1))

for (i in 1:POLY_ITERATIONS){
  model_x <- lm(Production ~ poly(Year_c, i), data = df_global)
  adj_r2_arr[i] = summary(model_x)$adj.r.squared
}

adj_r2_results <- tibble(
  Degree = 1:POLY_ITERATIONS,
  Adjusted_R2 = adj_r2_arr
)


# Extract the best degree
best_deg <- adj_r2_results$Degree[which.max(adj_r2_results$Adjusted_R2)]
plm_formula_str <- sprintf("Production ~ poly(Year_c, %d)", best_deg[1])
plm_formula <- as.formula(plm_formula_str)

plm_mod <- train(
  plm_formula,
  data = df_global,
  method = "lm",
  trControl = train_ctrl
)


# Comparing all models using CV
resamps <- resamples(list(LM = lm_mod, DT = dt_mod, RF = rf_mod, PLM = plm_mod))
summary(resamps)

# Visualize Predicted vs Actual for all models
df_global$pred_lm <- predict(lm_mod, df_global)
df_global$pred_rf <- predict(rf_mod, df_global)
df_global$pred_plm <- predict(plm_mod, df_global)

ggplot(df_global, aes(x = Year)) +
  geom_point(aes(y = Production), size = 3) +
  geom_line(aes(y = pred_lm, color = "Linear"), linetype = "dashed") +
  geom_line(aes(y = pred_rf, color = "Random Forest")) +
  geom_line(aes(y = pred_plm, color = "Polynomial (Degree 3)"), linetype = "dotdash") +
  scale_color_manual(values = c("Linear" = "blue", "Random Forest" = "darkgreen", "Polynomial (Degree 3)" = "purple")) +
  labs(
    title = "Global Lithium Production: Actual vs Predicted",
    y = "Production (kilotonnes)",
    color = "Model"
  ) +
  theme_minimal()



# Make sure you have dplyr and tidyr loaded:
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)

# Check column names to ensure RMSE columns exist:
print(colnames(resamps$values))

# Create the dataframe for boxplot (adapt if column names differ)
cv_df <- resamps$values %>%
  select(contains("RMSE")) %>%
  pivot_longer(cols = everything(), names_to = "Model", values_to = "RMSE") %>%
  mutate(Model = case_when(
    str_detect(Model, "^LM") ~ "Linear Regression",
    str_detect(Model, "^PLM") ~ "Polynomial Linear Regression",
    str_detect(Model, "^DT") ~ "Decision Tree",
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

# Optional: Save plot as image
ggsave("plot_slide4_cv_comparison.png", width = 8, height = 5, dpi = 300)
