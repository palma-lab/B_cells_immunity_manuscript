library(ggfortify)
library(cowplot)
library(ggplot2)
library(dplyr)
library(factoextra)
library(ggpubr)
library(pcaMethods)
library(wesanderson)
library("ggsci")
library(paletteer)
library(ggthemes)
library("RColorBrewer")
library(scales)
library(ggstar)
library(Hmisc)



                                                           ######################################
                                                           # TARA cohort B cells phenotypes PCA #
                                                           ######################################
#load data
data_in = R01_DB_PCA

#Outlier removal (based on noremoval PCA) 
data_in <- data_in[-c(95),] 

######################################
#             Metadata               #
######################################

Response <- data_in$`TT Response`
Group <- data_in$Group
HIV <- data_in$`HIV cp/mL`
Entry_vir <- data_in$`Entry Viremia`
WHO <- data_in$`WHO stage`
age <- data_in$`AGE IN MONTHS (days/30)`

hiv_class <- cut(HIV, breaks = c(0, 1000, 100000, 1000000, 10000000), include.lowest = TRUE)

######################################
#          FACS + Metadata           #
######################################

my_data_facs_ent <- data_in[, -c((ncol(data_in) - 11): ncol(data_in))]

######################################
#       Scaling and Centering        #
######################################

df <- my_data_facs_ent[, -c(1:6)]

df <- df[,!apply(df, MARGIN = 2, function(x) max(x, na.rm = TRUE) == min(x, na.rm = TRUE))]

df_prep <- prep(df, scale="uv", center= TRUE)

######################################
#       Response Classification      #
######################################

Response[Response == "R"] <- "P"
Response[Response == "LTM"] <- "P"
Response[Response == "PB"] <- "UP"
Response[Response == "RB"] <- "UP"
Response[Response == "UR"] <- "UP"
Response[Response == "NM"] <- "UP"
Response[Response == "MB"] <- "UP"

######################################
#                 PCA                #
######################################

pca <- prcomp(df_prep, scale = TRUE)
pca_x <- as.data.frame(pca$x)
ID = data_in[, c(1:4)]
pca_x <- merge(ID, pca_x, by.x= 0, by.y="row.names", sort=FALSE)
pca_x <- data.frame(pca_x[,-1], row.names=pca_x[,1])

######################################
#             PCA plot               #
######################################

fviz_pca_ind(pca, 
             axes = c(1,2), 
             select.var = list(contrib = 10),
             mean.point = FALSE,
             repel = TRUE, 
             choose.var = "over",
             ellipse.alpha = 0.1,
             geom = c("point"),
             labelsize = 8,
             addEllipses = FALSE, 
             ellipse.type = "norm") + 
  geom_star(aes(fill = age), starshape = 15, size = 6) +
  scale_fill_paletteer_c("grDevices::Zissou 1", name = "Age", limits = c(1, 18), breaks = c(1,5,9,10,18), labels = c(1,5,9, 10,18)) +
  guides(colour = guide_legend("Age"), size = guide_legend("Age"), shape = guide_legend("Age")) +
  guides(fill = guide_colorbar(ticks.colour = "black", draw.ulim = TRUE, draw.llim = TRUE)) + 
  labs(title ="TARA cohort B cells profiling - PCA", x = "PC1 (17.9%)", y = "PC2 (11.3%)") +
  theme_classic(base_size = 35) +
  theme(plot.title = element_text(size = 30)) +
  theme(legend.key.size = unit(1.5, 'cm'), legend.title=element_text(size=25), legend.text = element_text(size = 15)) 

######################################
#         PCA loadings plot          #
######################################

load_PC1 <- fviz_pca_var(pca,
                         axes = c(1,1),
                         col.var = "contrib", 
                         gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
                         repel = TRUE,
                         labelsize=3,
                         select.var = list(contrib = 10) ) 
load_PC1data <- as.data.frame(load_PC1$data)
load_PC1data <- load_PC1data[order(load_PC1data$x, decreasing = T),]
load_PC1data$name <- factor(load_PC1data$name, levels = load_PC1data$name)

ggplot(load_PC1data, aes(x=name, y=x)) +
  geom_segment( aes(x=name, xend=name, y=0, yend=x), color="grey", size = 4) +
  geom_point( color="#A5AA99", size=7) +
  theme_classic(base_size = 50) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.border = element_blank(),
    axis.ticks.x = element_blank() ) +
  xlab("") +
  ylab("Contribution to PC1") +
  coord_flip()

load_PC2 <- fviz_pca_var(pca,
                         axes = c(2,2),
                         col.var = "contrib", 
                         gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
                         repel = TRUE,
                         labelsize=3,
                         select.var = list(contrib = 10) ) 
load_PC2data <- as.data.frame(load_PC2$data)
load_PC2data <- load_PC2data[order(load_PC2data$x, decreasing = T),]
load_PC2data$name <- factor(load_PC2data$name, levels = load_PC2data$name)

ggplot(load_PC2data, aes(x=name, y=x)) +
  geom_segment( aes(x=name, xend=name, y=0, yend=x), color="grey", size = 3) +
  geom_point( color="#A5AA99", size=7) +
  theme_classic(base_size = 30) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.border = element_blank(),
    axis.ticks.x = element_blank() ) +
  xlab("") +
  ylab("Contribution to PC2") +
  coord_flip()

                                                           ######################################
                                                           #      HEU B cells phenotypes PCA    #
                                                           ######################################

data_in = R01_DB_PCA
data_in <- data_in[data_in$Group == "HEU", ]
#Outlier removal (based on noremoval PCA) 
data_in <- data_in[-c(95),] 

######################################
#             Metadata               #
######################################

Response <- data_in$`TT Response`
Group <- data_in$Group
HIV <- data_in$`HIV cp/mL`
Entry_vir <- data_in$`Entry Viremia`
WHO <- data_in$`WHO stage`
age <- data_in$`AGE IN MONTHS (days/30)`

hiv_class <- cut(HIV, breaks = c(0, 1000, 100000, 1000000, 10000000), include.lowest = TRUE)

######################################
#          FACS + Metadata           #
######################################

my_data_facs_ent <- data_in[, -c((ncol(data_in) - 11): ncol(data_in))]

######################################
#       Scaling and Centering        #
######################################

df <- my_data_facs_ent[, -c(1:6)]

df <- df[,!apply(df, MARGIN = 2, function(x) max(x, na.rm = TRUE) == min(x, na.rm = TRUE))]
df_prep <- prep(df, scale="uv", center= TRUE)

######################################
#       Response Classification      #
######################################

Response[Response == "R"] <- "P"
Response[Response == "LTM"] <- "P"
Response[Response == "PB"] <- "UP"
Response[Response == "RB"] <- "UP"
Response[Response == "UR"] <- "UP"
Response[Response == "NM"] <- "UP"
Response[Response == "MB"] <- "UP"

######################################
#                 PCA                #
######################################

pca <- prcomp(df_prep, scale = TRUE)
pca_x <- as.data.frame(pca$x)
ID = data_in[, c(1:4)]
pca_x <- merge(ID, pca_x, by.x= 0, by.y="row.names", sort=FALSE)
pca_x <- data.frame(pca_x[,-1], row.names=pca_x[,1])

######################################
#             PCA plot               #
######################################

fviz_pca_ind(pca, 
             axes = c(1,2), 
             select.var = list(contrib = 10),
             mean.point = FALSE,
             repel = TRUE, 
             choose.var = "over",
             ellipse.alpha = 0.1,
             geom = c("point"),
             labelsize = 8,
             addEllipses = FALSE, 
             ellipse.type = "norm") + 
  geom_star(aes(fill = age), starshape = 15, size = 6) +
  scale_fill_paletteer_c("grDevices::Zissou 1", name = "Age", limits = c(1, 18), breaks = c(1,5,9,10,18), labels = c(1,5,9, 10,18)) +
  guides(colour = guide_legend("Age"), size = guide_legend("Age"), shape = guide_legend("Age")) +
  guides(fill = guide_colorbar(ticks.colour = "black", draw.ulim = TRUE, draw.llim = TRUE)) + 
  labs(title ="HEU B cells profiling - PCA", x = "PC1 (21.6%)", y = "PC2 (11.4%)") +
  theme_classic(base_size = 35) +
  theme(plot.title = element_text(size = 30)) +
  theme(legend.key.size = unit(1.5, 'cm'), legend.title=element_text(size=25), legend.text = element_text(size = 15)) 

######################################
#         PCA loadings plot          #
######################################

load_PC1 <- fviz_pca_var(pca,
                         axes = c(1,1),
                         col.var = "contrib", 
                         gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
                         repel = TRUE,
                         labelsize=3,
                         select.var = list(contrib = 10) ) 
load_PC1data <- as.data.frame(load_PC1$data)
load_PC1data <- load_PC1data[order(load_PC1data$x, decreasing = T),]
load_PC1data$name <- factor(load_PC1data$name, levels = load_PC1data$name)

ggplot(load_PC1data, aes(x=name, y=x)) +
  geom_segment( aes(x=name, xend=name, y=0, yend=x), color="grey", size = 4) +
  geom_point( color="#A5AA99", size=7) +
  theme_classic(base_size = 50) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.border = element_blank(),
    axis.ticks.x = element_blank() ) +
  xlab("") +
  ylab("Contribution to PC1") +
  coord_flip()

load_PC2 <- fviz_pca_var(pca,
                         axes = c(2,2),
                         col.var = "contrib", 
                         gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
                         repel = TRUE,
                         labelsize=3,
                         select.var = list(contrib = 10) ) 
load_PC2data <- as.data.frame(load_PC2$data)
load_PC2data <- load_PC2data[order(load_PC2data$x, decreasing = T),]
load_PC2data$name <- factor(load_PC2data$name, levels = load_PC2data$name)

ggplot(load_PC2data, aes(x=name, y=x)) +
  geom_segment( aes(x=name, xend=name, y=0, yend=x), color="grey", size = 3) +
  geom_point( color="#A5AA99", size=7) +
  theme_classic(base_size = 30) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.border = element_blank(),
    axis.ticks.x = element_blank() ) +
  xlab("") +
  ylab("Contribution to PC2") +
  coord_flip()

######################################
#  Phenotypes Correlations with age  #
######################################

df_age = as.data.frame(df_prep)
df_age$Age = age

# Compute the correlation matrix with p-values
cor_matrix_p <- rcorr(as.matrix(df_age))

# Extract the correlation and p-values of each variable with age
cor_with_age <- cor_matrix_p$r[, "Age"]
p_values <- cor_matrix_p$P[, "Age"]

# Determine which variables are significant (p-value < 0.05)
significant_vars <- names(p_values[p_values < 0.05 ])

# Determine which variable has the lowest correlation with age
not_correlated_vars <- names(cor_with_age[abs(cor_with_age) < 0.2 & p_values[names(cor_with_age)] < 0.05])
correlated_vars <- names(cor_with_age[abs(cor_with_age) >= 0.65 & p_values < 0.05])
correlated_vars = correlated_vars[-length(correlated_vars)]

cor_df = data.frame(variable = c(correlated_vars), correlation = c(cor_with_age[correlated_vars]))
cor_df = cor_df[order(cor_df$correlation),]
cor_df$order = 1:nrow(cor_df)

nocor_df = data.frame(variable = c(not_correlated_vars), correlation = c(cor_with_age[not_correlated_vars]))
nocor_df = nocor_df[order(nocor_df$correlation),]
nocor_df$order = 1:nrow(nocor_df)

# Plot the significant variables as a horizontal barplot
ggplot(cor_df,
       aes(x = correlation, y = reorder(variable, order))) +
  geom_bar(stat = "identity", width = 0.5, fill = "lightblue") +
  geom_text(aes(x = correlation, 
                label = round(correlation, 2)), 
            hjust = 0, 
            vjust = 0.5) +
  ggtitle(paste("HEU Age Correlated Variable:", length(correlated_vars))) +
  xlab("Correlation with age") +
  theme_classic(base_size = 20) +
  ylab("Variable") 

# Plot the non correlated variables as a horizontal barplot
ggplot(nocor_df,
       aes(x = correlation, y = reorder(variable, order))) +
  geom_bar(stat = "identity", width = 0.5, fill = "lightblue") +
  geom_text(aes(x = correlation, 
                label = round(correlation, 2)), 
            hjust = 0, 
            vjust = 0.5) +
  ggtitle(paste("HEU Age Uncorrelated Variable:", length(not_correlated_vars))) +
  xlab("Correlation with age") +
  theme_classic(base_size = 20) +
  ylab("Variable") 


######################################
#      PC1 Correlation with age      #
######################################

pc_age_df = data.frame(PC1 = pca_x$PC1, age = df_age$Age)

lm_model <- lm(PC1 ~ age, data = pc_age_df)
lm_coefs <- coef(lm_model)
intercept <- lm_coefs[1]
slope <- lm_coefs[2]

# Create a text label with the regression coefficients
pval <- summary(lm_model)$coefficients[2,4]

# Calculate Pearson correlation coefficient and p-value
cor_test <- cor.test(pc_age_df$age, pc_age_df$PC1, method = "pearson")
r <- cor_test$estimate

text_label <- paste("Intercept: ", round(intercept, 2), "\nSlope: ", round(slope, 2), "\nPearson r = ", round(r, 2), "\np-value: ", signif(pval, digits=2))

# create plot
ggplot(pc_age_df, aes(x = age, y = PC1)) + 
  geom_point() + 
  geom_smooth(method = "lm") + 
  ggtitle("HEU: Correlation between PC1 values and age") + 
  xlab("Age") + 
  ylab("PC1") + 
  theme_classic(base_size = 30) +
  geom_text(x = 15, y = max(pc_age_df$PC1), label = text_label, hjust = 1, vjust = 1, size = 7) +
  theme(plot.title = element_text(size = 30)) +
  scale_x_continuous(breaks = seq(0, 18, by = 1), labels = seq(0, 18, by = 1))


                                                           ######################################
                                                           #      HEI B cells phenotypes PCA    #
                                                           ######################################


data_in = R01_DB_PCA
data_in <- data_in[data_in$Group == "HEI", ]
#Outlier removal (based on PCA without removal) 
data_in <- data_in[-c(26),] 

######################################
#             Metadata               #
######################################

Response <- data_in$`TT Response`
Group <- data_in$Group
HIV <- data_in$`HIV cp/mL`
Entry_vir <- data_in$`Entry Viremia`
WHO <- data_in$`WHO stage`
age <- data_in$`AGE IN MONTHS (days/30)`

hiv_class <- cut(HIV, breaks = c(0, 1000, 100000, 1000000, 10000000), include.lowest = TRUE)

######################################
#          FACS + Metadata           #
######################################

my_data_facs_ent <- data_in[, -c((ncol(data_in) - 11): ncol(data_in))]

######################################
#       Scaling and Centering        #
######################################

df <- my_data_facs_ent[, -c(1:6)]

df <- df[,!apply(df, MARGIN = 2, function(x) max(x, na.rm = TRUE) == min(x, na.rm = TRUE))]
df_prep <- prep(df, scale="uv", center= TRUE)

######################################
#       Response Classification      #
######################################

Response[Response == "R"] <- "P"
Response[Response == "LTM"] <- "P"
Response[Response == "PB"] <- "UP"
Response[Response == "RB"] <- "UP"
Response[Response == "UR"] <- "UP"
Response[Response == "NM"] <- "UP"
Response[Response == "MB"] <- "UP"

######################################
#                 PCA                #
######################################

pca <- prcomp(df_prep, scale = TRUE)
pca_x <- as.data.frame(pca$x)
ID = data_in[, c(1:4)]
pca_x <- merge(ID, pca_x, by.x= 0, by.y="row.names", sort=FALSE)
pca_x <- data.frame(pca_x[,-1], row.names=pca_x[,1])

######################################
#             PCA plot               #
######################################

fviz_pca_ind(pca, 
             axes = c(1,2), 
             select.var = list(contrib = 10),
             mean.point = FALSE,
             repel = TRUE, 
             choose.var = "over",
             ellipse.alpha = 0.1,
             geom = c("point"),
             labelsize = 8,
             addEllipses = FALSE, 
             ellipse.type = "norm") + 
  geom_star(aes(fill = age, size = log10(HIV)), starshape = 15) +
  scale_size(range = c(4, 14), 
             breaks = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 
             labels = sprintf("%0.0e", 10^c(0:9))) +
  scale_fill_paletteer_c("grDevices::Zissou 1", name = "Age", limits = c(1, 18), breaks = c(1,5,9,10,18), labels = c(1,5,9,10,18)) +
  guides(colour = guide_legend("Age"), 
         size = guide_legend("HIV cp/mL", override.aes = list(size = c(4, 6, 8, 10, 12, 14)))) +
  guides(fill = guide_colorbar(ticks.colour = "black", draw.ulim = TRUE, draw.llim = TRUE, override.aes = list(size = 4))) + 
  labs(title ="HEI B cells profiling - PCA", x = "PC1 (16.3%)", y = "PC2 (11.7%)") +
  theme_classic(base_size = 35) +
  theme(plot.title = element_text(size = 30)) +
  theme(legend.key.size = unit(1.1, 'cm'), legend.title=element_text(size=25), legend.text = element_text(size = 15)) 

######################################
#         PCA loadings plot          #
######################################

load_PC1 <- fviz_pca_var(pca,
                         axes = c(1,1),
                         col.var = "contrib", 
                         gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
                         repel = TRUE,
                         labelsize=3,
                         select.var = list(contrib = 10) ) 
load_PC1data <- as.data.frame(load_PC1$data)
load_PC1data <- load_PC1data[order(load_PC1data$x, decreasing = T),]
load_PC1data$name <- factor(load_PC1data$name, levels = load_PC1data$name)

ggplot(load_PC1data, aes(x=name, y=x)) +
  geom_segment( aes(x=name, xend=name, y=0, yend=x), color="grey", size = 4) +
  geom_point( color="#A5AA99", size=7) +
  theme_classic(base_size = 50) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.border = element_blank(),
    axis.ticks.x = element_blank() ) +
  xlab("") +
  ylab("Contribution to PC1") +
  coord_flip()

load_PC2 <- fviz_pca_var(pca,
                         axes = c(2,2),
                         col.var = "contrib", 
                         gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
                         repel = TRUE,
                         labelsize=3,
                         select.var = list(contrib = 10) ) 
load_PC2data <- as.data.frame(load_PC2$data)
load_PC2data <- load_PC2data[order(load_PC2data$x, decreasing = T),]
load_PC2data$name <- factor(load_PC2data$name, levels = load_PC2data$name)

ggplot(load_PC2data, aes(x=name, y=x)) +
  geom_segment( aes(x=name, xend=name, y=0, yend=x), color="grey", size = 3) +
  geom_point( color="#A5AA99", size=7) +
  theme_classic(base_size = 30) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.border = element_blank(),
    axis.ticks.x = element_blank() ) +
  xlab("") +
  ylab("Contribution to PC2") +
  coord_flip()

######################################
#  Phenotypes Correlations with age  #
######################################

df_age = as.data.frame(df_prep)
df_age$Age = age

# Compute the correlation matrix with p-values
cor_matrix_p <- rcorr(as.matrix(df_age))

# Extract the correlation and p-values of each variable with age
cor_with_age <- cor_matrix_p$r[, "Age"]
p_values <- cor_matrix_p$P[, "Age"]

# Determine which variables are significant (p-value < 0.05)
significant_vars <- names(p_values[p_values < 0.05 ])

# Determine which variable has the lowest correlation with age
not_correlated_vars <- names(cor_with_age[abs(cor_with_age) < 0.2 & p_values[names(cor_with_age)] < 0.05])
correlated_vars <- names(cor_with_age[abs(cor_with_age) >= 0.65 & p_values < 0.05])
correlated_vars = correlated_vars[-length(correlated_vars)]

cor_df = data.frame(variable = c(correlated_vars), correlation = c(cor_with_age[correlated_vars]))
cor_df = cor_df[order(cor_df$correlation),]
cor_df$order = 1:nrow(cor_df)

nocor_df = data.frame(variable = c(not_correlated_vars), correlation = c(cor_with_age[not_correlated_vars]))
nocor_df = nocor_df[order(nocor_df$correlation),]
nocor_df$order = 1:nrow(nocor_df)

# Plot the significant variables as a horizontal barplot
ggplot(cor_df,
       aes(x = correlation, y = reorder(variable, order))) +
  geom_bar(stat = "identity", width = 0.5, fill = "lightblue") +
  geom_text(aes(x = correlation, 
                label = round(correlation, 2)), 
            hjust = 0, 
            vjust = 0.5) +
  ggtitle(paste("HEU Age Correlated Variable:", length(correlated_vars))) +
  xlab("Correlation with age") +
  theme_classic(base_size = 20) +
  ylab("Variable") 

# Plot the non correlated variables as a horizontal barplot
ggplot(nocor_df,
       aes(x = correlation, y = reorder(variable, order))) +
  geom_bar(stat = "identity", width = 0.5, fill = "lightblue") +
  geom_text(aes(x = correlation, 
                label = round(correlation, 2)), 
            hjust = 0, 
            vjust = 0.5) +
  ggtitle(paste("HEU Age Uncorrelated Variable:", length(not_correlated_vars))) +
  xlab("Correlation with age") +
  theme_classic(base_size = 20) +
  ylab("Variable") 

######################################
#      PC1 Correlation with age      #
######################################

pc_age_df = data.frame(PC1 = pca_x$PC1, age = df_age$Age)

lm_model <- lm(PC1 ~ age, data = pc_age_df)
lm_coefs <- coef(lm_model)
intercept <- lm_coefs[1]
slope <- lm_coefs[2]

# Create a text label with the regression coefficients
pval <- summary(lm_model)$coefficients[2,4]

# Calculate Pearson correlation coefficient and p-value
cor_test <- cor.test(pc_age_df$age, pc_age_df$PC1, method = "pearson")
r <- cor_test$estimate

text_label <- paste("Intercept: ", round(intercept, 2), "\nSlope: ", round(slope, 2), "\nPearson r = ", round(r, 2), "\np-value: ", signif(pval, digits=2))

# create plot
ggplot(pc_age_df, aes(x = age, y = PC1)) + 
  geom_point() + 
  geom_smooth(method = "lm") + 
  ggtitle("HEU: Correlation between PC1 values and age") + 
  xlab("Age") + 
  ylab("PC1") + 
  theme_classic(base_size = 30) +
  geom_text(x = 15, y = max(pc_age_df$PC1), label = text_label, hjust = 1, vjust = 1, size = 7) +
  theme(plot.title = element_text(size = 30)) +
  scale_x_continuous(breaks = seq(0, 18, by = 1), labels = seq(0, 18, by = 1))
