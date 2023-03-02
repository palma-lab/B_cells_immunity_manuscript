###Author Marco Sanna (marco-sanna) 02/03/2023

                                                           ######################################
                                                           #    Entry B cells phenotypes PCA    #
                                                           ######################################

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

####### Function for plotting by Group

color_by_group <- function(pca_x, xaxis, yaxis) {
  p <- ggplot(pca_x, aes(y = .data[[yaxis]], x = .data[[xaxis]]))
  p + geom_star(aes(fill = Group), starshape = 15, size = 2.2) +
    theme_classic(base_size = 22) +
    theme(legend.title = element_blank(), legend.position="none") + 
    scale_fill_manual(values=c("#E48725", "#2F8AC4")) + 
    labs(x=xaxis, y=yaxis) }

data_in = R01_DB_PCA
data_in <- data_in[data_in$Age == "entry", ]

#Outlier removal (based on PCA without removal) 
data_in <- data_in[-c(1, 53, 64),] 

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
#             PCA plots              #
######################################

# PCA pairs PC1:PC5
PCA12 <- color_by_group(pca_x, "PC1", "PC2") 
PCA13 <- color_by_group(pca_x, "PC1", "PC3") 
PCA23 <- color_by_group(pca_x, "PC2", "PC3") 
PCA14 <- color_by_group(pca_x, "PC1", "PC4")
PCA24 <- color_by_group(pca_x, "PC2", "PC4")
PCA34 <- color_by_group(pca_x, "PC3", "PC4")
PCA15 <- color_by_group(pca_x, "PC1", "PC5")
PCA25 <- color_by_group(pca_x, "PC2", "PC5")
PCA35 <- color_by_group(pca_x, "PC3", "PC5")
PCA45 <- color_by_group(pca_x, "PC4", "PC5")

legend <- get_legend(ggplot(pca_x, aes(x=PC1, y=PC2)) +
                       #geom_point(aes(color=Group), size = 4) +
                       geom_star(aes(fill=Group), starshape = 15, size = 4) +
                       theme_classic(base_size = 20) +
                       theme(legend.title = element_blank()) + 
                       scale_fill_manual(values=c("#E48725", "#2F8AC4")) +
                       #scale_shape_manual(values=c("P" = 19, "UP" = 1)) +
                       #labs(shape="Response") +
                       theme(legend.box.margin = margin(0, 0, 0, 12)) +
                       theme(legend.key.size = unit(1, 'cm'), legend.title=element_text(size=20))
) # create some space to the left of the legend


plot_grid(PCA12, NULL, NULL, legend, 
          PCA13, PCA23, NULL, NULL, 
          PCA14, PCA24, PCA34, NULL, 
          PCA15, PCA25, PCA35, PCA45,
          ncol = 4)

fviz_pca_ind(pca, 
             axes = c(4,5), 
             select.var = list(contrib = 10),
             col.var = "black",
             mean.point = FALSE,
             repel = TRUE, 
             habillage = Group,
             ellipse.alpha = 0.1,
             geom = c("point"),
             labelsize = 7,
             pointsize = 3,
             pointshape = 19,
             addEllipses = FALSE, 
             ellipse.type = "norm") + 
  geom_star(aes(fill=Group), size = 5, starshape = 15) +
  scale_fill_manual(values=colgr) +
  labs(title ="Baseline B cells phenotypes - PCA", x = "PC4 (7.7%)", y = "PC5 (6.1%)") +
  theme_classic(base_size = 30) +
  theme(legend.key.size = unit(1.5, 'cm')) 

######################################
#         PCA loadings plot          #
######################################

load_PC4 <- fviz_pca_var(pca,
                         axes = c(4,4),
                         col.var = "contrib", 
                         gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
                         repel = TRUE,
                         labelsize=2,
                         select.var = list(contrib = 10) ) 
load_PC4data <- as.data.frame(load_PC4$data)
load_PC4data <- load_PC4data[order(load_PC4data$x, decreasing = T),]
load_PC4data$name <- factor(load_PC4data$name, levels = load_PC4data$name)

ggplot(load_PC4data, aes(x=name, y=x)) +
  geom_segment( aes(x=name, xend=name, y=0, yend=x), color="grey", size = 4) +
  geom_point( color="#A5AA99", size=7) +
  theme_classic(base_size = 50) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.border = element_blank(),
    axis.ticks.x = element_blank() ) +
  xlab("") +
  ylab("Contribution to PC4") +
  coord_flip()


load_PC5 <- fviz_pca_var(pca,
                         axes = c(5,5),
                         col.var = "contrib", 
                         gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
                         repel = TRUE,
                         labelsize=2,
                         select.var = list(contrib = 10) ) 
load_PC5data <- as.data.frame(load_PC5$data)
load_PC5data <- load_PC5data[order(load_PC5data$x, decreasing = T),]
load_PC5data$name <- factor(load_PC5data$name, levels = load_PC5data$name)

ggplot(load_PC5data, aes(x=name, y=x)) +
  geom_segment( aes(x=name, xend=name, y=0, yend=x), color="grey", size = 4) +
  geom_point( color="#A5AA99", size=7) +
  theme_classic(base_size = 50) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.border = element_blank(),
    axis.ticks.x = element_blank() ) +
  xlab("") +
  ylab("Contribution to PC5") +
  coord_flip()
