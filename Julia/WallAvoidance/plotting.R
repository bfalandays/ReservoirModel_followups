rm(list = ls(all = TRUE))

library(data.table)
library(tidyverse)
library(ggplot2)
library(rCAT)
library(BAMBI)
library(plotly)
library(gplots)


setwd('/Users/jfalanda/Documents/GitHub/ReservoirModel_followups/Julia/WallAvoidance')

pos = as.data.table(read.csv("Data/pos_base.csv"))
spikes = as.data.table(read.csv("Data/spikes_base.csv"))
heading = as.data.table(read.csv("Data/heading_base.csv"))
heading$heading = zero_to_2pi(heading$heading)
sens= as.data.table(read.csv("Data/sens_base.csv"))
eff = as.data.table(read.csv("Data/eff_base.csv"))
hits = as.data.table(read.csv("Data/hits_base.csv"))

t = seq(1,nrow(pos))
pos$t = t
heading$t = t
sens$t = t
eff$t = t
hits$t = t

ggplot(pos, aes(x = X1, y = X2)) + geom_path(alpha=pos$t/1000, arrow = arrow(length = unit(0.1, "inches")))+
  theme(axis.title.x = element_blank(),
        axis.title.y = element_blank())
ggsave('Figs/pos_base.pdf', units="in",width=5,height=5)

ggplot(heading, aes(x = t, y =heading))+geom_line()

test = data.table(sens = sens$X1-sens$X2, heading = heading$heading, t = heading$t)
ggplot(test, aes(x = sens, y = sin(heading))) +geom_path(alpha=test$t/1000)

ggplot(sens, aes(x = X1, y = X2))+geom_path(alpha=test$t/1000)

ggplot(hits, aes(x = t, y = cumsum(hits))) + geom_point()

pca = prcomp(spikes)
pc1 = pca$x[,1]
pc2 = pca$x[,2]
pcdata=data.table(cbind(pc1,pc2))
pcdata$time=seq(1,nrow(pcdata))#+4499
ggplot(pcdata, aes(x = pc1, y = pc2,color=time))+geom_line()


fig <- plot_ly(pcdata, x = ~time, y = ~pc1, z = ~pc2, type = "scatter3d", mode='lines',
               line=list(width=10,color=~time))
fig

###
plotdata = t(as.matrix(spikes))
rem=c()
for(i in 1:nrow(plotdata)){
  sd = sd(plotdata[,i])
  if(sd == 0){
    rem=c(rem, i)
  }
}
plotdata = plotdata[,-rem]


res = cor(plotdata)
# colnames(res)=seq(1,nrow(res))#+249
# rownames(res)=seq(1,nrow(res))#+249
# coul <- rich.colors(10)#colorRampPalette(brewer.pal(8,"Spectral"))(8)
# heatmap(res,Colv = NA, Rowv=NA, col=coul, symm=T)

p <- plot_ly(z = res, type = "heatmap", colorscale = "Jet") %>%
  layout(yaxis = list(dtick = 1, tickmode="array"))
p 
#width = 800, height = 700

ggplot(hits, aes(x = t, y = 1, fill = as.factor(hits))) + geom_tile(show.legend = FALSE)+ 
  scale_fill_manual(values=c("white","black"))+
  scale_y_continuous(limits=c(.5, 1.5), expand = c(0, 0))+
  scale_x_continuous(limits=c(0, 1000), expand = c(0, 0),breaks=c(0,200,400,600,800))+
  theme(axis.ticks.x = element_blank(),
        axis.text.x = element_blank(),
        axis.title.x = element_blank(),
        axis.ticks.y = element_blank(),
        axis.text.y = element_blank(),
        axis.title.y = element_blank())
  ggsave('Figs/hits_base.pdf', units="in",width=5,height=.5)

#########
  ##NOISE
  
pos = as.data.table(read.csv("Data/pos_noise.csv"))
spikes = as.data.table(read.csv("Data/spikes_noise.csv"))
heading = as.data.table(read.csv("Data/heading_noise.csv"))
heading$heading = zero_to_2pi(heading$heading)
sens= as.data.table(read.csv("Data/sens_noise.csv"))
eff = as.data.table(read.csv("Data/eff_noise.csv"))
hits = as.data.table(read.csv("Data/hits_noise.csv"))

t = seq(1,nrow(pos))
pos$t = t
heading$t = t
sens$t = t
eff$t = t
hits$t = t

ggplot(pos, aes(x = X1, y = X2)) + geom_path(alpha=pos$t/1000, arrow = arrow(length = unit(0.1, "inches")))+
  theme(axis.title.x = element_blank(),
        axis.title.y = element_blank())
ggsave('Figs/pos_noise.pdf', units="in",width=5,height=5)

ggplot(heading, aes(x = t, y =heading))+geom_line()

test = data.table(sens = sens$X1-sens$X2, heading = heading$heading, t = heading$t)
ggplot(test, aes(x = sens, y = sin(heading))) +geom_path(alpha=test$t/1000)

ggplot(sens, aes(x = X1, y = X2))+geom_path(alpha=test$t/1000)

ggplot(hits, aes(x = t, y = cumsum(hits))) + geom_point()

pca = prcomp(spikes)
pc1 = pca$x[,1]
pc2 = pca$x[,2]
pcdata=data.table(cbind(pc1,pc2))
pcdata$time=seq(1,nrow(pcdata))#+4499
ggplot(pcdata, aes(x = pc1, y = pc2,color=time))+geom_line()


fig <- plot_ly(pcdata, x = ~time, y = ~pc1, z = ~pc2, type = "scatter3d", mode='lines',
               line=list(width=10,color=~time))
fig

###
plotdata = t(as.matrix(spikes))
rem=c()
for(i in 1:nrow(plotdata)){
  sd = sd(plotdata[,i])
  if(sd == 0){
    rem=c(rem, i)
  }
}
plotdata = plotdata[,-rem]


res = cor(plotdata)
# colnames(res)=seq(1,nrow(res))#+249
# rownames(res)=seq(1,nrow(res))#+249
# coul <- rich.colors(10)#colorRampPalette(brewer.pal(8,"Spectral"))(8)
# heatmap(res,Colv = NA, Rowv=NA, col=coul, symm=T)

p <- plot_ly(z = res, type = "heatmap", colorscale = "Jet") %>%
  layout(yaxis = list(dtick = 1, tickmode="array"))
p 
#width = 800, height = 700

ggplot(hits, aes(x = t, y = 1, fill = as.factor(hits))) + geom_tile(show.legend = FALSE)+ 
  scale_fill_manual(values=c("white","black"))+
  scale_y_continuous(limits=c(.5, 1.5), expand = c(0, 0))+
  scale_x_continuous(limits=c(0, 2000), expand = c(0, 0),breaks=c(0,500,1000,1500))+
  theme(axis.ticks.x = element_blank(),
        axis.text.x = element_blank(),
        axis.title.x = element_blank(),
        axis.ticks.y = element_blank(),
        axis.text.y = element_blank(),
        axis.title.y = element_blank(),
        panel.background =element_rect(fill = "white"))
ggsave('Figs/hits_noise.pdf', units="in",width=5,height=.5)

#########
##PERTURB

pos = as.data.table(read.csv("Data/pos_perturb.csv"))
spikes = as.data.table(read.csv("Data/spikes_perturb.csv"))
heading = as.data.table(read.csv("Data/heading_perturb.csv"))
heading$heading = zero_to_2pi(heading$heading)
sens= as.data.table(read.csv("Data/sens_perturb.csv"))
eff = as.data.table(read.csv("Data/eff_perturb.csv"))
hits = as.data.table(read.csv("Data/hits_perturb.csv"))

t = seq(1,nrow(pos))
pos$t = t
heading$t = t
sens$t = t
eff$t = t
hits$t = t

ggplot(pos, aes(x = X1, y = X2)) + geom_path(alpha=pos$t/1000, arrow = arrow(length = unit(0.1, "inches")))+
  theme(axis.title.x = element_blank(),
        axis.title.y = element_blank())
ggsave('Figs/pos_perturb.pdf', units="in",width=5,height=5)

ggplot(heading, aes(x = t, y =heading))+geom_line()

test = data.table(sens = sens$X1-sens$X2, heading = heading$heading, t = heading$t)
ggplot(test, aes(x = sens, y = sin(heading))) +geom_path(alpha=test$t/1000)

ggplot(sens, aes(x = X1, y = X2))+geom_path(alpha=test$t/1000)

ggplot(hits, aes(x = t, y = cumsum(hits))) + geom_point()

pca = prcomp(spikes)
PC1 = pca$x[,1]
PC2 = pca$x[,2]
pcdata=data.table(cbind(PC1,PC2))
pcdata$Time=seq(1,nrow(pcdata))#+4499

fig <- plot_ly(pcdata, x = ~Time, y = ~PC1, z = ~PC2, type = "scatter3d", mode='lines',
               line=list(width=10,color=~Time))
fig

###
plotdata = t(as.matrix(spikes))
rem=c()
for(i in 1:nrow(plotdata)){
  sd = sd(plotdata[,i])
  if(sd == 0){
    rem=c(rem, i)
  }
}
plotdata = plotdata[,-rem]


res = cor(plotdata)
# colnames(res)=seq(1,nrow(res))#+249
# rownames(res)=seq(1,nrow(res))#+249
# coul <- rich.colors(10)#colorRampPalette(brewer.pal(8,"Spectral"))(8)
# heatmap(res,Colv = NA, Rowv=NA, col=coul, symm=T)

p <- plot_ly(z = res, type = "heatmap", colorscale = "Jet") %>%
  layout(yaxis = list(dtick = 1, tickmode="array"))
p 
#width = 800, height = 700

ggplot(hits, aes(x = t, y = 1, fill = as.factor(hits))) + geom_tile(show.legend = FALSE)+ 
  scale_fill_manual(values=c("white","black"))+
  scale_y_continuous(limits=c(.5, 1.5), expand = c(0, 0))+
  scale_x_continuous(limits=c(0, 2001), expand = c(0, 0),breaks=c(0,500,1000,1500))+
  theme(axis.ticks.x = element_blank(),
        axis.text.x = element_blank(),
        axis.title.x = element_blank(),
        axis.ticks.y = element_blank(),
        axis.text.y = element_blank(),
        axis.title.y = element_blank(),
        panel.background =element_rect(fill = "white"))
ggsave('Figs/hits_perturb.pdf', units="in",width=5,height=.5)

# t.test.fromSummaryStats <- function(mu,n,s) {
#   -diff(mu) / sqrt( sum( s^2/n ) )
# }
# mu = c()
