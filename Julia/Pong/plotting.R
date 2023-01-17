rm(list = ls(all = TRUE))

library(data.table)
library(tidyverse)
library(ggplot2)
library(rCAT)
library(BAMBI)
library(plotly)
library(gplots)


setwd('/Users/jfalanda/Documents/GitHub/ReservoirModel_followups/Julia/Pong')

pos = as.data.table(read.csv("Data/pos.csv"))
spikes = as.data.table(read.csv("Data/spikes.csv"))
sens= as.data.table(read.csv("Data/sens.csv"))
eff = as.data.table(read.csv("Data/eff.csv"))
stim = as.data.table(read.csv("Data/stim.csv"))

t = seq(1,nrow(pos))
pos$t = t
sens$t = t
eff$t = t
stim$t = t

stim$agent = 0
pos$agent=1
plotdata = rbind(pos, stim)
ggplot(plotdata, aes(x = t, y = X2, color = as.factor(agent),group=as.factor(agent))) + 
  geom_line(size=2,alpha=.7)+
  scale_color_manual(values=c('darkgreen','black'),name=NULL,labels=c('Ball','Paddle'))+
  scale_x_continuous(limits = c(0,1000),name="Time")+
  scale_y_continuous(name = "Y-position")

ggsave('Figs/pos.pdf', units="in",width=8,height=4)

plotdata2 = plotdata[t<=1000,]
plotdata2$opac = plotdata2$t/max(plotdata2$t)
fig <- plot_ly(plotdata2, x = ~t, y = ~X1, z = ~X2, color = ~as.factor(agent), type = "scatter3d", opacity = ~opac)
fig

# stim$hit = 0
# dir = -1
# for(i in 2:nrow(stim)){
#   if((stim$X1[i] > stim$X1[i-1]) && dir == -1){
#     stim$hit[i-1] = 1
#     stim$dir[i] = dir 
#   }
# }

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
ggplot(pcdata, aes(x = pc1, y = pc2))+geom_line(alpha=pcdata$time/5000)


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


