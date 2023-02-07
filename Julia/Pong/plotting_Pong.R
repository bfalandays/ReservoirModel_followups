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
hits = as.data.table(read.csv("Data/hits.csv"))

t = seq(1,nrow(pos))
pos$t = t
sens$t = t
eff$t = t
stim$t = t
hits$t = t

stim$agent = 0
pos$agent=1
plotdata = rbind(pos, stim)
plotdata2 = subset(hits, ts_hits != 0 & t <=2000)
phits = subset(plotdata2, ts_hits == 1)
pmiss = subset(plotdata2, ts_hits == -1)

ggplot(plotdata, aes(x = t, y = X2, linetype = as.factor(agent),group=as.factor(agent))) + 
  geom_line(size=.5,alpha=1)+
  #scale_color_manual(values=c('darkgreen','black', 'blue'),name=NULL,labels=c('Ball','Paddle'))+
  scale_x_continuous(limits = c(0,2000),name="Time")+
  scale_y_continuous(name = "Y-position")+
  scale_linetype_discrete(name = NULL,labels= c("Stimulus","Agent"))+
  geom_vline(data = phits, aes(xintercept =t),color = 'darkgreen', size=4, alpha=.5)+
  geom_vline(data = pmiss, aes(xintercept =t),color = 'red', size=4, alpha=.5)


ggsave('Figs/pos.pdf', units="in",width=8,height=2)

# plotdata2 = plotdata[t<=2000,]
# plotdata2$opac = plotdata2$t/max(plotdata2$t)
# fig <- plot_ly(plotdata2, x = ~t, y = ~X1, z = ~X2, color = ~as.factor(agent), type = "scatter3d", opacity = ~opac)
# fig

# stim$hit = 0
# dir = -1
# for(i in 2:nrow(stim)){
#   if((stim$X1[i] > stim$X1[i-1]) && dir == -1){
#     stim$hit[i-1] = 1
#     stim$dir[i] = dir 
#   }
# }

pca = prcomp(spikes[1:2000,])
PC1 = pca$x[,1]
PC2 = pca$x[,2]
PC3 = pca$x[,3]
PC4 = pca$x[,4]
pcdata=data.table(cbind(PC1,PC2, PC3,PC4))
pcdata$ypos_a = pos$X2[1:2000]
stim = as.data.table(read.csv("Data/stim.csv"))
pcdata$ypos_b = stim$X2[1:2000]
pcdata$Time=seq(1,nrow(pcdata))#+4499
ggplot(pcdata, aes(x = PC1, y = PC2))+geom_line(color=pcdata$time)+
  scale_color_continuous()


# fig <- plot_ly(pcdata, x = ~ypos_a, y = ~ypos_b, z = ~pc2, type = "scatter3d", mode='lines',
#                line=list(width=10,color=~time))

pcdata$color = 1-pcdata$Time/2000
fig <- plot_ly(pcdata, x = ~Time, y = ~PC2, z = ~PC3, type = "scatter3d", mode='lines',
               line=list(width=10,color=~Time), opacity = .7)

fig



###
plotdata = t(as.matrix(spikes[1:2000]))

# replace all non-finite values with 0
#res[!is.finite(res)] <- 0


rem=c()
for(i in 1:nrow(plotdata)){
  sd = sd(plotdata[,i])
  if(sd == 0){
    rem=c(rem, i)
  }
}
plotdata = plotdata[,-rem]

res = cor(plotdata)

# res[!rowSums(!is.finite(res)),]

# colnames(res)=seq(1,nrow(res))#+249
# rownames(res)=seq(1,nrow(res))#+249
# coul <- rich.colors(10)#colorRampPalette(brewer.pal(8,"Spectral"))(8)
# heatmap(res,Colv = NA, Rowv=NA, col=coul, symm=T)

p <- plot_ly(z = res, type = "heatmap", colorscale = "Jet") %>%
  layout(yaxis = list(dtick = 1, tickmode="array"))
p 

library(ComplexHeatmap)
library(colorRamp2)
library(RColorBrewer)
p=Heatmap(
  res, # putting the top on top
  col = colorRamp2(seq(quantile(res, 0.01,na.rm = TRUE), quantile(res, 0.99,na.rm = TRUE), len = 11), rev(brewer.pal(11, "Spectral"))),
  cluster_rows = FALSE, cluster_columns = FALSE,
  name="Autocorrelation",
  #show_heatmap_legend = FALSE,
  #column_title = "No rasterisation",
  use_raster = TRUE,
  raster_resize_mat = TRUE,
  raster_quality = 2)

p

#width = 800, height = 700
plotdata = subset(hits, ts_hits != 0)

ggplot(plotdata) + 
  geom_vline(aes(xintercept = plotdata$t, color = as.factor(plotdata$ts_hits)), show.legend = FALSE,size=2)+ 
  scale_color_manual(values=c("red","darkgreen"))+
  scale_y_continuous(limits=c(0, 1), expand = c(0, 0))+
  scale_x_continuous(limits=c(0, 2000), expand = c(0, 0))+
  theme(
        axis.ticks.x = element_blank(),
        axis.text.x = element_blank(),
        axis.title.x = element_blank(),
        axis.ticks.y = element_blank(),
        axis.text.y = element_blank(),
        axis.title.y = element_blank())
  ggsave('Figs/hits_base_gray.pdf', units="in",width=5,height=.5)


meanspikes = data.table(pSpike=rowMeans(spikes), Time = seq(1,nrow(spikes)))
ggplot(meanspikes, aes(x = Time, y = pSpike)) + geom_line()+
  scale_y_continuous(name='Prop. Spiked')+
  scale_x_continuous(limits=c(0,2000))
ggsave('Figs/spikes.pdf', units="in",width=7,height=2)


### 
meanHits = as.data.table(read.csv("Data/meanHits.csv"))
meanHits$condition = 1
meanHits_allo = as.data.table(read.csv("Data/meanHits_allo.csv"))
meanHits_allo$condition = 4
meanHits_nL = as.data.table(read.csv("Data/meanHits_nL.csv"))
meanHits_nL$condition = 3
meanHits_f50 = as.data.table(read.csv("Data/meanHits_f100.csv"))[1:50]
meanHits_f50$condition = 0
meanHits_l50 = as.data.table(read.csv("Data/meanHits_l100.csv"))[1:50]
meanHits_l50$condition = 2



meanHits = rbind(meanHits, meanHits_allo, meanHits_nL, meanHits_f100, meanHits_l100)

ggplot(meanHits, aes(x = as.factor(condition), y = meanHits, color = as.factor(condition)))+
  geom_jitter(alpha=.6, width = .2)+
  stat_summary(fun.data = mean_cl_boot, geom = "errorbar", width = 0.2, color = 'black',size=1)+
  scale_y_continuous(limits=c(0,1), breaks=seq(0,1,.2), name = "Proportion Hits")+
  scale_x_discrete(name=NULL, labels=c('First 50','Baseline', 'Last 50', 'No Learning', 'Allocentric'))+
  #scale_color_brewer(palette = "Set1", name=NULL, labels = NULL)+
  scale_color_manual(values = c("grey60", "grey40", "grey20", "blue2","red3"))+
  guides(color=FALSE)+
  theme(axis.text.x = element_text(size=10),
        axis.text.y = element_text(size=12),
        axis.title.y = element_text(size=15))
ggsave("Figs/propHits.pdf", units = "in", width=5, height = 5)

mean(meanHits_f50$meanHits)
sd(meanHits_f50$meanHits)

mean(meanHits_l50$meanHits)
sd(meanHits_l50$meanHits)

t.test(meanHits_f50$meanHits, meanHits_l50$meanHits)
