rm(list = ls(all = TRUE))

library(data.table)
library(tidyverse)
library(ggplot2)
library(rCAT)
library(BAMBI)
library(plotly)
library(gplots)
library(tseries)


setwd('/Users/jfalanda/Documents/GitHub/ReservoirModel_followups/Julia/ObjectTracking')

spikes = as.data.table(read.csv("Data/spikes.csv"))
heading = as.data.table(read.csv("Data/heading.csv"))
colnames(heading)='heading'
sens= as.data.table(read.csv("Data/sens.csv"))
eff = as.data.table(read.csv("Data/eff.csv"))
stim = as.data.table(read.csv("Data/stim.csv"))
acts = as.data.table(read.csv("Data/acts.csv"))



t = seq(1,nrow(spikes))
heading$t = t
sens$t = t
eff$t = t
stim$t = t
acts$t = t

#####

stim$agent = 0
heading$agent=1
plotdata = rbind(heading, stim)

ggplot(plotdata, aes(x = t, y = rad2deg(heading), color = as.factor(agent),group=as.factor(agent))) + 
  geom_point(size=1,alpha=.8)+
  scale_x_continuous(limits = c(0,7200),name=NULL)+
  scale_y_continuous(name = "Heading (deg.)")+
  scale_color_manual(name = NULL, labels = c("Stimulus", "Agent"), values = c('black','red'))
  #scale_linetype_discrete(name = NULL,labels= c("Stimulus","Agent"))
ggsave('Figs/heading_ot3.pdf', units="in",width=5.5,height=1.5)


####
ggplot(heading, aes(x = t, y =heading))+geom_line()

pca = prcomp(spikes,center = FALSE, scale = FALSE)
PC1 = pca$x[,1]
PC2 = pca$x[,2]
PC3 = pca$x[,3]
pcdata=data.table(cbind(PC1,PC2,PC3))
pcdata$Time=seq(1,nrow(pcdata))#+1000
#ggplot(pcdata, aes(x = PC1, y = PC2,color=Time))+geom_line()


fig <- plot_ly(pcdata, x = ~Time, y = ~PC1, z = ~PC2, type = "scatter3d", mode='lines',
               line=list(width=10,color=~Time))
fig

###
plotdata = t(as.matrix(spikes[0:7200]))
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

# p <- plot_ly(z = res, type = "heatmap", colorscale = "Jet")# %>%
#   layout(yaxis = list(tickmode="array",tickvals = seq(0,1000,250),ticktext=as.character(seq(1000,2000,250))),
#          xaxis = list(tickmode="array",tickvals = seq(0,1000,250),ticktext=as.character(seq(1000,2000,250)))
#          )
# p 

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

# ###
# library(nonlinearTseries)
# 
# rqa.analysis=rqa(time.series = PC1, embedding.dim=2, time.lag=1,
#                  radius=1.2,lmin=2,do.plot=FALSE,distanceToBorder=2)
# plot(rqa.analysis)

library(crqa)
meanspikes = data.table(pSpike=rowMeans(spikes[1:2000]), Time = seq(1,2000))

res=crqa(meanspikes$pSpike, meanspikes$pSpike, method = 'rqa',delay = 1, embed = 10, radius = .05,normalize=1 )
RP = res$RP
## define plotting arguments
par = list(unit = 1, labelx = "Time", labely = "Time",
           cols = "black", pcex = .1, pch = 1,
           labax = NULL,
           labay = NULL,
           las = 1)
plotRP(RP, par)
# 
# 
# plotdata = as.matrix(spikes[1:2000])
# 
# # par = list(method = "mdcrqa", metric = "euclidean", maxlag = 1,
# #            radiusspan = 100, radiussample = 40, normalize = 0,
# #            rescale = 0, mindiagline = 2, minvertline = 2, tw = 0,
# #            whiteline = FALSE, recpt = FALSE, side = "both",
# #            datatype = "categorical", fnnpercent = NA,
# #            typeami = NA, nbins = 50, criterion = "firstBelow",
# #            threshold = 1, maxEmb = 20, numSamples = 500,
# #            Rtol = 10, Atol = 2)
# # results = optimizeParam(plotdata,plotdata, par, min.rec = 1, max.rec = 200)
# 


res=crqa(plotdata,plotdata, method = 'mdcrqa',radius = 8.5, delay = 1, embed = 1 )
RP = res$RP
## define plotting arguments
par = list(unit = 1, labelx = "Time", labely = "Time",
           cols = "black", pcex = .2, pch = 1,
           labax = seq(0, nrow(RP), 1),
           labay = seq(0, nrow(RP), 1),
           las = 1)
plotRP(RP, par)


####

meanspikes = data.table(pSpike=rowMeans(spikes[1:7200]), Time = seq(1,7200))
ggplot(meanspikes, aes(x = Time, y = pSpike)) + geom_line()+
  scale_y_continuous(name='Prop. Spiked')+
  scale_x_continuous(limits=c(0,7200))
ggsave('Figs/spikes.pdf', units="in",width=4.5,height=1.75)



########

heading = as.data.table(read.csv("Data/heading.csv"))
colnames(heading)='heading'

eff = as.data.table(read.csv("Data/eff.csv"))
stim = as.data.table(read.csv("Data/stim.csv"))



t = seq(1,nrow(spikes))
heading$t = t
eff$t = t
stim$t = t

heading$stim_heading = stim$heading
heading$diff = heading$heading - heading$stim_heading
heading$eDiff = eff$X1 - eff$X2
heading$pc1 = pcdata$PC1
heading$pc2 = pcdata$PC2
heading$pc3 = pcdata$PC3
heading$meanSpike = rowMeans(spikes)

fig <- plot_ly(heading, x = ~pc1, y = ~pc2, z = ~t, type = "scatter3d", mode='lines',
               line=list(width=10,color=~t))
fig

###

##correlate every node in the reservoir with the position of the stim, the heading, and the difference. Maybe do this for some different windows in case it is not stable.
cors1=c()
rows=c(1:1000)
for(i in 1:200){
  print(i)
  cur_spikes = tibble(spikes)[rows,i]
  cor_agent = cor(heading$heading[rows], cur_spikes)
  cor_stim = cor(stim$heading[rows], cur_spikes)
  cor_eff = cor(eff$X1[rows]-eff$X2[rows], cur_spikes)
  cor_diff = cor(stim$heading[rows]-heading$heading[rows], cur_spikes)

  cors1=rbind(cors1,c(cor_stim,cor_agent, cor_eff,cor_diff, i))
}
cors1=data.table(cors1)
colnames(cors1) = c("cor_stim", "cor_agent","cor_eff","cor_diff","node")

order1 = order(cors1$cor_eff, decreasing = TRUE)

cors2=c()
rows=c(250:1250)
for(i in 1:200){
  print(i)
  cur_spikes = tibble(spikes)[rows,i]
  cor_agent = cor(heading$heading[rows], cur_spikes)
  cor_stim = cor(stim$heading[rows], cur_spikes)
  cor_eff = cor(eff$X1[rows]-eff$X2[rows], cur_spikes)
  cor_diff = cor(stim$heading[rows]-heading$heading[rows], cur_spikes)
  
  cors2=rbind(cors2,c(cor_stim,cor_agent, cor_eff,cor_diff, i))
}
cors2=data.table(cors2)
colnames(cors2) = c("cor_stim", "cor_agent","cor_eff","cor_diff","node")

order2 = order(cors2$cor_eff, decreasing = TRUE)

cors3=c()
rows=c(500:1500)
for(i in 1:200){
  print(i)
  cur_spikes = tibble(spikes)[rows,i]
  cor_agent = cor(heading$heading[rows], cur_spikes)
  cor_stim = cor(stim$heading[rows], cur_spikes)
  cor_eff = cor(eff$X1[rows]-eff$X2[rows], cur_spikes)
  cor_diff = cor(stim$heading[rows]-heading$heading[rows], cur_spikes)
  
  cors3=rbind(cors3,c(cor_stim,cor_agent, cor_eff,cor_diff, i))
}
cors3=data.table(cors3)
colnames(cors3) = c("cor_stim", "cor_agent","cor_eff","cor_diff","node")

order3 = order(cors3$cor_eff, decreasing = TRUE)

cors4=c()
rows=c(750:1750)
for(i in 1:200){
  print(i)
  cur_spikes = tibble(spikes)[rows,i]
  cor_agent = cor(heading$heading[rows], cur_spikes)
  cor_stim = cor(stim$heading[rows], cur_spikes)
  cor_eff = cor(eff$X1[rows]-eff$X2[rows], cur_spikes)
  cor_diff = cor(stim$heading[rows]-heading$heading[rows], cur_spikes)
  
  cors4=rbind(cors4,c(cor_stim,cor_agent, cor_eff,cor_diff, i))
}
cors4=data.table(cors4)
colnames(cors4) = c("cor_stim", "cor_agent","cor_eff","cor_diff","node")

order4 = order(cors4$cor_eff, decreasing = TRUE)

####
#plotting correlation with movement direction (diff of effectors) at 1000:1500 and 1500:2000, ordered by correlation in each windows

pCors1.1 = cors1
pCors1.1$node=factor(pCors1.1$node, levels = c(order1))
p1.1=ggplot(pCors1.1, aes(x = as.factor(node), y = cor_eff))+
  geom_bar(stat = "identity")+
  scale_y_continuous(limits=c(-1,1),name='Node-Mvt Corr.')+
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.title.x = element_blank(),axis.title.y=element_blank())

pCors1.2 = cors1
pCors1.2$node=factor(pCors1.2$node, levels = c(order2))
p1.2=ggplot(pCors1.2, aes(x = as.factor(node), y = cor_eff))+
  geom_bar(stat = "identity")+
  scale_y_continuous(limits=c(-1,1),name='Node-Mvt Corr.')+
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.title.x = element_blank(),axis.title.y=element_blank())

pCors1.3 = cors1
pCors1.3$node=factor(pCors1.3$node, levels = c(order3))
p1.3=ggplot(pCors1.3, aes(x = as.factor(node), y = cor_eff))+
  geom_bar(stat = "identity")+
  scale_y_continuous(limits=c(-1,1),name='Node-Mvt Corr.')+
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.title.x = element_blank(),axis.title.y=element_blank())

pCors1.4 = cors1
pCors1.4$node=factor(pCors1.4$node, levels = c(order4))
p1.4=ggplot(pCors1.4, aes(x = as.factor(node), y = cor_eff))+
  geom_bar(stat = "identity")+
  scale_y_continuous(limits=c(-1,1),name='Node-Mvt Corr.')+
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.title.x = element_blank(),axis.title.y=element_blank())


pCors2.1 = cors2
pCors2.1$node=factor(pCors2.1$node, levels = c(order1))
p2.1=ggplot(pCors2.1, aes(x = as.factor(node), y = cor_eff))+
  geom_bar(stat = "identity")+
  scale_y_continuous(limits=c(-1,1),name='Node-Mvt Corr.')+
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.title.x = element_blank(),axis.title.y=element_blank(), axis.text.y = element_blank())

pCors2.2 = cors2
pCors2.2$node=factor(pCors2.2$node, levels = c(order2))
p2.2=ggplot(pCors2.2, aes(x = as.factor(node), y = cor_eff))+
  geom_bar(stat = "identity")+
  scale_y_continuous(limits=c(-1,1),name='Node-Mvt Corr.')+
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.title.x = element_blank(),axis.title.y=element_blank(), axis.text.y = element_blank())

pCors2.3 = cors2
pCors2.3$node=factor(pCors2.3$node, levels = c(order3))
p2.3=ggplot(pCors2.3, aes(x = as.factor(node), y = cor_eff))+
  geom_bar(stat = "identity")+
  scale_y_continuous(limits=c(-1,1),name='Node-Mvt Corr.')+
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.title.x = element_blank(),axis.title.y=element_blank(), axis.text.y = element_blank())

pCors2.4 = cors2
pCors2.4$node=factor(pCors2.4$node, levels = c(order4))
p2.4=ggplot(pCors2.4, aes(x = as.factor(node), y = cor_eff))+
  geom_bar(stat = "identity")+
  scale_y_continuous(limits=c(-1,1),name='Node-Mvt Corr.')+
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.title.x = element_blank(),axis.title.y=element_blank(), axis.text.y = element_blank())

pCors3.1 = cors3
pCors3.1$node=factor(pCors3.1$node, levels = c(order1))
p3.1=ggplot(pCors3.1, aes(x = as.factor(node), y = cor_eff))+
  geom_bar(stat = "identity")+
  scale_y_continuous(limits=c(-1,1),name='Node-Mvt Corr.')+
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.title.x = element_blank(),axis.title.y=element_blank(), axis.text.y = element_blank())

pCors3.2 = cors3
pCors3.2$node=factor(pCors3.2$node, levels = c(order2))
p3.2=ggplot(pCors3.2, aes(x = as.factor(node), y = cor_eff))+
  geom_bar(stat = "identity")+
  scale_y_continuous(limits=c(-1,1),name='Node-Mvt Corr.')+
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.title.x = element_blank(),axis.title.y=element_blank(), axis.text.y = element_blank())

pCors3.3 = cors3
pCors3.3$node=factor(pCors3.3$node, levels = c(order3))
p3.3=ggplot(pCors3.3, aes(x = as.factor(node), y = cor_eff))+
  geom_bar(stat = "identity")+
  scale_y_continuous(limits=c(-1,1),name='Node-Mvt Corr.')+
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.title.x = element_blank(),axis.title.y=element_blank(), axis.text.y = element_blank())

pCors3.4 = cors3
pCors3.4$node=factor(pCors3.4$node, levels = c(order4))
p3.4=ggplot(pCors3.4, aes(x = as.factor(node), y = cor_eff))+
  geom_bar(stat = "identity")+
  scale_y_continuous(limits=c(-1,1),name='Node-Mvt Corr.')+
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.title.x = element_blank(),axis.title.y=element_blank(), axis.text.y = element_blank())

pCors4.1 = cors4
pCors4.1$node=factor(pCors4.1$node, levels = c(order1))
p4.1=ggplot(pCors4.1, aes(x = as.factor(node), y = cor_eff))+
  geom_bar(stat = "identity")+
  scale_y_continuous(limits=c(-1,1),name='Node-Mvt Corr.')+
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.title.x = element_blank(),axis.title.y=element_blank(), axis.text.y = element_blank())

pCors4.2 = cors4
pCors4.2$node=factor(pCors4.2$node, levels = c(order2))
p4.2=ggplot(pCors4.2, aes(x = as.factor(node), y = cor_eff))+
  geom_bar(stat = "identity")+
  scale_y_continuous(limits=c(-1,1),name='Node-Mvt Corr.')+
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.title.x = element_blank(),axis.title.y=element_blank(), axis.text.y = element_blank())

pCors4.3 = cors4
pCors4.3$node=factor(pCors4.3$node, levels = c(order3))
p4.3=ggplot(pCors4.3, aes(x = as.factor(node), y = cor_eff))+
  geom_bar(stat = "identity")+
  scale_y_continuous(limits=c(-1,1),name='Node-Mvt Corr.')+
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.title.x = element_blank(),axis.title.y=element_blank(), axis.text.y = element_blank())

pCors4.4 = cors4
pCors4.4$node=factor(pCors4.4$node, levels = c(order4))
p4.4=ggplot(pCors4.4, aes(x = as.factor(node), y = cor_eff))+
  geom_bar(stat = "identity")+
  scale_y_continuous(limits=c(-1,1),name='Node-Mvt Corr.')+
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.title.x = element_blank(),axis.title.y=element_blank(), axis.text.y = element_blank())


library(patchwork)
p1.1+p2.1+p3.1+p4.1+p1.2+p2.2+p3.2+p4.2+p1.3+p2.3+p3.3+p4.3+p1.4+p2.4+p3.4+p4.4

ggsave('Figs/repDrift_ot.pdf', units="in",width=8,height=7.5)



