rm(list = ls(all = TRUE))
setwd('/Users/jbenfalandays/Documents/GitHub/ReservoirModel_followups/')

library(ggplot2)
library(data.table)
library(tidyverse)
library(RColorBrewer)
library(gplots)
library(plotly)

df = data.table(read_csv('spikes_overTime_Rotating_4.csv')) %>% select(-X1)
df_p = data.table(read_csv('positionData_Rotating_4.csv')) %>% select(-c("0"))
colnames(df_p)=c('stim','agent','time')
df_p = df_p %>% pivot_longer(cols = c(agent,stim),values_to = 'Angle',names_to = 'var') %>% filter(time !=0)

#deg2rad <- function(deg) {(deg * pi) / (180)}

#df_p$agent = deg2rad(df_p$agent)
#df_p$stim = deg2rad(df_p$stim)

ggplot(df_p, aes(x=time, y = Angle, color=var))+
  geom_point(size=1)+
  scale_x_continuous('Time',breaks=seq(0,7200,by=500), expand=c(0,0))+
  scale_y_continuous('Angle (deg.)', breaks=c(0,45,90,135,180,225,270,315,360))+
  scale_color_manual(name = "",
                     values = c( "stim" = "chartreuse4", "agent" = "black"),
                     labels = c("Stimulus", "Agent"))+
  ggsave('AngleOverTime.pdf', height=2, width=8, units="in")

#####
plotdata=df[5500:6500,]
pca=prcomp(plotdata)
summary(pca)

pc1 = pca$x[,1]
pc2 = pca$x[,2]
pcdata=data.table(cbind(pc1,pc2))
pcdata$time=seq(1,nrow(pcdata))+4499
ggplot(pcdata, aes(x = pc1, y = pc2,color=time))+geom_line()

fig <- plot_ly(pcdata, x = ~time, y = ~pc1, z = ~pc2, type = "scatter3d", mode='lines',
               line=list(width=10,color=~time))
fig

###
res = cor(t(plotdata))
colnames(res)=seq(1,nrow(pcdata))+2199
rownames(res)=seq(1,nrow(pcdata))+2199
coul <- rich.colors(10)#colorRampPalette(brewer.pal(8,"Spectral"))(8)
heatmap(res,Colv = NA, Rowv=NA, col=coul, symm=T)

