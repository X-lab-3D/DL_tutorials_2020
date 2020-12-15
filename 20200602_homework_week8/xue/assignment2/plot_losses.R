# Li Xue
# 14-Jul-2020 10:49
#
# INPUT:
# losses
# 0.71355736
# 0.5958583
# 0.57191604
library(ggplot2)
library(reshape2)
args = commandArgs(trailingOnly=TRUE)
lossFL = args[1]

print(paste ('Plot losses from ', lossFL))

df = read.csv(lossFL, sep = '\t')
df['epoch'] =  1:nrow(df)
df = melt(df, id.vars = c('epoch'))
colnames(df) = c('epoch', 'label', 'losses')
#head(df)
p<-ggplot(df) + aes_string(x = 'epoch', y = 'losses', color = 'label') + geom_line()
ggsave('loss.png')
#ggsave('loss.pdf')
print('loss.png generated')


