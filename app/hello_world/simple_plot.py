import numpy as np
import pandas as pd
from plotnine import *

df = pd.read_csv("datasets/houses.csv")
p = ggplot(df) + geom_point(aes(x='sq_footage',y='price'))
ggsave(plot=p, filename='/app/plots/plot.png', dpi=100)
