library(tidyverse)

M <- read_csv("2020-11-08T13:58:31.337993.csv")

M$idx <- 1:nrow(M)

M %>%
  select(idx, Thermistor_Temp_NIST, Analog) %>%
  pivot_longer(cols = -idx) %>%
  ggplot(aes(idx, value, color = name)) + geom_line()

