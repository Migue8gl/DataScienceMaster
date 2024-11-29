library(tidyverse)
library(readxl)
data <- read_excel("car_example.xls", sheet = 1)

ggplot(data, aes(x = top_speed_mph)) +
  geom_histogram(binwidth = 10,
                 fill = "blue",
                 color = "black") +
  labs(title = "Histogram of Maximum Velocity", x = "Maximum Velocity", y = "Frequency") +
  theme_minimal() +
  facet_wrap( ~ decade)

cars <- data %>% filter(year >= 1990, top_speed_mph == 155) %>% 
  group_by(make_nm) %>% 
  summarise(count_controlled = n()) %>% 
  arrange(desc(count_controlled))
cars

another_cars <- data %>%
  filter(horsepower_bhp > 750, year >= 2010) %>%
  group_by(make_nm) %>%
  summarise(count = n()) %>% 
  arrange(desc(count))

another_cars
