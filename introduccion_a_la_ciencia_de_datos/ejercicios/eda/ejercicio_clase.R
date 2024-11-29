library(dplyr)
library(tidyverse)
data(starwars)
View(starwars)

--------------------------------------------------------------------------------

starwars %>%
  filter(!is.na(species)) %>%
  count(species, sort = TRUE)

starwars %>%
  filter(!is.na(skin_color)) %>%
  count(skin_color, sort = TRUE)

starwars %>%
  mutate(skin_color = fct_lump(skin_color, n = 5)) %>%
  count(skin_color, sort = TRUE)

avg_mass_eye_color <- starwars %>%
  filter(!is.na(mass)) %>%
  mutate(eye_color = fct_lump(eye_color, n =
                                6)) %>%
  group_by(eye_color) %>%
  summarise(mean_mass = mean(mass, na.rm =
                               TRUE))

avg_mass_eye_color

--------------------------------------------------------------------------------

gender <- c("f", "m ", "male ","male", "female", "FEMALE",
            "Male", "f", "m")
gender <- as_factor(gender)

gender

gender <- fct_collapse(
  gender,
  Female = c("f", "female", "FEMALE"),
  Male = c("m ", "m", "male ", "male", "Male")
)
fct_count(gender)

--------------------------------------------------------------------------------

gender <- c("f", "m ", "male ","male", "female", "FEMALE",
            "Male", "f", "m")
gender <- as_factor(gender)
gender <- fct_anon(gender)
fct_count(gender)







