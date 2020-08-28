library(tidymodels)
library(sf)
library(patchwork)
library(RColorBrewer)
library(grid)
library(ragg)

## -----------------------------------------------------------------------------

data(ames)

# Source:
# https://mapcruzin.com/free-united-states-shapefiles/free-iowa-arcgis-maps-shapefiles.htm
# Look for "Iowa Highway Shapefile". This also requires "iowa_highway.shx"
ia_roads <- st_read(dsn = "iowa_highway.shp") 

## -----------------------------------------------------------------------------

col_vals <- brewer.pal(8, "Dark2")

ames_cols <- c(
  North_Ames = col_vals[3],
  College_Creek = col_vals[4],
  Old_Town = col_vals[8],
  Edwards = col_vals[5],
  Somerset = col_vals[8],
  Northridge_Heights = col_vals[4],
  Gilbert = col_vals[2],
  Sawyer = col_vals[7],
  Northwest_Ames = col_vals[6],
  Sawyer_West = col_vals[8],
  Mitchell = col_vals[3],
  Brookside = col_vals[1],
  Crawford = col_vals[4],
  Iowa_DOT_and_Rail_Road = col_vals[2],
  Timberland = col_vals[2],
  Northridge = col_vals[1],
  Stone_Brook = col_vals[5],
  South_and_West_of_Iowa_State_University = col_vals[3],
  Clear_Creek = col_vals[1],
  Meadow_Village = col_vals[1],
  Briardale = col_vals[7],
  Bloomington_Heights = col_vals[1],
  Veenker = col_vals[2],
  Northpark_Villa = col_vals[2],
  Blueste = col_vals[5],
  Greens = col_vals[6],
  Green_Hills = col_vals[8],
  Landmark = col_vals[3],
  Hayden_Lake = "red"
)

ames_pch <- c(
  North_Ames = 16,
  College_Creek = 16,
  Old_Town = 16,
  Edwards = 16,
  Somerset = 15,
  Northridge_Heights = 15,
  Gilbert = 16,
  Sawyer = 16,
  Northwest_Ames = 16,
  Sawyer_West = 17,
  Mitchell = 15,
  Brookside = 16,
  Crawford = 17,
  Iowa_DOT_and_Rail_Road = 17,
  Timberland = 15,
  Northridge = 15,
  Stone_Brook = 15,
  South_and_West_of_Iowa_State_University = 17,
  Clear_Creek = 17,
  Meadow_Village = 18,
  Briardale = 15,
  Bloomington_Heights = 20,
  Veenker = 18,
  Northpark_Villa = 20,
  Blueste = 17,
  Greens = 15,
  Green_Hills = 18,
  Landmark = 18,
  Hayden_Lake = 16
)


## -----------------------------------------------------------------------------

ames_x <- extendrange(ames$Longitude)
ames_y <- extendrange(ames$Latitude)
ames_ratio <- diff(ames_x)/diff(ames_y)

all_ames <- 
  ggplot() +
  xlim(ames_x) +
  ylim(ames_y) + 
  theme_void() + 
  theme(legend.position = "bottom", legend.title = element_blank()) +
  geom_sf(data = ia_roads, aes(geometry = geometry), alpha = .1) +
  geom_point(
    data = ames,
    aes(
      x = Longitude,
      y = Latitude,
      col = Neighborhood,
      shape = Neighborhood
    ),
    size = 1, 
    alpha = .5
  ) + 
  scale_color_manual(values = ames_cols) + 
  scale_shape_manual(values = ames_pch)

agg_png("ames.png", width = 820 * 3, height = 820 * 3, res = 300, scaling = 1)
print(all_ames)
dev.off()

## -----------------------------------------------------------------------------

mitchell_x <- extendrange(ames$Longitude[ames$Neighborhood == "Mitchell"], f = .1)
mitchell_y <- extendrange(ames$Latitude[ames$Neighborhood == "Mitchell"], f = .1)

mitchell_box <- 
  ggplot() +
  xlim(extendrange(ames$Longitude)) +
  ylim(extendrange(ames$Latitude)) + 
  theme_void() + 
  theme(legend.position = "none", plot.title = element_text(hjust = 0.5),
        plot.background = element_rect(linetype = 'solid', colour = 'black', size = 1)) +
  geom_sf(data = ia_roads, aes(geometry = geometry), alpha = .1) +
  geom_point(
    data = ames %>% filter(Neighborhood %in% c("Meadow_Village", "Mitchell")),
    aes(
      x = Longitude,
      y = Latitude,
      col = Neighborhood,
      shape = Neighborhood
    ),
    size = .3, 
    alpha = .5
  ) + 
  scale_color_manual(values = ames_cols) + 
  scale_shape_manual(values = ames_pch) +
  geom_rect(
    aes(
      xmin = mitchell_x[1],
      xmax = mitchell_x[2],
      ymin = mitchell_y[1],
      ymax = mitchell_y[2]
    ),
    fill = NA,
    color = "black"
  )


mitchell_with_space <- mitchell_x
mitchell_with_space[2] <- mitchell_x[2] +  diff(mitchell_x) * .5
mitchell_ratio <- diff(mitchell_with_space)/diff(mitchell_y)

mitchell <- 
  ggplot() +
  xlim(mitchell_with_space) +
  ylim(mitchell_y) +
  theme_void() + 
  theme(
    legend.position = c(.82, .55 ),
    legend.background = element_rect(
      fill = "white",
      colour = NA,
      size = 3
    ),
    legend.text  = element_text(size = rel(1.5)),
    legend.title = element_text(size = rel(1.5))
  ) +
  geom_sf(data = ia_roads, aes(geometry = geometry), alpha = .3) +
  geom_point(
    data = ames %>% filter(Neighborhood %in% c("Meadow_Village", "Mitchell")),
    aes(
      x = Longitude,
      y = Latitude,
      col = Neighborhood,
      shape = Neighborhood
    ),
    size = 4, 
    alpha = .5
  ) + 
  scale_color_manual(values = ames_cols) + 
  scale_shape_manual(values = ames_pch)


# make plot and guide side-by-side
# mitchell_box +  plot_spacer() + mitchell +  plot_layout(widths = c(2, 0.1, 3))

# guide inset in plot
agg_png("mitchell.png", width = 480 * mitchell_ratio * 2, height = 480 * 2, res = 200)
print(mitchell)
print(mitchell_box, vp = viewport(0.8, 0.27, width = 0.3 * ames_ratio, height = 0.3))
dev.off()


## -----------------------------------------------------------------------------

timberland_x <- extendrange(ames$Longitude[ames$Neighborhood == "Timberland"], f = .2)
timberland_y <- extendrange(ames$Latitude[ames$Neighborhood == "Timberland"], f = .2)

timberland_box <- 
  ggplot() +
  xlim(extendrange(ames$Longitude)) +
  ylim(extendrange(ames$Latitude)) + 
  theme_void() + 
  theme(legend.position = "none", plot.title = element_text(hjust = 0.5),
        plot.background = element_rect(linetype = 'solid', colour = 'black', size = 1)) +
  geom_sf(data = ia_roads, aes(geometry = geometry), alpha = .1) +
  geom_point(
    data = ames,
    aes(
      x = Longitude,
      y = Latitude,
      col = Neighborhood,
      shape = Neighborhood
    ),
    size = .2, 
    alpha = .5
  ) + 
  scale_color_manual(values = ames_cols) + 
  scale_shape_manual(values = ames_pch) +
  geom_rect(
    aes(
      xmin = timberland_x[1],
      xmax = timberland_x[2],
      ymin = timberland_y[1],
      ymax = timberland_y[2]
    ),
    fill = NA,
    color = "black"
  )


timberland_with_space <- timberland_x
timberland_with_space[2] <- timberland_x[2] +  diff(timberland_x) * .15
timberland_ratio <- diff(timberland_with_space)/diff(timberland_y)

timberland <- 
  ggplot() +
  xlim(timberland_with_space) +
  ylim(timberland_y) +
  theme_void() + 
  theme(legend.position = "none") +
  geom_sf(data = ia_roads, aes(geometry = geometry), alpha = .3) +
  geom_point(
    data = ames,
    aes(
      x = Longitude,
      y = Latitude,
      col = Neighborhood,
      shape = Neighborhood
    ),
    size = 5, 
    alpha = .5
  ) + 
  scale_color_manual(values = ames_cols) + 
  scale_shape_manual(values = ames_pch)

# guide inset in plot
agg_png("timberland.png", width = 480 * timberland_ratio)
print(timberland)
print(timberland_box, vp = viewport(0.85, 0.2, width = 0.3 * ames_ratio, height = 0.3))
dev.off()

## -----------------------------------------------------------------------------

dot_rr_x <- extendrange(ames$Longitude[ames$Neighborhood == "Iowa_DOT_and_Rail_Road"], f = .05)
dot_rr_y <- extendrange(ames$Latitude[ames$Neighborhood == "Iowa_DOT_and_Rail_Road"], f = .1)

dot_rr_box <- 
  ggplot() +
  xlim(extendrange(ames$Longitude)) +
  ylim(extendrange(ames$Latitude)) + 
  theme_void() + 
  theme(legend.position = "none", plot.title = element_text(hjust = 0.5),
        plot.background = element_rect(linetype = 'solid', colour = 'black', size = 1)) +
  geom_sf(data = ia_roads, aes(geometry = geometry), alpha = .1) +
  geom_point(
    data = ames %>% filter(Neighborhood %in% c("Iowa_DOT_and_Rail_Road")),
    aes(
      x = Longitude,
      y = Latitude,
      col = Neighborhood,
      shape = Neighborhood
    ),
    size = .3, 
    alpha = .5
  ) + 
  scale_color_manual(values = ames_cols) + 
  scale_shape_manual(values = ames_pch) +
  geom_rect(
    aes(
      xmin = dot_rr_x[1],
      xmax = dot_rr_x[2],
      ymin = dot_rr_y[1],
      ymax = dot_rr_y[2]
    ),
    fill = NA,
    color = "black"
  )


dot_rr_with_space <- dot_rr_y
dot_rr_with_space[1] <- dot_rr_y[1] -  diff(dot_rr_y)
dot_rr_ratio <- diff(dot_rr_x)/diff(dot_rr_with_space)

dot_rr <- 
  ggplot() +
  xlim(dot_rr_x) +
  ylim(dot_rr_with_space) +
  theme_void() + 
  theme(legend.position = "none") +
  geom_sf(data = ia_roads, aes(geometry = geometry), alpha = .3) +
  geom_point(
    data = ames %>% filter(Neighborhood %in% c("Iowa_DOT_and_Rail_Road")),
    aes(
      x = Longitude,
      y = Latitude,
      col = Neighborhood,
      shape = Neighborhood
    ),
    size = 6, 
    alpha = .5
  ) + 
  scale_color_manual(values = ames_cols) + 
  scale_shape_manual(values = ames_pch)

# guide inset in plot
agg_png("dot_rr.png", width = 480 * dot_rr_ratio)
print(dot_rr)
print(dot_rr_box, vp = viewport(0.5, 0.26, width = 0.45 * ames_ratio, height = 0.45))
dev.off()


## -----------------------------------------------------------------------------

crawford_x <- extendrange(ames$Longitude[ames$Neighborhood == "Crawford"], f = .1)
crawford_y <- extendrange(ames$Latitude[ames$Neighborhood == "Crawford"], f = .1)

crawford_box <- 
  ggplot() +
  xlim(extendrange(ames$Longitude)) +
  ylim(extendrange(ames$Latitude)) + 
  theme_void() + 
  theme(legend.position = "none", plot.title = element_text(hjust = 0.5),
        plot.background = element_rect(linetype = 'solid', colour = 'black', size = 1)) +
  geom_sf(data = ia_roads, aes(geometry = geometry), alpha = .1) +
  geom_point(
    data = ames %>% filter(Neighborhood %in% c("Crawford")),
    aes(
      x = Longitude,
      y = Latitude,
      col = Neighborhood,
      shape = Neighborhood
    ),
    size = .3, 
    alpha = .5
  ) + 
  scale_color_manual(values = ames_cols) + 
  scale_shape_manual(values = ames_pch) +
  geom_rect(
    aes(
      xmin = crawford_x[1],
      xmax = crawford_x[2],
      ymin = crawford_y[1],
      ymax = crawford_y[2]
    ),
    fill = NA,
    color = "black"
  )


crawford_with_space <- crawford_y
crawford_with_space[1] <- crawford_y[1] -  diff(crawford_y) * .4
crawford_ratio <- diff(crawford_x)/diff(crawford_with_space)

crawford <- 
  ggplot() +
  xlim(crawford_x) +
  ylim(crawford_with_space) +
  theme_void() + 
  theme(legend.position = "none") +
  geom_sf(data = ia_roads, aes(geometry = geometry), alpha = .3) +
  geom_point(
    data = ames %>% filter(Neighborhood %in% c("Crawford")),
    aes(
      x = Longitude,
      y = Latitude,
      col = Neighborhood,
      shape = Neighborhood
    ),
    size = 5, 
    alpha = .5
  ) + 
  scale_color_manual(values = ames_cols) + 
  scale_shape_manual(values = ames_pch)

# guide inset in plot
agg_png("crawford.png", width = 480 * crawford_ratio)
print(crawford)
print(crawford_box, vp = viewport(0.5, 0.2, width = 0.35 * ames_ratio, height = 0.35))
dev.off()

## -----------------------------------------------------------------------------


northridge_x <- extendrange(ames$Longitude[ames$Neighborhood %in% c("Northridge", "Somerset")], f = .1)
northridge_y <- extendrange(ames$Latitude[ames$Neighborhood %in% c("Northridge", "Somerset")], f = .1)

northridge_box <- 
  ggplot() +
  xlim(extendrange(ames$Longitude)) +
  ylim(extendrange(ames$Latitude)) + 
  theme_void() + 
  theme(legend.position = "none", plot.title = element_text(hjust = 0.5),
        plot.background = element_rect(linetype = 'solid', colour = 'black', size = 1)) +
  geom_sf(data = ia_roads, aes(geometry = geometry), alpha = .1) +
  geom_point(
    data = ames %>% filter(Neighborhood %in% c("Northridge", "Somerset")),
    aes(
      x = Longitude,
      y = Latitude,
      col = Neighborhood,
      shape = Neighborhood
    ),
    size = .3, 
    alpha = .5
  ) + 
  scale_color_manual(values = ames_cols) + 
  scale_shape_manual(values = ames_pch) +
  geom_rect(
    aes(
      xmin = northridge_x[1],
      xmax = northridge_x[2],
      ymin = northridge_y[1],
      ymax = northridge_y[2]
    ),
    fill = NA,
    color = "black"
  )


northridge_with_space <- northridge_x
northridge_with_space[2] <- northridge_x[2] +  diff(northridge_x) * .3
northridge_ratio <- diff(northridge_with_space)/diff(northridge_y)

northridge <- 
  ggplot() +
  xlim(northridge_with_space) +
  ylim(northridge_y) +
  theme_void() + 
  theme(
    legend.position = c(.82, .5),
    legend.background = element_rect(
      fill = "white",
      colour = NA,
      size = 3
    ),
    legend.text  = element_text(size = rel(2)),
    legend.title = element_text(size = rel(2))
  ) +
  geom_sf(data = ia_roads, aes(geometry = geometry), alpha = .3) +
  geom_point(
    data = ames %>% filter(Neighborhood %in% c("Northridge", "Somerset")),
    aes(
      x = Longitude,
      y = Latitude,
      col = Neighborhood,
      shape = Neighborhood
    ),
    size = 4, 
    alpha = .5
  ) + 
  scale_color_manual(values = ames_cols) + 
  scale_shape_manual(values = ames_pch)


# make plot and guide side-by-side
# northridge_box +  plot_spacer() + northridge +  plot_layout(widths = c(2, 0.1, 3))

# guide inset in plot
agg_png("northridge.png", width = 480 * northridge_ratio)
print(northridge)
print(northridge_box, vp = viewport(0.85, 0.21, width = 0.35 * ames_ratio, height = 0.35))
dev.off()
