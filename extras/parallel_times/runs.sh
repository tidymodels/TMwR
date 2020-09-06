#!/bin/sh

R CMD BATCH --vanilla mlp_times_01.R
R CMD BATCH --vanilla mlp_times_10.R
R CMD BATCH --vanilla mlp_times_05.R
R CMD BATCH --vanilla mlp_times_01.R
R CMD BATCH --vanilla mlp_times_05.R
R CMD BATCH --vanilla mlp_times_10.R
R CMD BATCH --vanilla mlp_times_05.R
R CMD BATCH --vanilla mlp_times_01.R
R CMD BATCH --vanilla mlp_times_10.R
R CMD BATCH --vanilla collect_times.R
