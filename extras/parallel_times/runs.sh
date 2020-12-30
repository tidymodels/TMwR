#!/bin/sh

R CMD BATCH --vanilla everything_05_03_expensive.R
sleep 20
R CMD BATCH --vanilla everything_05_10_expensive.R
sleep 20
R CMD BATCH --vanilla resamples_05_04_without.R
sleep 20
R CMD BATCH --vanilla resamples_05_03_without.R
sleep 20
R CMD BATCH --vanilla everything_05_04_without.R
sleep 20
R CMD BATCH --vanilla everything_05_01_expensive.R
sleep 20
R CMD BATCH --vanilla everything_05_02_with.R
sleep 20
R CMD BATCH --vanilla everything_05_03_without.R
sleep 20
R CMD BATCH --vanilla everything_05_15_without.R
sleep 20
R CMD BATCH --vanilla everything_05_20_with.R
sleep 20
R CMD BATCH --vanilla resamples_05_01_with.R
sleep 20
R CMD BATCH --vanilla resamples_05_05_expensive.R
sleep 20
R CMD BATCH --vanilla everything_05_15_with.R
sleep 20
R CMD BATCH --vanilla everything_05_10_without.R
sleep 20
R CMD BATCH --vanilla resamples_05_02_without.R
sleep 20
R CMD BATCH --vanilla resamples_05_02_expensive.R
sleep 20
R CMD BATCH --vanilla everything_05_20_without.R
sleep 20
R CMD BATCH --vanilla resamples_05_04_with.R
sleep 20
R CMD BATCH --vanilla everything_05_01_with.R
sleep 20
R CMD BATCH --vanilla resamples_05_03_with.R
sleep 20
R CMD BATCH --vanilla resamples_05_05_without.R
sleep 20
R CMD BATCH --vanilla everything_05_10_with.R
sleep 20
R CMD BATCH --vanilla resamples_05_01_without.R
sleep 20
R CMD BATCH --vanilla everything_05_05_with.R
sleep 20
R CMD BATCH --vanilla everything_05_05_without.R
sleep 20
R CMD BATCH --vanilla everything_05_03_with.R
sleep 20
R CMD BATCH --vanilla everything_05_05_expensive.R
sleep 20
R CMD BATCH --vanilla resamples_05_01_expensive.R
sleep 20
R CMD BATCH --vanilla everything_05_02_without.R
sleep 20
R CMD BATCH --vanilla everything_05_15_expensive.R
sleep 20
R CMD BATCH --vanilla everything_05_02_expensive.R
sleep 20
R CMD BATCH --vanilla resamples_05_02_with.R
sleep 20
R CMD BATCH --vanilla resamples_05_03_expensive.R
sleep 20
R CMD BATCH --vanilla resamples_05_04_expensive.R
sleep 20
R CMD BATCH --vanilla resamples_05_05_with.R
sleep 20
R CMD BATCH --vanilla everything_05_20_expensive.R
sleep 20
R CMD BATCH --vanilla everything_05_04_expensive.R
sleep 20
R CMD BATCH --vanilla everything_05_04_with.R
sleep 20
R CMD BATCH --vanilla everything_05_01_without.R
